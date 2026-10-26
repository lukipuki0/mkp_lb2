"""Run adaptive MH with the original hybrid_mkp iteration pipeline and outputs."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import re
import sys
import traceback

import numpy as np

from .config import RunConfig, monitor_with_cli_overrides, POPULATION_POOL, TRAJECTORY_POOL
from .exports import write_json, write_run, write_batch, write_csv
from .orchestrator import run_pipeline
from hybrid_mkp.mkp_core.data_loader import parsear_instancias
from hybrid_mkp.mkp_core.problem import MKPInstance

ROOT = Path(__file__).resolve().parents[1]


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def seed_value(value):
    number = int(value)
    if not 0 <= number < 2**32:
        raise argparse.ArgumentTypeError("seed must fit an unsigned 32-bit integer")
    return number


def environment():
    paths = list((ROOT / "adaptive_mh_selection").glob("*.py"))
    for folder in ("hybrid_mkp", "mkp_core", "lb2"):
        paths += list((ROOT / folder).rglob("*.py"))
    paths.append(ROOT / "dtw_stagnation.py")
    versions = {}
    for package in ("numpy", "scipy", "matplotlib", "requests"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {"python": sys.version, "platform": platform.platform(), "dependencies": versions,
            "source_sha256": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in sorted(paths)}, "execution": "sequential; not thread-safe"}


def parser_for_run():
    study = RunConfig()
    monitor = study.monitor
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", type=Path, default=ROOT / "instancias/mknapcb1.txt", help="Local OR-Library MKP file")
    parser.add_argument("--index", type=int, default=0, help="Zero-based instance index (default: 0)")
    seeds = parser.add_mutually_exclusive_group()
    seeds.add_argument("--seed", type=seed_value, default=study.seed, help="First run seed (default: %(default)s)")
    seeds.add_argument("--seeds", type=seed_value, nargs="+", help="Explicit distinct seeds; replaces --seed/--runs expansion")
    parser.add_argument("--runs", type=positive, default=1, help="Independent runs, seed+i (default: 1)")
    parser.add_argument("--iterations", "--max-iters", dest="iterations", type=positive, default=study.max_iters,
                        help="Native accumulated iterations (default: %(default)s); finishes the last epoch")
    parser.add_argument("--max-epoch-iters", type=positive, help="Optional native epoch cap; unset preserves original epoch lengths")
    parser.add_argument("--selectors", nargs="+", choices=("uniform", "adaptive", "adaptive_exploratory"),
                        default=[study.selector], help="Selection policies; adaptive_exploratory is experimental opt-in")
    parser.add_argument("--population-pool", nargs="+", metavar="MH",
                        help="Nonempty unique population subset, in selection order (default: "
                             + " ".join(POPULATION_POOL) + ")")
    parser.add_argument("--trajectory-pool", nargs="+", metavar="MH",
                        help="Nonempty unique trajectory subset, in selection order (default: "
                             + " ".join(TRAJECTORY_POOL) + ")")
    parser.add_argument("--modes", nargs="+", choices=("dtw", "ddtw"),
                        default=["ddtw" if study.use_ddtw else "dtw"])
    parser.add_argument("--alpha", type=float, default=study.alpha, help="EWMA learning rate (default: %(default)s)")
    parser.add_argument("--epsilon", type=float, default=study.epsilon, help="Exploration mixture: uniform for adaptive, inverse-sqrt uses for adaptive_exploratory (default: %(default)s)")
    parser.add_argument("--window", type=positive, default=monitor.window)
    parser.add_argument("--min-slope", type=float, default=monitor.min_slope,
                        help="Configured ramp slope (default: %(default)s)")
    parser.add_argument("--plateau-max", type=positive, default=monitor.plateau_max)
    parser.add_argument("--patience", type=positive, default=monitor.patience, help="Existing per-MH patience minima still apply")
    parser.add_argument("--reference-value", type=float, help="Known positive reporting reference; not used by selector")
    parser.add_argument("--selection-trace", action="store_true",
                        help="Also save choice/reward CSV. The conditional-probability figure does not depend on this flag.")
    parser.add_argument("--output", type=Path, help="NEW campaign directory; existing paths are refused")
    parser.add_argument("--no-plots", action="store_true", help="Write native CSV/TXT/JSON reports without PNGs")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print protocol without running or writing files")
    return parser


def main(argv=None):
    parser = parser_for_run()
    args = parser.parse_args(argv)
    try:
        if args.index < 0:
            raise ValueError("index must be nonnegative")
        if args.seeds and args.runs != 1:
            raise ValueError("Use --seeds or --runs, not both")
        seeds = args.seeds or [args.seed + i for i in range(args.runs)]
        for name, values in (("seeds", seeds), ("selectors", args.selectors), ("modes", args.modes)):
            if len(values) != len(set(values)):
                raise ValueError(f"{name} must be unique")
        if args.reference_value is not None and (not math.isfinite(args.reference_value) or args.reference_value <= 0):
            raise ValueError("reference-value must be finite and positive")
        if not args.instance.is_file():
            raise ValueError("instance must identify an existing local file")
        output = args.output or ROOT / "resultados/adaptive_mh_selection" / datetime.now(timezone.utc).strftime("run_iterations_%Y%m%d_%H%M%S_%f")
        if output.exists() or output.is_symlink():
            raise ValueError(f"Output already exists: {output}; use a new directory")
        data = args.instance.read_bytes()
        instances = parsear_instancias(data.decode("utf-8"))
        if args.index >= len(instances):
            raise ValueError(f"index must be less than {len(instances)}")
        raw = instances[args.index]
        if raw["n"] < 1 or raw["m"] < 1 or not all(np.all(np.isfinite(raw[k])) for k in ("p", "r", "b")):
            raise ValueError("Instance dimensions and numeric data must be valid and finite")
        reference_source = "file header; zero means unknown"
        if args.reference_value is not None:
            raw["valor_optimo"] = args.reference_value
            reference_source = "explicit override"
        elif float(raw["valor_optimo"]) == 0:
            from .best_known import lookup_best_feasible, reference_source as best_known_source
            match = re.fullmatch(r"mknapcb([1-9])\.txt", args.instance.name)
            digest = hashlib.sha256(data).hexdigest()
            known = (lookup_best_feasible(int(match.group(1)), args.index, digest)
                     if match else None)
            if known is not None:
                raw["valor_optimo"] = known
                reference_source = best_known_source()
        instance = MKPInstance.from_dict(raw)
        monitor = monitor_with_cli_overrides(args.window, args.min_slope, args.plateau_max, args.patience)
        configs = [RunConfig(seed=seed, selector=selector, max_iters=args.iterations,
                             max_epoch_iters=args.max_epoch_iters, alpha=args.alpha,
                             epsilon=args.epsilon, use_ddtw=mode == "ddtw", monitor=monitor,
                             population_pool=args.population_pool, trajectory_pool=args.trajectory_pool)
                   for mode in args.modes for selector in args.selectors for seed in seeds]
        protocol = {"schema_version": 2, "budget_unit": "native_iterations", "instance_file": str(args.instance.resolve()),
                    "index": args.index, "instance_dimensions": {"n": instance.n, "m": instance.m},
                    "instance_sha256": hashlib.sha256(data).hexdigest(), "reference_value": float(instance.valor_optimo),
                    "reference_source": reference_source,
                    "runs": [asdict(config) for config in configs], "plots": not args.no_plots,
                    "selection_trace": args.selection_trace,
                    "stopping": "Same native epoch-boundary stopping; report requested and actual iterations separately"}
    except (ValueError, TypeError, IndexError, OSError, OverflowError) as error:
        parser.error(str(error))
    if args.dry_run:
        print(json.dumps(protocol, sort_keys=True, allow_nan=False))
        return 0
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "protocol.json", protocol)
    write_json(output / "environment.json", environment())
    successes = failures = artifact_failures = 0
    name = f"{args.instance.stem}_inst{args.index:02d}"
    for mode in args.modes:
        for selector in args.selectors:
            condition = f"{selector}_{mode}"
            directory = output / condition
            directory.mkdir()
            results = []
            selected = [c for c in configs if c.selector == selector and c.use_ddtw == (mode == "ddtw")]
            for run_index, config in enumerate(selected, 1):
                folder = directory / f"run_{run_index:02d}"
                folder.mkdir()
                write_json(folder / "config.json", config)
                try:
                    result = run_pipeline(instance, config)
                    result["summary"].update(condition=condition, run=run_index, instance_name=name)
                    write_run(result, folder, name=name, instance=instance)
                    if args.selection_trace:
                        records = result["selection_records"]
                        write_csv(folder / "selection_trace.csv", records, list(records[0]))
                    results.append(result)
                    successes += 1
                    summary = result["summary"]
                    print(f"[ok] {condition} seed={config.seed} fitness={summary['best_fitness']:g} "
                          f"iterations={summary['actual_iterations']} requested={summary['requested_iterations']} "
                          f"overshoot={summary['iteration_overshoot']}")
                except Exception as error:
                    failures += 1
                    write_json(folder / "summary.json", {"schema_version": 2, "budget_unit": "native_iterations",
                               "status": "failed", "seed": config.seed, "selector": selector,
                               "use_ddtw": config.use_ddtw, "error": f"{type(error).__name__}: {error}"})
                    (folder / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
                    print(f"[failed] {condition} seed={config.seed}: {error}", file=sys.stderr)
            if results:
                try:
                    write_batch(results, directory, name=name, instance=instance, plots=not args.no_plots)
                except Exception as error:
                    artifact_failures += 1
                    (directory / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
                    print(f"[export failed] {condition}: {error}", file=sys.stderr)
    print(f"Completed {successes} runs; failed {failures}; export failures {artifact_failures}. Campaign: {output}")
    return 1 if failures or artifact_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
