"""Sequential GWO then DE campaign.

Usage (from the repository root):
    python -m gwo_de_sequential.campaign --out gwo_de_sequential/results/campaign_demo

Defaults: mknapcb1..9, indices 0 1 2, seeds 1..31, four policies.
That is 9 x 3 x 31 x 4 = 3348 runs. LB2, DDTW, Kirito off.
Every generation is a full greedy GWO sweep followed by a full greedy DE sweep.
There is no competitive mode and no resume.

Preflight runs before the first write. ``--dry-run`` stops after that check.
``--out`` must not already exist. The default folder is a new
``gwo_de_sequential/results/campaign_<timestamp>``. A failed run stops the
campaign; finished rows stay on disk and a second launch against the same
folder is rejected.

Layout:
    campaign_config.json
    summary.csv                 one metrics row per run
    statistics.csv              one aggregate row per instance and policy
    <stem>/inst_<NN>/runs.csv
    <stem>/inst_<NN>/statistics.csv
    <stem>/inst_<NN>/convergence_comparison.png/.pdf
    <stem>/inst_<NN>/<policy>/best_run/result.json, trace.csv, fitness.png

Traces and plots are kept only for the best run of each policy. A fitness tie
keeps the lower seed. Metrics are kept for every run.
"""

import argparse
import copy
import csv
from dataclasses import asdict
from datetime import datetime
import json
import os
from pathlib import Path
import statistics
import sys
import time

from . import config
from .control import run_search
from .plots import save_best_trace, save_comparison
from .policies import POLICIES
from .problem import load_instance, parse_instances

# Reuses the repository monitor configuration type. Does not import gwo_de_dtw.
from dtw_stagnation import StagnationConfig

FILES = tuple(f"mknapcb{i}.txt" for i in range(1, 10))
INDICES = (0, 1, 2)
SEEDS = tuple(range(1, 32))
RESULTS_DIR = Path("gwo_de_sequential") / "results"
CONFIG_FILE = "campaign_config.json"
RUN_FIELDS = ["file", "index", "policy", "seed", "status", "best_fitness", "known_optimum",
              "gap_pct", "evaluations", "generations", "changes", "elapsed_s"]
STAT_FIELDS = ["file", "index", "policy", "runs", "best_fitness", "worst_fitness",
               "mean_fitness", "median_fitness", "std_fitness", "best_seed",
               "known_optimum", "best_gap_pct", "mean_gap_pct", "mean_elapsed_s"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m gwo_de_sequential.campaign",
        description="Sequential GWO then DE campaign: full greedy GWO sweep, then full "
                    "greedy in-place DE sweep, four policies. Metrics for every run; "
                    "traces and plots only for the best seed of each policy.",
        allow_abbrev=False,
    )
    parser.add_argument("--instances-dir", type=Path, default=Path(config.INSTANCE).parent)
    parser.add_argument("--files", nargs="+", default=list(FILES),
                        help="basenames inside --instances-dir (default: mknapcb1.txt .. 9)")
    parser.add_argument("--indices", nargs="+", type=int, default=list(INDICES))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--binarization", choices=("lb2", "sigmoid"), default=config.BINARIZATION)
    parser.add_argument("--signal", choices=("dtw", "ddtw"),
                        default="ddtw" if config.USE_DDTW else "dtw")
    parser.add_argument("--window", type=int, default=config.WINDOW)
    parser.add_argument("--band", type=int, default=config.BAND)
    parser.add_argument("--plateau-max", type=int, default=config.PLATEAU_MAX)
    parser.add_argument("--patience", type=int, default=config.PATIENCE)
    parser.add_argument("--pop", type=int, default=config.POP_SIZE)
    parser.add_argument("--iterations", type=int, default=config.ITERATIONS)
    parser.add_argument("--g1", type=float, default=config.G1)
    parser.add_argument("--g2", type=float, default=config.G2)
    parser.add_argument("--g3-exploit", type=float, default=config.EXPLOIT_G3)
    parser.add_argument("--g3-explore", type=float, default=config.EXPLORE_G3)
    parser.add_argument("--dtw-kirito", action=argparse.BooleanOptionalAction,
                        default=config.DTW_KIRITO,
                        help="Kirito rolling control for the binary policies (default: off)")
    parser.add_argument("--out", type=Path, default=None,
                        help="new campaign folder; it must not already exist "
                             "(default: gwo_de_sequential/results/campaign_<timestamp>)")
    parser.add_argument("--dry-run", action="store_true",
                        help="validate the selection and output, print the run count, "
                             "and stop without executing or writing")
    return parser


def monitor_config(args) -> StagnationConfig:
    return StagnationConfig(
        window=args.window, band=args.band, min_slope=config.MIN_SLOPE,
        plateau_max=args.plateau_max, patience=args.patience,
        use_ddtw=(args.signal == "ddtw"), adapt_thresholds=config.ADAPT_THRESHOLDS,
        p_low=config.P_LOW, p_high=config.P_HIGH, improvement_tol=config.IMPROVEMENT_TOL,
    )


def _duplicate(values):
    seen = set()
    for value in values:
        if value in seen:
            return value
        seen.add(value)
    return None


def validate(parser, args) -> None:
    """Every check before the first run or write."""
    for name in args.files:
        if name in ("", ".", "..") or Path(name).name != name or "/" in name or "\\" in name:
            parser.error(f"invalid instance file name (basename only): {name}")
    if _duplicate(args.files) is not None:
        parser.error(f"duplicate instance file: {_duplicate(args.files)}")
    if _duplicate(args.indices) is not None:
        parser.error(f"duplicate index: {_duplicate(args.indices)}")
    for seed in args.seeds:
        if seed < 0:
            parser.error(f"seeds must be nonnegative: {seed}")
    if _duplicate(args.seeds) is not None:
        parser.error(f"duplicate seed: {_duplicate(args.seeds)}")
    if args.pop < 4:
        parser.error("--pop must be >= 4 (DE/rand/1 needs three other individuals)")
    for option in ("iterations", "window", "plateau_max", "patience"):
        if getattr(args, option) <= 0:
            parser.error(f"--{option.replace('_', '-')} must be positive")
    if args.band < 0:
        parser.error("--band must be nonnegative")
    if args.binarization == "lb2" and args.g2 >= config.V_MAX:
        parser.error(f"--g2 must be below v_max={config.V_MAX:g} "
                     "(otherwise the LB2 slope changes sign)")
    for name in args.files:
        path = args.instances_dir / name
        if not path.is_file():
            parser.error(f"instance file not found: {path}")
        try:
            count = len(parse_instances(path.read_text(encoding="utf-8", errors="replace")))
        except (OSError, ValueError, IndexError) as exc:
            parser.error(f"cannot read instance file {path}: {exc}")
        for index in args.indices:
            if not 0 <= index < count:
                parser.error(f"index {index} out of range for {name} ({count} instances)")
    if args.out is not None and args.out.exists():
        parser.error(f"output path already exists: {args.out}")


def new_default_root() -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    n = 1
    while True:
        root = RESULTS_DIR / (f"campaign_{stamp}" if n == 1 else f"campaign_{stamp}_{n}")
        try:
            root.mkdir()
            return root
        except FileExistsError:
            n += 1


def _blank(value):
    return "" if value is None else value


def _start_csv(path: Path, fields) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=fields).writeheader()


def _append_csv(paths, fields, rows) -> None:
    for path in paths:
        with open(path, "a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore").writerows(rows)


def _write_json(path: Path, data) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=1, default=_jsonable), encoding="utf-8")
    os.replace(temporary, path)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        return value.tolist()
    if hasattr(value, "item") and type(value).__module__ == "numpy":
        return value.item()
    return value


def is_better(record, best) -> bool:
    """Higher fitness wins. A tie keeps the lower seed, whatever the run order."""
    if best is None:
        return True
    if record["best_fitness"] != best["best_fitness"]:
        return record["best_fitness"] > best["best_fitness"]
    return record["seed"] < best["seed"]


def policy_statistics(name, index, policy, rows, best) -> dict:
    fitness = [float(row["best_fitness"]) for row in rows]
    gaps = [float(row["gap_pct"]) for row in rows if row["gap_pct"] != ""]
    return {
        "file": name, "index": index, "policy": policy, "runs": len(rows),
        "best_fitness": max(fitness), "worst_fitness": min(fitness),
        "mean_fitness": statistics.mean(fitness),
        "median_fitness": statistics.median(fitness),
        "std_fitness": statistics.stdev(fitness) if len(fitness) > 1 else "",
        "best_seed": best["seed"],
        "known_optimum": rows[0]["known_optimum"],
        "best_gap_pct": min(gaps) if gaps else "",
        "mean_gap_pct": statistics.mean(gaps) if gaps else "",
        "mean_elapsed_s": statistics.mean(float(row["elapsed_s"]) for row in rows),
    }


def _header(args, total: int) -> str:
    text = (f"campaign files={len(args.files)} indices={len(args.indices)} "
            f"seeds={len(args.seeds)} policies={len(POLICIES)} runs={total} "
            f"binarization={args.binarization} signal={args.signal} W={args.window} "
            f"pop={args.pop} iterations={args.iterations} schedule={config.SCHEDULE}")
    if args.dtw_kirito:
        text += " dtw_kirito=True"
    return text


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate(parser, args)
    total = len(args.files) * len(args.indices) * len(args.seeds) * len(POLICIES)
    print(_header(args, total))
    if args.dry_run:
        print("dry-run: selection and output validated, nothing executed or written")
        return 0

    cfg = monitor_config(args)
    root = new_default_root() if args.out is None else args.out
    if args.out is not None:
        root.mkdir(parents=True)
    state = {
        "status": "running", "total_runs": total, "completed_runs": 0, "error": None,
        "started": datetime.now().isoformat(timespec="seconds"), "finished": None,
        "selection": {"instances_dir": str(args.instances_dir), "files": list(args.files),
                      "indices": list(args.indices), "seeds": list(args.seeds),
                      "policies": list(POLICIES)},
        "config": {"binarization": args.binarization, "signal": args.signal,
                   "pop": args.pop, "iterations": args.iterations,
                   "g1": args.g1, "g2": args.g2, "g3_exploit": args.g3_exploit,
                   "g3_explore": args.g3_explore, "dtw_kirito": bool(args.dtw_kirito),
                   "schedule": config.SCHEDULE, "monitor_config": asdict(cfg)},
    }
    config_path = root / CONFIG_FILE
    _write_json(config_path, state)
    summary = root / "summary.csv"
    overall = root / "statistics.csv"
    _start_csv(summary, RUN_FIELDS)
    _start_csv(overall, STAT_FIELDS)

    where = None
    try:
        for name in args.files:
            for index in args.indices:
                folder = root / Path(name).stem / f"inst_{index:02d}"
                folder.mkdir(parents=True)
                runs = folder / "runs.csv"
                _start_csv(runs, RUN_FIELDS)
                inst = load_instance(args.instances_dir / name, index)
                instance = f"{name}[{index}]"
                rows = {policy: [] for policy in POLICIES}
                best = {policy: None for policy in POLICIES}
                for seed in args.seeds:
                    for policy in POLICIES:
                        where = f"{instance} policy={policy} seed={seed}"
                        row = {"file": name, "index": index, "policy": policy, "seed": seed,
                               "status": "failed"}
                        start = time.perf_counter()
                        try:
                            result = run_search(
                                inst, cfg, pop_size=args.pop, iterations=args.iterations,
                                seed=seed, G1=args.g1, G2=args.g2,
                                g3_exploit=args.g3_exploit, g3_explore=args.g3_explore,
                                binarization=args.binarization, policy=policy,
                                dtw_kirito=args.dtw_kirito)
                        except BaseException:
                            _append_csv((runs, summary), RUN_FIELDS, [row])
                            raise
                        elapsed = time.perf_counter() - start
                        optimum = result["known_optimum"]
                        row.update(status="ok", best_fitness=result["best_fitness"],
                                   known_optimum=optimum if optimum and optimum > 0 else "",
                                   gap_pct=_blank(result["gap_pct"]),
                                   evaluations=result["evaluations"],
                                   generations=result["generations"],
                                   changes=len(result["changes"]),
                                   elapsed_s=round(elapsed, 4))
                        _append_csv((runs, summary), RUN_FIELDS, [row])
                        rows[policy].append(row)
                        if is_better(result, best[policy]):
                            best[policy] = copy.deepcopy(result)
                        state["completed_runs"] += 1
                        _write_json(config_path, state)
                        gap = result["gap_pct"]
                        print(f"[{state['completed_runs']}/{total}] {instance} seed={seed} "
                              f"policy={policy} best={result['best_fitness']:.0f} "
                              f"gap={f'{gap:.2f}%' if gap is not None else 'n/a'} "
                              f"elapsed={elapsed:.2f}s")
                where = f"{instance} export"
                stats = [policy_statistics(name, index, policy, rows[policy], best[policy])
                         for policy in POLICIES]
                winners = []
                for policy in POLICIES:
                    saved = best[policy]
                    saved["instance"] = instance
                    saved["signal"] = args.signal
                    run_dir = folder / policy / "best_run"
                    run_dir.mkdir(parents=True)
                    _write_json(run_dir / "result.json", saved)
                    save_best_trace(saved, run_dir)
                    winners.append(saved)
                save_comparison(winners, folder, seed_labels=True)
                _start_csv(folder / "statistics.csv", STAT_FIELDS)
                _append_csv((folder / "statistics.csv", overall), STAT_FIELDS, stats)
    except BaseException as exc:
        interrupted = isinstance(exc, KeyboardInterrupt)
        state["status"] = "interrupted" if interrupted else "failed"
        state["error"] = f"{where}: {type(exc).__name__}: {exc}"
        state["finished"] = datetime.now().isoformat(timespec="seconds")
        _write_json(config_path, state)
        print(f"status={state['status']} runs={state['completed_runs']}/{total}")
        print(f"output={root}")
        if not isinstance(exc, Exception):
            raise
        print(f"error: campaign failed at {state['error']}", file=sys.stderr)
        return 1

    state["status"] = "completed"
    state["finished"] = datetime.now().isoformat(timespec="seconds")
    _write_json(config_path, state)
    print(f"status=completed runs={state['completed_runs']}/{total}")
    print(f"output={root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
