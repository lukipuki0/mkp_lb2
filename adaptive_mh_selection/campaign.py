"""Sequential grouped MKP campaign over the native iteration pipeline.

Groups come from the loaded item count (100, 250 or 500), never from the family
number and never from a 240-item assumption. Every attempt is retained as a
lightweight metric. Only the winning full solution is exported, under its
original run ordinal. Unknown header references stay unknown.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time

import numpy as np

from .config import RunConfig, monitor_with_cli_overrides
from .exports import write_csv, write_json, write_run
from .orchestrator import run_pipeline
from .run import ROOT, environment, positive, seed_value
from dtw_stagnation import StagnationConfig
from hybrid_mkp.mkp_core.data_loader import parsear_instancias
from hybrid_mkp.mkp_core.problem import MKPInstance

ALLOWED_N = (100, 250, 500)
MODE_ORDER = {"dtw": 0, "ddtw": 1}
SELECTOR_ORDER = {"adaptive": 0, "uniform": 1, "adaptive_exploratory": 2}
RUN_FIELDS = [
    "run", "seed", "status", "error", "final_fitness", "gap_pct",
    "requested_iterations", "actual_iterations", "iteration_overshoot",
    "n_blocks", "transitions", "elapsed_seconds",
]
STATUS_FIELDS = [
    "family", "file", "index", "n", "m", "condition", "relative_output_path",
    "status", "planned_runs", "succeeded_runs", "failed_runs", "winner_run",
    "winner_fitness", "elapsed_seconds", "error",
]
SUMMARY_FIELDS = [
    "family", "file", "index", "n", "m", "condition", "relative_output_path",
    "status", "planned_runs", "succeeded_runs", "failed_runs", "best_fitness",
    "mean_fitness", "std_fitness", "median_fitness", "requested_iterations",
    "actual_iterations_min", "actual_iterations_max", "elapsed_seconds", "winner_run",
]


def parser_for_campaign():
    study = RunConfig()
    monitor = study.monitor
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--families", type=int, nargs="+", default=list(range(1, 10)),
                        help="OR-Library family numbers 1..9 (default: 1 2 3 4 5 6 7 8 9)")
    parser.add_argument("--indices", type=int, nargs="+", default=[0, 1, 2],
                        help="Zero-based indices 0..29 (default: 0 1 2)")
    parser.add_argument("--runs", type=positive, default=31, help="Runs per case condition (default: 31)")
    parser.add_argument("--seed", type=seed_value, default=study.seed,
                        help="First shared seed; runs use seed..seed+runs-1 (default: %(default)s)")
    parser.add_argument("--iterations", "--max-iters", dest="iterations", type=positive, default=study.max_iters,
                        help="Native accumulated iterations (default: %(default)s); no evaluation budget")
    parser.add_argument("--max-epoch-iters", type=positive,
                        help="Optional native epoch cap for short checks; unset preserves epoch overshoot")
    parser.add_argument("--selectors", nargs="+", choices=("uniform", "adaptive", "adaptive_exploratory"),
                        default=[study.selector], help="Selection policies; adaptive_exploratory is experimental opt-in")
    parser.add_argument("--modes", nargs="+", choices=("dtw", "ddtw"),
                        default=["ddtw" if study.use_ddtw else "dtw"])
    parser.add_argument("--alpha", type=float, default=study.alpha, help="EWMA learning rate (default: %(default)s)")
    parser.add_argument("--epsilon", type=float, default=study.epsilon, help="Exploration mixture: uniform for adaptive, inverse-sqrt uses for adaptive_exploratory (default: %(default)s)")
    parser.add_argument("--window", type=positive, default=monitor.window)
    parser.add_argument("--min-slope", type=float, default=monitor.min_slope,
                        help="Configured ramp slope (default: %(default)s)")
    parser.add_argument("--plateau-max", type=positive, default=monitor.plateau_max)
    parser.add_argument("--patience", type=positive, default=monitor.patience)
    parser.add_argument("--instances-dir", type=Path, default=ROOT / "instancias",
                        help="Local instance directory; files are never downloaded")
    parser.add_argument("--output", type=Path, help="New campaign directory; existing paths are refused")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--selection-trace", action="store_true",
                        help="Save selection_trace.csv for the winning run only. The conditional-probability figure does not depend on this flag.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate, print the resolved protocol, and do not run or write")
    return parser


def unique(values, name):
    copied = list(values)
    if not copied or len(copied) != len(set(copied)):
        raise ValueError(f"{name} must be unique")
    return copied


def refuse_existing(path: Path):
    if path.exists() or path.is_symlink():
        raise ValueError(f"Output already exists: {path}; use a new directory")


def load_family(instances_dir: Path, family: int):
    path = instances_dir / f"mknapcb{family}.txt"
    if path.is_symlink() and not path.is_file():
        raise ValueError(f"missing local instance file: {path.name}")
    if not path.is_file():
        raise ValueError(f"missing local instance file: {path.name}")
    data = path.read_bytes()
    try:
        parsed = parsear_instancias(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, IndexError, TypeError, KeyError) as error:
        raise ValueError(f"invalid instance data in {path.name}: {error}") from error
    return path, hashlib.sha256(data).hexdigest(), parsed


def load_instance(path_name, parsed, index, instance_file_sha256=None):
    if index >= len(parsed):
        raise ValueError(f"index {index} is missing from {path_name}")
    raw = parsed[index]
    try:
        n, m = int(raw["n"]), int(raw["m"])
        reference = float(raw["valor_optimo"])
        profit, weights, capacity = raw["p"], raw["r"], raw["b"]
        valid_shape = len(profit) == n and tuple(weights.shape) == (m, n) and len(capacity) == m
        finite = all(np.all(np.isfinite(array)) for array in (profit, weights, capacity))
    except (TypeError, ValueError, KeyError, AttributeError) as error:
        raise ValueError(f"invalid instance data in {path_name}: {error}") from error
    if n < 1 or m < 1 or not valid_shape or not finite or not math.isfinite(reference) or reference < 0:
        raise ValueError(f"invalid instance data in {path_name}")
    reference_source = "file header; zero means unknown"
    if reference == 0 and instance_file_sha256 is not None:
        from .best_known import lookup_best_feasible, reference_source as best_known_source
        family_match = re.fullmatch(r"mknapcb([1-9])\.txt", Path(path_name).name)
        if family_match:
            known = lookup_best_feasible(int(family_match.group(1)), index, instance_file_sha256)
            if known is not None:
                reference = float(known)
                reference_source = best_known_source()
                raw = dict(raw, valor_optimo=reference)
    if n not in ALLOWED_N:
        raise ValueError(f"loaded n={n} is not an allowed group; groups are 100, 250 or 500")
    try:
        instance = MKPInstance.from_dict(raw)
    except (ValueError, IndexError, FloatingPointError) as error:
        raise ValueError(f"invalid instance data in {path_name}: {error}") from error
    if float(instance.valor_optimo) != reference:
        raise ValueError("file header reference was replaced; unknown zero must stay unknown")
    return instance, reference, reference_source


def output_paths(n, family, index, condition, *, multiple_cases, multiple_conditions):
    """Drop repeated case folders for one case, and the condition folder unless conditions are compared."""
    case = f"items_{n}/mknapcb{family}_inst_{index:02d}" if multiple_cases else ""
    if not multiple_conditions:
        return case, case
    return case, f"{case}/{condition}" if case else condition


def build_plan(args):
    families = sorted(unique(args.families, "families"))
    indices = sorted(unique(args.indices, "indices"))
    selectors = unique(args.selectors, "selectors")
    modes = unique(args.modes, "modes")
    if any(family < 1 or family > 9 for family in families):
        raise ValueError("families must be in 1..9")
    if any(index < 0 or index > 29 for index in indices):
        raise ValueError("indices must be in 0..29")
    seeds = [args.seed + offset for offset in range(args.runs)]
    if seeds[-1] >= 2**32:
        raise ValueError("seed range exceeds an unsigned 32-bit integer")
    selectors = sorted(selectors, key=lambda name: (SELECTOR_ORDER[name], name))
    modes = sorted(modes, key=lambda name: (MODE_ORDER[name], name))
    output = args.output or (ROOT / "resultados" / "adaptive_mh_selection" / datetime.now(timezone.utc).strftime(
        "campaign_%Y%m%d_%H%M%S_%f"))
    refuse_existing(output)
    if not args.instances_dir.is_dir():
        raise ValueError("instances-dir must be an existing local directory")
    monitor = monitor_with_cli_overrides(args.window, args.min_slope, args.plateau_max, args.patience)
    checked = RunConfig(seed=seeds[0], selector=selectors[0], max_iters=args.iterations,
                        max_epoch_iters=args.max_epoch_iters, alpha=args.alpha, epsilon=args.epsilon,
                        use_ddtw=modes[0] == "ddtw", monitor=monitor)
    condition_templates = [
        {"condition": f"{selector}_{mode}", "selector": selector, "mode": mode, "use_ddtw": mode == "ddtw"}
        for mode in modes for selector in selectors
    ]
    multiple_cases = len(families) * len(indices) > 1
    multiple_conditions = len(condition_templates) > 1
    cases, jobs = [], []
    for family in families:
        _path, digest, parsed = load_family(args.instances_dir, family)
        for index in indices:
            instance, reference, reference_source = load_instance(
                f"mknapcb{family}.txt", parsed, index, digest)
            relative, _condition_path = output_paths(
                instance.n, family, index, condition_templates[0]["condition"],
                multiple_cases=multiple_cases, multiple_conditions=multiple_conditions)
            conditions = []
            for template in condition_templates:
                _case_path, condition_path = output_paths(
                    instance.n, family, index, template["condition"],
                    multiple_cases=multiple_cases, multiple_conditions=multiple_conditions)
                conditions.append({**template, "relative_output_path": condition_path})
            case = {
                "family": family, "file": f"mknapcb{family}.txt", "index": index,
                "n": instance.n, "m": instance.m, "relative_output_path": relative,
                "instance_sha256": digest, "reference_value": reference,
                "reference_source": reference_source,
                "conditions": conditions,
            }
            cases.append(case)
            for condition in conditions:
                jobs.append({**case, **condition, "instance": instance, "conditions": None})
    cases.sort(key=lambda case: (case["n"], case["family"], case["index"]))
    jobs.sort(key=lambda job: (job["n"], job["family"], job["index"], MODE_ORDER[job["mode"]],
                               SELECTOR_ORDER[job["selector"]]))
    for job in jobs:
        job.pop("conditions", None)
    groups = {str(size): sum(case["n"] == size for case in cases) for size in ALLOWED_N}
    protocol = {
        "schema_version": 3, "budget_unit": "native_iterations", "kind": "grouped_mkp_campaign",
        "families": families, "indices": indices, "selectors": selectors, "modes": modes,
        "conditions": [item["condition"] for item in condition_templates],
        "seeds": seeds, "runs": len(seeds), "iterations": args.iterations,
        "max_epoch_iters": args.max_epoch_iters, "alpha": args.alpha, "epsilon": args.epsilon,
        "pop_injection_mode": checked.pop_injection_mode,
        "monitor": asdict(monitor),
        "plots": not args.no_plots, "selection_trace": args.selection_trace,
        "instances_dir": str(args.instances_dir.resolve()), "output": str(output),
        "groups": groups, "planned_cases": len(cases),
        "planned_conditions": len(jobs), "planned_runs": len(jobs) * len(seeds),
        "cases": cases,
        "grouping": "loaded n in {100, 250, 500}; m comes from the instance; family number is not a size",
        "reference_policy": "Nonzero file header takes precedence; otherwise an exact SHA-256 match to bundled mknapcb1..9 enables the corresponding OR-Library mkcbres.txt best feasible value (not a certified optimum); unmatched zero stays unknown; references are mapped by family, n, m and index, never copied from index 0.",
        "winner_rule": "strictly greater fitness replaces the incumbent; ties keep the earliest original run ordinal",
        "retention": "every attempt is in runs_resultados.csv; only the winning run is exported in full",
        "statistics": "within-case only; failures excluded; sample std ddof=1; no pooled objective; no significance test",
        "execution": "sequential; no overwrites; no retries; no parallelism; no resume",
        "stopping": "Same native epoch-boundary stopping; report requested and actual iterations separately; no implicit epoch cap",
    }
    return protocol, jobs


def bit_solution(solution, size):
    if not isinstance(solution, list) or len(solution) != size:
        raise ValueError("invalid native outcome: solution")
    bits = []
    for item in solution:
        if isinstance(item, bool) or not isinstance(item, (int, np.integer)) or int(item) not in (0, 1):
            raise ValueError("invalid native outcome: solution")
        bits.append(int(item))
    return bits


def validate_outcome(result, instance, config):
    """Accept only a finite feasible native completion with consistent iteration fields."""
    if not isinstance(result, dict):
        raise ValueError("invalid native outcome: result")
    summary, native = result.get("summary"), result.get("native_result")
    if not isinstance(summary, dict) or native is None or summary.get("status") != "success":
        raise ValueError("invalid native outcome: status")
    fitness = summary.get("best_fitness")
    if isinstance(fitness, bool) or not isinstance(fitness, (int, float)) or not math.isfinite(float(fitness)):
        raise ValueError("invalid native outcome: nonfinite objective")
    bits = bit_solution(summary.get("best_solution"), instance.n)
    if not instance.es_factible(bits):
        raise ValueError("invalid native outcome: infeasible")
    evaluated = instance.evaluar(bits)
    if not math.isfinite(evaluated) or abs(evaluated - float(fitness)) > 1e-6:
        raise ValueError("invalid native outcome: objective mismatch")
    requested, actual, overshoot = (summary.get("requested_iterations"), summary.get("actual_iterations"),
                                    summary.get("iteration_overshoot"))
    if requested != config.max_iters or isinstance(actual, bool) or not isinstance(actual, int):
        raise ValueError("invalid native outcome: iterations")
    if isinstance(overshoot, bool) or not isinstance(overshoot, int) or overshoot != max(0, actual - requested):
        raise ValueError("invalid native outcome: overshoot")
    if actual < requested or (config.max_epoch_iters is not None and actual != requested):
        raise ValueError("invalid native outcome: iteration budget was not honored")
    history, instant = getattr(native, "historial_global", None), getattr(native, "historial_inst_global", None)
    if not isinstance(history, list) or len(history) != actual or not isinstance(instant, list) or len(instant) != actual:
        raise ValueError("invalid native outcome: history length")
    switches = list(getattr(native, "log_switches", []) or [])
    if summary.get("n_blocks") != len(switches) or sum(int(switch.n_iters) for switch in switches) != actual:
        raise ValueError("invalid native outcome: blocks")
    gap = summary.get("gap_pct")
    if float(instance.valor_optimo) == 0:
        if gap is not None:
            raise ValueError("invalid native outcome: unknown reference must have null gap")
    elif isinstance(gap, bool) or not isinstance(gap, (int, float)) or not math.isfinite(float(gap)):
        raise ValueError("invalid native outcome: gap")
    summary["best_fitness"] = float(fitness)
    summary["best_solution"] = bits
    return summary


def make_config(protocol, seed, selector, mode):
    """Rebuild a run from the full stored monitor. The selected mode owns use_ddtw."""
    return RunConfig(seed=seed, selector=selector, max_iters=protocol["iterations"],
                     max_epoch_iters=protocol["max_epoch_iters"], alpha=protocol["alpha"],
                     epsilon=protocol["epsilon"], use_ddtw=mode == "ddtw",
                     pop_injection_mode=protocol.get("pop_injection_mode", "mixed"),
                     monitor=StagnationConfig(**protocol["monitor"]))


def metric_row(ordinal, seed, status, *, error=None, summary=None, elapsed=None):
    row = {field: None for field in RUN_FIELDS}
    row.update(run=ordinal, seed=seed, status=status, error=error, elapsed_seconds=elapsed,
               requested_iterations=None if summary is None else summary.get("requested_iterations"))
    if summary is not None and status == "success":
        gap = summary.get("gap_pct")
        row.update(final_fitness=summary.get("best_fitness"),
                   gap_pct=float(gap) if isinstance(gap, (int, float)) and math.isfinite(gap) else None,
                   requested_iterations=summary.get("requested_iterations"),
                   actual_iterations=summary.get("actual_iterations"),
                   iteration_overshoot=summary.get("iteration_overshoot"),
                   n_blocks=summary.get("n_blocks"), transitions=summary.get("transitions"))
    return row


def apply_stats(row, metrics):
    values = [float(item["final_fitness"]) for item in metrics
              if item["status"] == "success" and item["final_fitness"] is not None]
    actuals = [int(item["actual_iterations"]) for item in metrics
               if item["status"] == "success" and item["actual_iterations"] is not None]
    row["succeeded_runs"] = sum(item["status"] == "success" for item in metrics)
    row["failed_runs"] = sum(item["status"] == "failed" for item in metrics)
    if not values:
        row.update(best_fitness=None, mean_fitness=None, std_fitness=None, median_fitness=None,
                   winner_fitness=None, actual_iterations_min=None, actual_iterations_max=None)
        return
    row.update(best_fitness=max(values), mean_fitness=float(np.mean(values)),
               std_fitness=float(np.std(values, ddof=1)) if len(values) > 1 else None,
               median_fitness=float(np.median(values)),
               actual_iterations_min=min(actuals) if actuals else None,
               actual_iterations_max=max(actuals) if actuals else None)


def display(value):
    if value is None or value == "":
        return "--"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def write_summaries(output: Path, rows):
    write_csv(output / "resumen_global.csv", rows, SUMMARY_FIELDS)
    lines = [
        "Grouped MKP campaign",
        "Within-case summaries only. Failures are excluded and are not imputed as zero. No pooled raw objective and no significance test.",
        "std_fitness is the within-case sample standard deviation (ddof=1).",
        "",
        "\t".join(SUMMARY_FIELDS),
    ]
    lines.extend("\t".join(display(row.get(field)) for field in SUMMARY_FIELDS) for row in rows)
    (output / "resumen_global.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for size in ALLOWED_N:
        selected = [row for row in rows if int(row["n"]) == size]
        if not selected:
            continue
        folder = output / f"items_{size}"
        folder.mkdir(parents=True, exist_ok=True)
        write_csv(folder / "resumen_grupo.csv", selected, SUMMARY_FIELDS)
        group_lines = [f"Group items_{size}", lines[1], lines[2], "", lines[4]]
        group_lines.extend("\t".join(display(row.get(field)) for field in SUMMARY_FIELDS) for row in selected)
        (folder / "resumen_grupo.txt").write_text("\n".join(group_lines) + "\n", encoding="utf-8")


def write_case_report(directory: Path, row):
    """Native-style case TXT, including incomplete/failed case status."""
    values = [
        ("Family", f"mknapcb{row['family']}"), ("Instance index", row["index"]),
        ("Items", row["n"]), ("Constraints", row["m"]), ("Condition", row["condition"]),
        ("Status", row["status"]), ("Planned runs", row["planned_runs"]),
        ("Successful runs", row["succeeded_runs"]), ("Failed runs", row["failed_runs"]),
        ("Winner original run", row["winner_run"]), ("Best fitness", row.get("best_fitness")),
        ("Mean fitness", row.get("mean_fitness")), ("Sample std fitness", row.get("std_fitness")),
        ("Median fitness", row.get("median_fitness")),
        ("Requested native iterations", row["requested_iterations"]),
        ("Actual iterations minimum", row.get("actual_iterations_min")),
        ("Actual iterations maximum", row.get("actual_iterations_max")),
        ("Elapsed case seconds", row["elapsed_seconds"]), ("Error", row["error"]),
    ]
    lines = ["MKP pipeline case summary", *[f"{label}: {display(value)}" for label, value in values],
             "", "All seeds and outcomes are in runs_resultados.csv; failures are not imputed as zero.",
             "Only the winner retains full traces; unknown values are shown as --."]
    (directory / "resumen_pipeline.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def persist(output: Path, rows):
    write_csv(output / "estado_ejecucion.csv", rows, STATUS_FIELDS)
    write_summaries(output, rows)


def export_winner(protocol, job, best, metrics, directory: Path):
    ordinal, result, config = best["ordinal"], best["result"], best["config"]
    best_dir = directory / f"best_run_{ordinal:02d}"
    name = f"mknapcb{job['family']}_inst{job['index']:02d}"
    write_run(result, best_dir, name=name, instance=job["instance"], best=True)
    write_json(best_dir / "config.json", config)
    if protocol["selection_trace"] and result.get("selection_records"):
        records = result["selection_records"]
        write_csv(best_dir / "selection_trace.csv", records, list(records[0]))
    if protocol["plots"]:
        from . import plotting
        values = [item["final_fitness"] for item in metrics if item["status"] == "success"]
        plotting.plot_run(result["native_result"], best_dir, metadata=result["summary"],
                          selection_records=result.get("selection_records"))
        plotting.plot_distribution(name, values, job["instance"].valor_optimo, directory)


def run_condition(output: Path, protocol, job, row, rows):
    """Run one case condition sequentially. Full results are not accumulated."""
    directory = output / job["relative_output_path"]
    if directory != output:
        directory.mkdir(parents=True, exist_ok=False)
    metrics, best = [], None
    started = time.perf_counter()
    row.update(status="running", error=None, elapsed_seconds=0)
    write_csv(directory / "runs_resultados.csv", metrics, RUN_FIELDS)
    persist(output, rows)
    try:
        for ordinal, seed in enumerate(protocol["seeds"], 1):
            attempt = time.perf_counter()
            config = make_config(protocol, seed, job["selector"], job["mode"])
            try:
                result = run_pipeline(job["instance"], config)
                summary = validate_outcome(result, job["instance"], config)
                summary["requested_iterations"] = config.max_iters
                metrics.append(metric_row(ordinal, seed, "success", summary=summary,
                                          elapsed=time.perf_counter() - attempt))
                fitness = summary["best_fitness"]
                if best is None or fitness > best["fitness"]:
                    best = {"ordinal": ordinal, "fitness": fitness, "result": result, "config": config}
                print(f"[ok] {job['condition']} family={job['family']} index={job['index']} "
                      f"run={ordinal} seed={seed} fitness={fitness:g} "
                      f"iterations={summary['actual_iterations']} requested={summary['requested_iterations']} "
                      f"overshoot={summary['iteration_overshoot']}")
            except KeyboardInterrupt:
                metrics.append(metric_row(ordinal, seed, "interrupted", error="KeyboardInterrupt",
                                          summary={"requested_iterations": protocol["iterations"]},
                                          elapsed=time.perf_counter() - attempt))
                raise
            except Exception as error:
                metrics.append(metric_row(ordinal, seed, "failed", error=f"{type(error).__name__}: {error}",
                                          summary={"requested_iterations": protocol["iterations"]},
                                          elapsed=time.perf_counter() - attempt))
                print(f"[failed] {job['condition']} family={job['family']} index={job['index']} "
                      f"run={ordinal} seed={seed}: {error}", file=sys.stderr)
            write_csv(directory / "runs_resultados.csv", metrics, RUN_FIELDS)
            apply_stats(row, metrics)
            row["elapsed_seconds"] = time.perf_counter() - started
            row["status"] = "running"
            write_csv(output / "estado_ejecucion.csv", rows, STATUS_FIELDS)
        if best is None:
            row.update(status="failed", winner_run=None, winner_fitness=None,
                       error=next((item["error"] for item in metrics if item["error"]), "all runs failed"))
        else:
            try:
                export_winner(protocol, job, best, metrics, directory)
            except Exception as error:
                row.update(status="export_failed", winner_run=best["ordinal"], winner_fitness=best["fitness"],
                           error=f"{type(error).__name__}: {error}")
                print(f"[export failed] {job['condition']}: {error}", file=sys.stderr)
            else:
                failed = any(item["status"] == "failed" for item in metrics)
                row.update(status="partial_failure" if failed else "complete",
                           winner_run=best["ordinal"], winner_fitness=best["fitness"],
                           error=next((item["error"] for item in metrics if item["status"] == "failed"), None))
    except KeyboardInterrupt:
        apply_stats(row, metrics)
        row.update(status="interrupted", winner_run=None, winner_fitness=None, error="KeyboardInterrupt",
                   elapsed_seconds=time.perf_counter() - started)
        write_csv(directory / "runs_resultados.csv", metrics, RUN_FIELDS)
        write_case_report(directory, row)
        persist(output, rows)
        raise
    apply_stats(row, metrics)
    row["elapsed_seconds"] = time.perf_counter() - started
    write_case_report(directory, row)
    persist(output, rows)
    return 0 if row["status"] == "complete" else 1


def status_row(job, planned, iterations):
    return {
        "family": job["family"], "file": job["file"], "index": job["index"], "n": job["n"], "m": job["m"],
        "condition": job["condition"], "relative_output_path": job["relative_output_path"],
        "status": "pending", "planned_runs": planned, "succeeded_runs": 0, "failed_runs": 0,
        "winner_run": None, "winner_fitness": None, "elapsed_seconds": None, "error": None,
        "best_fitness": None, "mean_fitness": None, "std_fitness": None, "median_fitness": None,
        "requested_iterations": iterations, "actual_iterations_min": None, "actual_iterations_max": None,
    }


def execute(protocol, jobs):
    output = Path(protocol["output"])
    refuse_existing(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "protocol.json", protocol)
    write_json(output / "environment.json", environment())
    rows = [status_row(job, protocol["runs"], protocol["iterations"]) for job in jobs]
    persist(output, rows)
    exit_code = 0
    try:
        for job, row in zip(jobs, rows):
            exit_code = max(exit_code, run_condition(output, protocol, job, row, rows))
    except KeyboardInterrupt:
        return 1
    print(f"Campaign finished with status {exit_code}. Output: {output}")
    return exit_code


def main(argv=None):
    parser = parser_for_campaign()
    args = parser.parse_args(argv)
    try:
        protocol, jobs = build_plan(args)
    except (ValueError, TypeError, IndexError, OSError, OverflowError) as error:
        parser.error(str(error))
    if args.dry_run:
        print(json.dumps(protocol, sort_keys=True, allow_nan=False))
        return 0
    try:
        return execute(protocol, jobs)
    except KeyboardInterrupt:
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
