"""Descriptive MKP OAT campaign. Preflight rejects before any write or optimizer call.

Usage (from the repository root):
    python -m gwo_de_dtw.sensitivity.campaign --dry-run
    python -m gwo_de_dtw.sensitivity.campaign --configurations baseline window_20 \\
        --indices 29 --seeds 1 --pop 4 --iterations 2 --out <fresh-dir>
    python -m gwo_de_dtw.sensitivity.campaign --workers 2 --indices 0 15 --seeds 1 \\
        --configurations baseline --pop 4 --iterations 1 --out <fresh-dir>
"""

import argparse
import csv
from dataclasses import asdict
from datetime import datetime
import importlib
import json
import multiprocessing
import os
from pathlib import Path
import platform
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..controller import run_hybrid  # noqa: E402
from ..plots import PLOTS  # noqa: E402
from ..problem import cargar_instancia, parsear_instancias  # noqa: E402
from ..run import _to_json  # noqa: E402
from .analysis import comparison_table, save_sensitivity_plot, summarize  # noqa: E402
from .config import (  # noqa: E402
    BINARIZATION, CONFIGURATIONS, G1, G2, G3_EXPLOIT, G3_EXPLORE, HYBRID_MODE,
    INDICES, INSTANCE_FILE, ITERATIONS, POLICIES, POP_SIZE, SEEDS, oat_problems,
)

RESULTS_DIR = Path("gwo_de_dtw") / "results"
DRY_LINE = "dry-run: selection and output validated, nothing executed or written"
RUN_FIELDS = [
    "instance", "index", "policy", "configuration", "parameter", "parameter_value",
    "seed", "status", "best_fitness", "known_optimum", "gap_pct", "evaluations",
    "elapsed_s", "profile_changes",
]
STAT_FIELDS = [
    "instance", "index", "policy", "configuration", "parameter", "parameter_value",
    "runs", "best_fitness", "worst_fitness", "mean_fitness", "median_fitness",
    "std_fitness", "best_seed", "known_optimum", "best_gap_pct", "mean_gap_pct",
    "mean_elapsed_s", "mean_fitness_delta_vs_baseline",
]
COMPARISON_FIELDS = [
    "instance", "index", "policy", "parameter", "configuration", "parameter_value",
    "mean_fitness", "baseline_mean_fitness", "mean_fitness_delta_vs_baseline",
]


def build_parser() -> argparse.ArgumentParser:
    names = ", ".join(spec.name for spec in CONFIGURATIONS)
    parser = argparse.ArgumentParser(
        prog="python -m gwo_de_dtw.sensitivity.campaign",
        description="Descriptive one-at-a-time MKP sensitivity of binary_simple and "
                    "binary_complex. Configurations: " + names + ". Short smoke: "
                    "--configurations baseline window_20.",
        allow_abbrev=False,
    )
    parser.add_argument("--indices", nargs="+", type=int, default=list(INDICES),
                        help="zero-based indices in instancias/mknapcb1.txt "
                             "(default: 0 15 29)")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS),
                        help="nonnegative unique seeds (default: 1..31)")
    parser.add_argument("--pop", type=int, default=POP_SIZE,
                        help="population size, at least 4 (default: %(default)s)")
    parser.add_argument("--iterations", type=int, default=ITERATIONS,
                        help="generations per run (default: %(default)s)")
    parser.add_argument("--configurations", nargs="+", default=None,
                        help="optional subset of baseline, window_20, window_80, "
                             "window_160, band_1, band_2, band_8, p_low_20, p_low_30, "
                             "p_low_50, p_high_50, p_high_70, p_high_80, plateau_max_5, "
                             "plateau_max_10, plateau_max_30, patience_2, patience_4, "
                             "patience_16 (default: all 19)")
    parser.add_argument("--out", type=Path, default=None,
                        help="fresh or empty output directory (default: "
                             "gwo_de_dtw/results/sensitivity_<timestamp>/)")
    parser.add_argument("--dry-run", action="store_true",
                        help="validate and print the plan; do not run or write")
    parser.add_argument("--workers", type=int, default=3,
                        help="parallel instance workers, capped at the number of "
                             "selected indices (default: %(default)s). Use 1 for one "
                             "in-process instance at a time.")
    return parser


def _duplicate(values):
    seen = set()
    for value in values:
        if value in seen:
            return value
        seen.add(value)
    return None


def plan_line(indices, seeds, configurations, pop, iterations) -> str:
    names = ",".join(spec.name for spec in configurations)
    runs = len(indices) * len(seeds) * len(configurations) * len(POLICIES)
    return (
        f"sensitivity file={INSTANCE_FILE} indices={len(indices)} seeds={len(seeds)} "
        f"policies={len(POLICIES)} configurations={len(configurations)} runs={runs} "
        f"pop={pop} iterations={iterations} binarization={BINARIZATION} signal=ddtw "
        f"hybrid_mode={HYBRID_MODE} names={names}"
    )


def resolve(parser, args):
    """Reject invalid input before any directory is created or the optimizer runs."""
    if _duplicate(args.indices) is not None:
        parser.error(f"duplicate index: {_duplicate(args.indices)}")
    for seed in args.seeds:
        if seed < 0:
            parser.error(f"seeds must be nonnegative: {seed}")
    if _duplicate(args.seeds) is not None:
        parser.error(f"duplicate seed: {_duplicate(args.seeds)}")
    if args.pop < 4:
        parser.error("--pop must be >= 4")
    if args.iterations <= 0:
        parser.error("--iterations must be positive")
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    path = Path(INSTANCE_FILE)
    if not path.is_file():
        parser.error(f"instance file not found: {path}")
    try:
        count = len(parsear_instancias(path.read_text()))
    except (OSError, ValueError, IndexError) as exc:
        parser.error(f"cannot read instance file {path}: {exc}")
    for index in args.indices:
        if not 0 <= index < count:
            parser.error(f"index {index} out of range for {path.name} ({count} instances)")
    catalog = list(CONFIGURATIONS)
    by_name = {spec.name: spec for spec in catalog}
    if args.configurations is None:
        selected = catalog
    else:
        if _duplicate(args.configurations) is not None:
            parser.error(f"duplicate configuration: {_duplicate(args.configurations)}")
        for name in args.configurations:
            if name not in by_name:
                parser.error(f"unknown configuration: {name}")
        wanted = set(args.configurations)
        selected = [spec for spec in catalog if spec.name in wanted]
    baseline = catalog[0]
    for spec in selected:
        problems = oat_problems(spec, baseline)
        if problems:
            parser.error(problems[0])
    if args.out is not None:
        if args.out.exists() and not args.out.is_dir():
            parser.error(f"output path is not a directory: {args.out}")
        if args.out.is_dir() and any(args.out.iterdir()):
            parser.error(f"output directory is not empty: {args.out}")
    return selected


def new_default_root() -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    n = 1
    while True:
        root = RESULTS_DIR / (f"sensitivity_{stamp}" if n == 1 else f"sensitivity_{stamp}_{n}")
        try:
            root.mkdir()
            return root
        except FileExistsError:
            n += 1


def _write_json(path: Path, data: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def _csv_cell(value):
    return "" if value is None else value


def _append_csv(path: Path, fields, rows) -> None:
    with path.open("a", newline="", encoding="utf-8") as fh:
        csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore").writerows(
            [{key: _csv_cell(row.get(key)) for key in fields} for row in rows]
        )


def _append_jsonl(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")


def _positive(value):
    if value is None or value == "":
        return None
    number = float(value)
    return number if number > 0 else None


def _call_plot(plot, res, output_dir):
    """Close only figures this call left open. Do not close unrelated figures."""
    before = set(plt.get_fignums())
    try:
        return plot(res, output_dir)
    finally:
        for num in set(plt.get_fignums()) - before:
            plt.close(num)


def _winner_dir(root: Path, index: int, policy: str, configuration: str) -> Path:
    return root / "mknapcb1" / f"inst_{index:02d}" / policy / configuration / "best_run"


def _resumen(stats) -> str:
    lines = [
        "Descriptive OAT summary",
        "Sample SD uses n-1 and is blank when n=1. Unknown gaps are blank, not zero.",
        "Mean fitness delta uses the same instance and policy baseline. Instances are not pooled.",
        "No inferential test is reported. This does not claim robustness.",
        "Times are wall-clock seconds (elapsed_s).",
        "",
    ]
    for row in stats:
        std = "blank" if row["std_fitness"] == "" else row["std_fitness"]
        gap = "blank" if row["mean_gap_pct"] == "" else row["mean_gap_pct"]
        lines.append(
            f"mknapcb1.txt[{row['index']}] {row['policy']} {row['configuration']}: "
            f"mean={row['mean_fitness']} median={row['median_fitness']} "
            f"best={row['best_fitness']} worst={row['worst_fitness']} "
            f"sample_sd={std} mean_gap={gap} "
            f"delta_vs_baseline={row['mean_fitness_delta_vs_baseline']} "
            f"mean_elapsed_s={row['mean_elapsed_s']}"
        )
    return "\n".join(lines) + "\n"


def _export(root, winners, rows, state) -> None:
    for record in winners.values():
        folder = _winner_dir(root, record["index"], record["policy"], record["configuration"])
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "result.json").write_text(
            json.dumps(record, default=_to_json, indent=1), encoding="utf-8")
    grouped = {}
    for row in rows:
        grouped.setdefault((row["index"], row["policy"], row["configuration"]), []).append(row)
    stats = []
    for key in sorted(grouped):
        group = grouped[key]
        sample = group[0]
        item = {
            "instance": sample["instance"], "index": sample["index"],
            "policy": sample["policy"], "configuration": sample["configuration"],
            "parameter": sample["parameter"], "parameter_value": sample["parameter_value"],
            "runs": len(group), "known_optimum": sample["known_optimum"],
        }
        item.update(summarize(group))
        stats.append(item)
    compared = comparison_table(stats)
    delta = {(row["index"], row["policy"], row["configuration"]): row["mean_fitness_delta_vs_baseline"]
             for row in compared}
    for item in stats:
        item["mean_fitness_delta_vs_baseline"] = delta[(item["index"], item["policy"], item["configuration"])]
    for path, fields, table in (
        (root / "statistics.csv", STAT_FIELDS, stats),
        (root / "comparisons.csv", COMPARISON_FIELDS, compared),
    ):
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows([{key: _csv_cell(row.get(key)) for key in fields} for row in table])
    (root / "resumen.txt").write_text(_resumen(stats), encoding="utf-8")
    state["where"] = "plots"
    _write_json(root / "manifest.json", state)
    for record in winners.values():
        folder = _winner_dir(root, record["index"], record["policy"], record["configuration"])
        for plot in PLOTS:
            _call_plot(plot, record, folder)
    state["where"] = "sensitivity plots"
    plot_dir = root / "plots"
    for policy in POLICIES:
        save_sensitivity_plot(compared, policy, plot_dir)


OPTIMIZER_ENV = "GWO_DE_SENSITIVITY_OPTIMIZER"
JOB_INDEX_ENV = "GWO_DE_SENSITIVITY_JOB_INDEX"


class InstanceFailed(Exception):
    """Stops the remaining runs of one instance after a durable failure record."""


def effective_workers(requested: int, indices) -> int:
    """Never start more processes than selected instances."""
    return min(requested, len(indices))


def _optimizer():
    """In-process tests patch `run_hybrid`. Spawned children honor an import path instead."""
    spec = os.environ.get(OPTIMIZER_ENV)
    if spec:
        module_name, attr = spec.rsplit(".", 1)
        return getattr(importlib.import_module(module_name), attr)
    return run_hybrid


def _instance_dir(root: Path, index: int) -> Path:
    return root / "mknapcb1" / f"inst_{index:02d}"


def _journal_path(root: Path, index: int) -> Path:
    return _instance_dir(root, index) / "_journal" / "runs.jsonl"


def _status_path(root: Path, index: int) -> Path:
    return _instance_dir(root, index) / "_journal" / "status.json"


def _identity(index, policy, spec, seed) -> dict:
    return {
        "instance": "mknapcb1.txt", "index": index, "policy": policy,
        "configuration": spec.name, "parameter": spec.parameter,
        "parameter_value": "" if spec.value is None else spec.value,
        "seed": seed,
    }


def _where(index, policy, spec, seed) -> str:
    return f"mknapcb1.txt[{index}] policy={policy} configuration={spec.name} seed={seed}"


def _invoke(optimizer, inst, spec, policy, seed, pop, iterations):
    cfg = spec.stagnation_config()
    start = time.perf_counter()
    res = optimizer(
        inst, cfg, pop_size=pop, iterations=iterations, seed=seed,
        G1=G1, G2=G2, g3_exploit=G3_EXPLOIT, g3_explore=G3_EXPLORE,
        binarization=BINARIZATION, policy=policy, dtw_kirito=False,
        hybrid_mode=HYBRID_MODE,
    )
    return res, round(time.perf_counter() - start, 6)


def _ok_payload(identity, res, elapsed_s):
    optimum = _positive(res.get("known_optimum"))
    gap = None if res.get("gap_pct") is None or res.get("gap_pct") == "" else float(res["gap_pct"])
    compact = dict(
        identity, status="ok", best_fitness=float(res["best_fitness"]),
        known_optimum=optimum, gap_pct=gap, evaluations=res["evaluations"],
        elapsed_s=elapsed_s, profile_changes=len(res.get("changes") or []),
        error=None,
    )
    record = dict(res)
    record.update(
        instance=f"mknapcb1.txt[{identity['index']}]", signal="ddtw",
        configuration=identity["configuration"], index=identity["index"],
        parameter=identity["parameter"],
    )
    return compact, record


def _failed_payload(identity, exc):
    failed = dict(
        identity, status="failed", best_fitness="", known_optimum="", gap_pct="",
        evaluations="", elapsed_s="", profile_changes="",
        error=f"{type(exc).__name__}: {exc}",
    )
    jsonl = {**identity, "status": "failed", "error": failed["error"], "gap_pct": None}
    return failed, jsonl


def _append_jsonl_durable(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, default=_to_json) + "\n"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(line)
        fh.flush()
        os.fsync(fh.fileno())


def _write_winner(root: Path, record: dict) -> None:
    folder = _winner_dir(root, record["index"], record["policy"], record["configuration"])
    folder.mkdir(parents=True, exist_ok=True)
    _write_json(folder / "result.json", json.loads(json.dumps(record, default=_to_json)))


def _selected_by_names(names):
    by_name = {spec.name: spec for spec in CONFIGURATIONS}
    return [by_name[name] for name in names]


def _run_instance(index, inst, selected, seeds, pop, iterations, on_ok, on_fail, on_begin=None) -> None:
    """Sequential policies, configurations and seeds for one instance."""
    previous = os.environ.get(JOB_INDEX_ENV)
    os.environ[JOB_INDEX_ENV] = str(index)
    winners = {}
    try:
        optimizer = _optimizer()
        for policy in POLICIES:
            for spec in selected:
                for seed in seeds:
                    where = _where(index, policy, spec, seed)
                    identity = _identity(index, policy, spec, seed)
                    if on_begin is not None:
                        on_begin(where)
                    try:
                        res, elapsed_s = _invoke(
                            optimizer, inst, spec, policy, seed, pop, iterations)
                    except KeyboardInterrupt:
                        raise
                    except Exception as exc:
                        on_fail(where, identity, exc)
                        return
                    compact, record = _ok_payload(identity, res, elapsed_s)
                    key = (index, policy, spec.name)
                    winner = None
                    if key not in winners or _better(record, winners[key]):
                        winners[key] = record
                        winner = record
                    on_ok(where, compact, winner, res, elapsed_s)
    finally:
        if previous is None:
            os.environ.pop(JOB_INDEX_ENV, None)
        else:
            os.environ[JOB_INDEX_ENV] = previous


def run_instance_job(job: dict) -> None:
    """Spawn target. One instance. No plots and no root aggregate or manifest writes."""
    index = job["index"]
    root = Path(job["root"])
    selected = _selected_by_names(job["configurations"])
    outcome = {"failed": None}

    def on_ok(where, compact, winner, res, elapsed_s):
        if winner is not None:
            _write_winner(root, winner)
        _append_jsonl_durable(_journal_path(root, index), {
            "ok": True, "csv": compact, "jsonl": compact, "where": where,
        })

    def on_fail(where, identity, exc):
        failed, jsonl = _failed_payload(identity, exc)
        _append_jsonl_durable(_journal_path(root, index), {
            "ok": False, "csv": failed, "jsonl": jsonl, "where": where,
        })
        outcome["failed"] = {
            "status": "failed", "where": where,
            "error_type": type(exc).__name__, "error": str(exc),
        }
        _write_json(_status_path(root, index), outcome["failed"])

    try:
        inst = cargar_instancia(INSTANCE_FILE, index)
        _run_instance(
            index, inst, selected, job["seeds"], job["pop"], job["iterations"],
            on_ok, on_fail,
        )
    except KeyboardInterrupt:
        _write_json(_status_path(root, index), {
            "status": "interrupted", "where": None,
            "error_type": "KeyboardInterrupt", "error": "KeyboardInterrupt",
        })
        raise
    except Exception as exc:
        if outcome["failed"] is None:
            outcome["failed"] = {
                "status": "failed", "where": f"mknapcb1.txt[{index}]",
                "error_type": type(exc).__name__, "error": str(exc),
            }
            _write_json(_status_path(root, index), outcome["failed"])
        sys.exit(1)
    if outcome["failed"] is not None:
        sys.exit(1)
    _write_json(_status_path(root, index), {
        "status": "ok", "where": None, "error_type": None, "error": None,
    })


def _complete_journal_lines(path: Path) -> list:
    if not path.is_file():
        return []
    raw = path.read_text(encoding="utf-8")
    if not raw or "\n" not in raw:
        return []
    if not raw.endswith("\n"):
        raw = raw[:raw.rfind("\n") + 1]
    return [json.loads(line) for line in raw.splitlines() if line]


def _read_status(path: Path):
    for _ in range(20):
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
        time.sleep(0.05)
    return None


def _remove_journals(root: Path, indices) -> None:
    for index in indices:
        folder = _instance_dir(root, index) / "_journal"
        if not folder.exists():
            continue
        for path in folder.glob("*"):
            if path.is_file():
                path.unlink()
        folder.rmdir()


def _load_winners(root: Path) -> dict:
    winners = {}
    base = root / "mknapcb1"
    if not base.exists():
        return winners
    for path in base.glob("inst_*/*/*/best_run/result.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        winners[(record["index"], record["policy"], record["configuration"])] = record
    return winners


def _rewrite_runs(root, rows, indices, seeds, selected) -> list:
    policy_order = {name: i for i, name in enumerate(POLICIES)}
    index_order = {index: i for i, index in enumerate(indices)}
    seed_order = {seed: i for i, seed in enumerate(seeds)}
    config_order = {spec.name: i for i, spec in enumerate(selected)}
    ordered = sorted(rows, key=lambda row: (
        index_order[row["index"]], policy_order[row["policy"]],
        config_order[row["configuration"]], seed_order[row["seed"]],
    ))
    with (root / "runs.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=RUN_FIELDS)
        writer.writeheader()
        writer.writerows([{key: _csv_cell(row.get(key)) for key in RUN_FIELDS} for row in ordered])
    (root / "runs.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in ordered), encoding="utf-8")
    return ordered


def _stop_processes(running) -> None:
    """Termination is not a cooperative cancel inside run_hybrid."""
    for proc in running.values():
        if proc.is_alive():
            proc.terminate()
    for proc in running.values():
        proc.join(timeout=5)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=5)


def _apply_journal_line(root, state, total, where_box, rows, line) -> None:
    where_box[0] = line["where"]
    state["where"] = line["where"]
    _append_csv(root / "runs.csv", RUN_FIELDS, [line["csv"]])
    _append_jsonl(root / "runs.jsonl", line["jsonl"])
    if not line["ok"]:
        _write_json(root / "manifest.json", state)
        return
    rows.append(line["jsonl"])
    state["completed_runs"] += 1
    _write_json(root / "manifest.json", state)
    print(f"[{state['completed_runs']}/{total}] {line['where']} "
          f"best={line['jsonl']['best_fitness']} elapsed_s={line['jsonl']['elapsed_s']}")


def _run_serial(root, args, selected, instances, state, total, where_box):
    rows = []
    winners = {}

    def on_begin(where):
        where_box[0] = where
        state["where"] = where

    def on_ok(where, compact, winner, res, elapsed_s):
        where_box[0] = where
        state["where"] = where
        _append_csv(root / "runs.csv", RUN_FIELDS, [compact])
        _append_jsonl(root / "runs.jsonl", compact)
        rows.append(compact)
        if winner is not None:
            winners[(winner["index"], winner["policy"], winner["configuration"])] = winner
        state["completed_runs"] += 1
        _write_json(root / "manifest.json", state)
        print(f"[{state['completed_runs']}/{total}] {where} "
              f"best={res['best_fitness']} elapsed_s={elapsed_s}")

    def on_fail(where, identity, exc):
        where_box[0] = where
        state["where"] = where
        failed, jsonl = _failed_payload(identity, exc)
        _append_csv(root / "runs.csv", RUN_FIELDS, [failed])
        _append_jsonl(root / "runs.jsonl", jsonl)
        raise exc

    for index in args.indices:
        _run_instance(
            index, instances[index], selected, args.seeds, args.pop, args.iterations,
            on_ok, on_fail, on_begin,
        )
    return rows, winners


def _run_parallel(root, args, selected, state, total, where_box):
    """One spawned process per started instance.

    Communication is the instance journal of compact rows, not run traces.
    Pending instances are not started after a failure. A running sibling is
    terminated; that does not unwind run_hybrid and does not retry it.
    """
    import gwo_de_dtw.sensitivity.campaign as campaign_mod

    ctx = multiprocessing.get_context("spawn")
    target = campaign_mod.run_instance_job
    workers = state["selection"]["workers_effective"]
    pending = list(args.indices)
    running = {}
    seen = {index: 0 for index in args.indices}
    rows = []
    failure = None

    def job(index):
        return {
            "index": index, "root": str(root), "seeds": list(args.seeds),
            "pop": args.pop, "iterations": args.iterations,
            "configurations": [spec.name for spec in selected],
        }

    def ingest():
        for index in args.indices:
            lines = _complete_journal_lines(_journal_path(root, index))
            fresh = lines[seen[index]:]
            seen[index] = len(lines)
            for line in fresh:
                _apply_journal_line(root, state, total, where_box, rows, line)

    def start_more():
        while pending and len(running) < workers and failure is None:
            index = pending.pop(0)
            proc = ctx.Process(target=target, name=f"sensitivity-inst-{index}", args=(job(index),))
            proc.start()
            running[index] = proc

    def finish(index):
        nonlocal failure
        proc = running.pop(index)
        proc.join(timeout=5)
        ingest()
        status = _read_status(_status_path(root, index))
        if status and status.get("status") == "ok" and proc.exitcode == 0:
            return
        if failure is None:
            failure = status if status and status.get("status") == "failed" else {
                "status": "failed",
                "where": f"mknapcb1.txt[{index}]",
                "error_type": "WorkerExit",
                "error": (
                    f"worker exited {proc.exitcode} without a completed status; "
                    "in-flight run was not recorded or retried"
                ),
            }

    try:
        start_more()
        while running or (pending and failure is None):
            ingest()
            if failure is not None:
                break
            start_more()
            finished = [index for index, proc in running.items() if not proc.is_alive()]
            if not finished:
                if running:
                    time.sleep(0.05)
                continue
            for index in finished:
                finish(index)
            if failure is not None:
                break
        if failure is not None:
            _stop_processes(running)
            running.clear()
            ingest()
            pending.clear()
            where_box[0] = failure.get("where") or where_box[0]
            exc_type = failure.get("error_type") or "RuntimeError"
            message = failure.get("error") or "instance failed"
            raise type(exc_type, (Exception,), {})(message)
        ingest()
        ordered = _rewrite_runs(root, rows, args.indices, args.seeds, selected)
        _remove_journals(root, args.indices)
        return ordered
    except KeyboardInterrupt:
        _stop_processes(running)
        ingest()
        raise


def _manifest(args, selected, evidence, total, started) -> dict:
    monitors = {spec.name: asdict(spec.stagnation_config()) for spec in selected}
    return {
        "status": "running",
        "total_runs": total,
        "completed_runs": 0,
        "error": None,
        "where": None,
        "started": started,
        "finished": None,
        "timing": {"started": started, "finished": None, "elapsed_s": None, "unit": "seconds"},
        "selection": {
            "instance_file": INSTANCE_FILE,
            "indices": list(args.indices),
            "seeds": list(args.seeds),
            "policies": list(POLICIES),
            "configurations": [spec.name for spec in selected],
            "workers_requested": args.workers,
            "workers_effective": effective_workers(args.workers, args.indices),
        },
        "parallelism": {
            "by": "instance",
            "start_method": "spawn" if effective_workers(args.workers, args.indices) > 1 else "in-process",
            "pending_cancel": "instances not yet started are not started",
            "in_flight_cancel": "process termination; run_hybrid has no cooperative cancel, so a running call is not an in-frame interrupt and is not recorded as completed",
            "retry": False,
        },
        "effective_config": {
            "binarization": BINARIZATION,
            "hybrid_mode": HYBRID_MODE,
            "signal": "ddtw",
            "pop": args.pop,
            "iterations": args.iterations,
            "G1": G1, "G2": G2,
            "g3_exploit": G3_EXPLOIT, "g3_explore": G3_EXPLORE,
            "dtw_kirito": False,
            "min_slope": 2.0,
            "improvement_tol": 0.0,
            "adapt_thresholds": True,
            "monitor_by_configuration": monitors,
        },
        "provenance": {
            "runner": "gwo_de_dtw.controller.run_hybrid",
            "instance_loader": "gwo_de_dtw.problem.cargar_instancia",
            "design": "descriptive one-at-a-time",
            "inferential_tests": False,
            "seed_paired_initial_conditions": "not_claimed",
            "optimum_source": "instance header, else mkcbres.txt beside the file; non-positive means unknown",
        },
        "runtime": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__},
        "optimum_evidence": evidence,
    }


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    selected = resolve(parser, args)
    total = len(args.indices) * len(args.seeds) * len(selected) * len(POLICIES)
    print(plan_line(args.indices, args.seeds, selected, args.pop, args.iterations))
    if args.dry_run:
        print(DRY_LINE)
        return 0

    root = new_default_root() if args.out is None else args.out
    if args.out is not None:
        root.mkdir(parents=True, exist_ok=True)
    evidence = []
    instances = {}
    for index in args.indices:
        inst = cargar_instancia(INSTANCE_FILE, index)
        instances[index] = inst
        optimum = _positive(inst.get("optimo"))
        evidence.append({
            "index": index,
            "known_optimum": optimum,
            "reference": "known" if optimum is not None else "unknown",
        })
    started = datetime.now().isoformat(timespec="seconds")
    clock = time.perf_counter()
    state = _manifest(args, selected, evidence, total, started)
    _write_json(root / "manifest.json", state)
    with (root / "runs.csv").open("w", newline="", encoding="utf-8") as fh:
        csv.DictWriter(fh, fieldnames=RUN_FIELDS).writeheader()

    where_box = ["startup"]
    try:
        if state["selection"]["workers_effective"] == 1:
            rows, winners = _run_serial(
                root, args, selected, instances, state, total, where_box)
        else:
            rows = _run_parallel(root, args, selected, state, total, where_box)
            winners = _load_winners(root)
        where_box[0] = "export"
        state["where"] = "export"
        _export(root, winners, rows, state)
    except KeyboardInterrupt:
        state["status"] = "interrupted"
        state["error"] = f"{where_box[0]}: KeyboardInterrupt"
        state["finished"] = datetime.now().isoformat(timespec="seconds")
        state["timing"]["finished"] = state["finished"]
        state["timing"]["elapsed_s"] = round(time.perf_counter() - clock, 6)
        _write_json(root / "manifest.json", state)
        raise
    except Exception as exc:
        state["status"] = "failed"
        state["error"] = f"{where_box[0]}: {type(exc).__name__}: {exc}"
        state["finished"] = datetime.now().isoformat(timespec="seconds")
        state["timing"]["finished"] = state["finished"]
        state["timing"]["elapsed_s"] = round(time.perf_counter() - clock, 6)
        _write_json(root / "manifest.json", state)
        print(f"status=failed runs={state['completed_runs']}/{total}")
        print(f"output={root}")
        print(f"error: sensitivity failed at {state['error']}", file=sys.stderr)
        return 1

    state["status"] = "completed"
    state["where"] = None
    state["finished"] = datetime.now().isoformat(timespec="seconds")
    state["timing"]["finished"] = state["finished"]
    state["timing"]["elapsed_s"] = round(time.perf_counter() - clock, 6)
    _write_json(root / "manifest.json", state)
    print(f"status=completed runs={state['completed_runs']}/{total}")
    print(f"output={root}")
    return 0


def _better(candidate, incumbent) -> bool:
    if candidate["best_fitness"] != incumbent["best_fitness"]:
        return candidate["best_fitness"] > incumbent["best_fitness"]
    return candidate["seed"] < incumbent["seed"]


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
