"""Many-seed WOA-ABC campaign in policy mode; the best run of each policy is kept.

Usage (from the repository root):
    python -m woa_abc.seed_campaign --files mknapcb1.txt --indices 0 --out <folder>

Defaults: mknapcb1..9.txt from instancias/, indices 0 1 2, seeds 1..31, the
four policies, population 30, 1000 cycles, 180030 evaluations. The layout and
CSV columns are those of gwo_de_dtw.campaign:

    campaign_config.json        settings, selection, total/completed runs, status
    summary.csv                 one row per run (all instances), written after each run
    statistics.csv              one row per instance and policy
    <file stem>/inst_<NN>/runs.csv, statistics.csv
    <file stem>/inst_<NN>/convergence_comparison.png/.pdf   best run of each policy
    <file stem>/inst_<NN>/<policy>/best_run/                full export of that run

Each run is written by woa_abc.benchmark.export_single_run into a temporary
folder under <out>/.runs/. Only the current best run of each policy is kept;
a beaten run is deleted. Best run = highest best_fitness; ties go to the lower
seed. At the end of an instance the four winners move to <policy>/best_run/ and
the temporary folder is removed. A failing run stops the campaign (status
failed, exit 1); finished rows stay. There is no resume.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import statistics
import sys

from . import solver as hybrid
from .benchmark import ExportError, ExportOutcome, ExportRequest, _display_name, export_single_run
from .campaign import (
    DEFAULT_ITERATIONS,
    DEFAULT_MAX_EVALUATIONS,
    DEFAULT_POPULATION_SIZE,
    FILES,
    INDICES,
    CampaignError,
    _selections,
    _validate_settings,
)
from .plots import plot_convergence_comparison, plot_record
from .profiles import POLICIES

SEEDS = tuple(range(1, 32))
CONFIG_FILE = "campaign_config.json"
STAGING = ".runs"
RUN_FIELDS = ["file", "index", "policy", "seed", "status", "best_fitness", "known_optimum",
              "gap_pct", "evaluations", "generations", "changes", "elapsed_s"]
STAT_FIELDS = ["file", "index", "policy", "runs", "best_fitness", "worst_fitness",
               "mean_fitness", "median_fitness", "std_fitness", "best_seed",
               "known_optimum", "best_gap_pct", "mean_gap_pct", "mean_elapsed_s"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m woa_abc.seed_campaign",
        description="Many-seed WOA-ABC campaign in policy mode: every file x index x seed x "
                    "policy; only the best run of each policy keeps its export and plots.",
        allow_abbrev=False,
    )
    parser.add_argument("--instances-dir", type=Path, default=Path("instancias"),
                        help="folder of the instance files (default: %(default)s)")
    parser.add_argument("--files", nargs="+", default=list(FILES),
                        help="instance basenames among mknapcb1.txt .. mknapcb9.txt (default: all nine)")
    parser.add_argument("--indices", nargs="+", type=int, default=list(INDICES),
                        help="instance indices of every file (default: 0 1 2)")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS),
                        help="seeds, nonnegative and unique (default: 1 .. 31)")
    parser.add_argument("--population-size", type=int, default=DEFAULT_POPULATION_SIZE,
                        help="(default: %(default)s)")
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS,
                        help="hybrid-cycle horizon (default: %(default)s)")
    parser.add_argument("--max-evaluations", type=int, default=DEFAULT_MAX_EVALUATIONS,
                        help="shared evaluation cap (default: %(default)s)")
    parser.add_argument("--dtw-kirito", action="store_true",
                        help="Kirito rolling control for the binary policies (default: off)")
    parser.add_argument("--out", type=Path, required=True,
                        help="campaign folder; it must not exist or be empty")
    parser.add_argument("--dry-run", action="store_true",
                        help="validate, print the number of runs and stop without running or writing")
    return parser


def _duplicate(values):
    seen = set()
    for value in values:
        if value in seen:
            return value
        seen.add(value)
    return None


def validate(parser, args) -> list:
    """Every check before the first run or write; parser.error exits with code 2."""
    for name in args.files:
        if Path(name).name != name or "/" in name or "\\" in name:
            parser.error(f"invalid instance file name (basename only): {name}")
    for label, values in (("instance file", args.files), ("index", args.indices), ("seed", args.seeds)):
        if _duplicate(values) is not None:
            parser.error(f"duplicate {label}: {_duplicate(values)}")
    for seed in args.seeds:
        if seed < 0:
            parser.error(f"seeds must be nonnegative: {seed}")
    try:
        for seed in args.seeds:
            hybrid._validate_seed(seed)
        _validate_settings(args.seeds[0], args.population_size, args.iterations,
                           args.max_evaluations, None, bool(args.dtw_kirito), "policy")
        selections = _selections(args.instances_dir, "policy", None,
                                 files=args.files, indices=args.indices)
    except (CampaignError, ExportError, ValueError, OSError, IndexError) as exc:
        parser.error(str(exc))
    if args.out.exists() and not args.out.is_dir():
        parser.error(f"output path is not a directory: {args.out}")
    if args.out.is_dir() and any(args.out.iterdir()):
        parser.error(f"output directory is not empty: {args.out}")
    return selections


def _blank(value):
    return "" if value is None else value


def _start_csv(path: Path, fields) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        csv.DictWriter(fh, fieldnames=fields, lineterminator="\n").writeheader()


def _append_csv(paths, fields, rows) -> None:
    for path in paths:
        with open(path, "a", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=fields, lineterminator="\n").writerows(
                [{name: _blank(row.get(name)) for name in fields} for row in rows])


def _write_json(path: Path, data: dict) -> None:
    """Replaces `path` in one step so an interruption leaves the previous version."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def _read_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def is_better(row: dict, best) -> bool:
    """Higher best_fitness wins; on a tie the lower seed (independent of run order)."""
    if best is None:
        return True
    if row["best_fitness"] != best["best_fitness"]:
        return row["best_fitness"] > best["best_fitness"]
    return row["seed"] < best["seed"]


def policy_statistics(name: str, index: int, policy: str, rows, best: dict) -> dict:
    """Statistics row of one instance and policy (same definitions as gwo_de_dtw.campaign)."""
    fitness = [row["best_fitness"] for row in rows]
    gaps = [row["gap_pct"] for row in rows if row["gap_pct"] is not None]
    return {
        "file": name, "index": index, "policy": policy, "runs": len(rows),
        "best_fitness": max(fitness), "worst_fitness": min(fitness),
        "mean_fitness": statistics.mean(fitness),
        "median_fitness": statistics.median(fitness),
        "std_fitness": statistics.stdev(fitness) if len(fitness) > 1 else None,
        "best_seed": best["seed"], "known_optimum": rows[0]["known_optimum"],
        "best_gap_pct": min(gaps) if gaps else None,
        "mean_gap_pct": statistics.mean(gaps) if gaps else None,
        "mean_elapsed_s": statistics.mean(row["elapsed_s"] for row in rows),
    }


def run_row(selection, seed: int, destination: Path, outcome: ExportOutcome) -> dict:
    """One runs.csv row from the export's manifest and raw JSON; blank metrics when it failed."""
    manifest = _read_json(destination / "manifest.json")
    raw = _read_json(destination / "runs" / "run_01.json")
    row = {"file": selection.file_name, "index": selection.index, "policy": selection.policy,
           "seed": seed, "status": "failed", "best_fitness": None,
           "known_optimum": selection.reference if selection.reference > 0 else None,
           "gap_pct": None, "evaluations": None, "generations": None, "changes": None,
           "elapsed_s": None}
    if outcome.code != 0 or manifest.get("status") != "completed" or "best_value" not in raw:
        return row
    row.update({
        "status": "completed",
        "best_fitness": float(raw["best_value"]),
        "gap_pct": raw.get("gap"),
        "evaluations": raw.get("evaluations"),
        "generations": raw.get("cycles_completed"),
        "changes": (manifest.get("counts") or {}).get("n_profile_changes"),
        "elapsed_s": float((manifest.get("timing") or {}).get("solve_seconds") or 0.0),
    })
    return row


def plot_comparison(raws, folder: Path, display: str, reference: float) -> None:
    """convergence_comparison.png/.pdf from the raw JSON of the winners, one line per policy."""
    records = [plot_record(raw, instance=display, known_optimum=reference) for raw in raws]
    plot_convergence_comparison(records, folder, seed_labels=True)


def _remove(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


def run(args, selections) -> int:
    instances: dict[tuple[str, int], list] = {}
    for selection in selections:
        instances.setdefault((selection.file_name, selection.index), []).append(selection)
    total = len(selections) * len(args.seeds)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    config = {
        "status": "running", "started": datetime.now().isoformat(timespec="seconds"),
        "instances_dir": str(args.instances_dir), "files": list(dict.fromkeys(s.file_name for s in selections)),
        "indices": sorted({s.index for s in selections}), "seeds": list(args.seeds),
        "policies": list(POLICIES), "reaction": "policy",
        "population_size": args.population_size, "iterations": args.iterations,
        "max_evaluations": args.max_evaluations, "dtw_kirito": bool(args.dtw_kirito),
        "total_runs": total, "completed_runs": 0, "error": None,
    }
    _write_json(out / CONFIG_FILE, config)
    summary, overall = out / "summary.csv", out / "statistics.csv"
    _start_csv(summary, RUN_FIELDS)
    _start_csv(overall, STAT_FIELDS)
    print(f"woa_abc.seed_campaign: instances={len(instances)} seeds={len(args.seeds)} "
          f"policies={len(POLICIES)} runs={total} out={out}", flush=True)

    for (name, index), members in instances.items():
        folder = out / Path(name).stem / f"inst_{index:02d}"
        folder.mkdir(parents=True)
        runs_csv, stats_csv = folder / "runs.csv", folder / "statistics.csv"
        _start_csv(runs_csv, RUN_FIELDS)
        _start_csv(stats_csv, STAT_FIELDS)
        staging = out / STAGING / f"{Path(name).stem}_inst_{index:02d}"
        rows = {member.policy: [] for member in members}
        best: dict[str, tuple[dict, Path] | None] = {member.policy: None for member in members}
        for seed in args.seeds:
            for member in members:
                destination = staging / member.policy / f"seed_{seed:02d}"
                request = ExportRequest(
                    instance=member.path, instance_index=index, seed=seed,
                    population_size=args.population_size, iterations=args.iterations,
                    max_evaluations=args.max_evaluations, destination=destination,
                    reaction="policy", policy=member.policy, dtw_kirito=bool(args.dtw_kirito),
                )
                try:
                    outcome = export_single_run(request)
                except (ExportError, OSError, ValueError, RuntimeError) as exc:
                    outcome = ExportOutcome(1, destination, f"{type(exc).__name__}: {exc}")
                row = run_row(member, seed, destination, outcome)
                _append_csv((runs_csv, summary), RUN_FIELDS, [row])
                if row["status"] != "completed":
                    config.update(status="failed",
                                  error=f"{name}[{index}] policy={member.policy} seed={seed}: {outcome.message}")
                    _write_json(out / CONFIG_FILE, config)
                    print(f"error: campaign failed at {config['error']}; export kept in {destination}",
                          file=sys.stderr)
                    return 1
                config["completed_runs"] += 1
                _write_json(out / CONFIG_FILE, config)
                rows[member.policy].append(row)
                current = best[member.policy]
                if is_better(row, None if current is None else current[0]):
                    if current is not None:
                        _remove(current[1])
                    best[member.policy] = (row, destination)
                else:
                    _remove(destination)
                print(f"done {name}[{index}] {member.policy} seed={seed} "
                      f"fitness={row['best_fitness']:g}", flush=True)

        raws = []
        stats_rows = []
        for member in members:
            winner, source = best[member.policy]
            target = folder / member.policy / "best_run"
            target.parent.mkdir(parents=True)
            shutil.move(str(source), str(target))
            stats_rows.append(policy_statistics(name, index, member.policy, rows[member.policy], winner))
            raws.append(_read_json(target / "runs" / "run_01.json"))
        _append_csv((stats_csv, overall), STAT_FIELDS, stats_rows)
        plot_comparison(raws, folder, _display_name(members[0].path, index), members[0].reference)
        _remove(staging)
    _remove(out / STAGING)
    config.update(status="completed", finished=datetime.now().isoformat(timespec="seconds"))
    _write_json(out / CONFIG_FILE, config)
    print(f"completed {config['completed_runs']}/{total} {out}", flush=True)
    return 0


def main(argv=None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        selections = validate(parser, args)
    except SystemExit as exc:
        return 0 if exc.code is None else int(exc.code)
    total = len(selections) * len(args.seeds)
    if args.dry_run:
        print(f"woa_abc.seed_campaign dry run: files={len(set(s.file_name for s in selections))} "
              f"indices={len({s.index for s in selections})} seeds={len(args.seeds)} "
              f"policies={len(POLICIES)} runs={total} out={args.out}")
        return 0
    return run(args, selections)


if __name__ == "__main__":
    raise SystemExit(main())
