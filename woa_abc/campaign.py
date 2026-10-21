"""Sequential campaign for indices 0, 1, and 2 of the nine OR-Library files.

One export per variant v0–v4 is 135 runs. Results are grouped by the loaded
item count: mknapcb1/4/7 are 100, mknapcb2/5/8 are 250, and mknapcb3/6/9 are
500. This is not a 31-run study, it does not download data, and n=1 claims
no superiority.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from . import solver as hybrid
from .benchmark import (
    ExportError,
    ExportOutcome,
    ExportRequest,
    _reject_url,
    _safe_timestamp,
    build_hybrid_config,
    export_single_run,
)


DEFAULT_SEED = 43
DEFAULT_POPULATION_SIZE = 30
DEFAULT_ITERATIONS = 1000
DEFAULT_MAX_EVALUATIONS = 180030
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "resultados"

INDICES = (0, 1, 2)
VARIANTS = ("v0", "v1", "v2", "v3", "v4")
FILES = tuple(f"mknapcb{number}.txt" for number in range(1, 10))
GROUP_BY_FILE = {
    "mknapcb1.txt": 100,
    "mknapcb4.txt": 100,
    "mknapcb7.txt": 100,
    "mknapcb2.txt": 250,
    "mknapcb5.txt": 250,
    "mknapcb8.txt": 250,
    "mknapcb3.txt": 500,
    "mknapcb6.txt": 500,
    "mknapcb9.txt": 500,
}
PLANNED_RUNS = 135
GROUP_RUNS = 45
RESULT_FIELDS = (
    "status",
    "variant",
    "file",
    "index",
    "n",
    "m",
    "seed",
    "reference",
    "gap",
    "fitness",
    "evaluations",
    "cycles_completed",
    "scouts",
    "recoveries",
    "elapsed",
    "scout_limit",
    "output_path",
)
NOTE = (
    "One run per selection. n=1 is not inferential evidence and claims no superiority. "
    "A stored reference of 0 is unknown and is not copied from another index."
)


class CampaignError(Exception):
    """Refused path, grouping, or settings. No solve was started for that refusal."""


@dataclass(frozen=True)
class Selection:
    file_name: str
    path: Path
    index: int
    variant: str
    n: int
    m: int
    reference: float


@dataclass
class CampaignOutcome:
    code: int
    output_dir: Path | None
    message: str = ""


def _cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return format(value, ".10g")
    return str(value)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows([{name: _cell(row.get(name)) for name in RESULT_FIELDS} for row in rows])


def _read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise CampaignError(f"expected a JSON object: {path}")
    return payload


def destination_for(campaign: Path, selection: Selection) -> Path:
    return (
        campaign
        / str(selection.n)
        / Path(selection.file_name).stem
        / f"inst_{selection.index:02d}"
        / selection.variant
    )


def _validate_settings(seed: int, population_size: int, iterations: int, max_evaluations: int) -> None:
    hybrid._validate_seed(seed)
    build_hybrid_config(
        ExportRequest(
            instance=Path("."),
            seed=seed,
            population_size=population_size,
            iterations=iterations,
            max_evaluations=max_evaluations,
            variant="v0",
        )
    )


def _reject_local(value: object, label: str) -> None:
    try:
        _reject_url(str(value), label)
    except ExportError as exc:
        raise CampaignError(str(exc)) from exc


def _selections(instances_dir: Path) -> list[Selection]:
    _reject_local(instances_dir, "instances dir")
    directory = Path(instances_dir)
    if not directory.is_dir():
        raise CampaignError("instances dir must be an existing local directory")
    found: list[Selection] = []
    for name in FILES:
        path = directory / name
        if not path.is_file():
            raise CampaignError(f"missing instance file: {name}")
        expected = GROUP_BY_FILE[name]
        loaded: list[tuple[int, dict]] = []
        for index in INDICES:
            _instance, info = hybrid._load_local_instance(path, index)
            if info["n"] != expected:
                raise CampaignError(f"{name} index {index} loaded n={info['n']}; expected {expected}")
            loaded.append((index, info))
        for index, info in loaded:
            for variant in VARIANTS:
                found.append(
                    Selection(
                        file_name=name,
                        path=path,
                        index=index,
                        variant=variant,
                        n=int(info["n"]),
                        m=int(info["m"]),
                        reference=float(info["reference"]),
                    )
                )
    counts: dict[int, int] = {}
    for item in found:
        counts[item.n] = counts.get(item.n, 0) + 1
    if len(found) != PLANNED_RUNS or any(counts.get(n) != GROUP_RUNS for n in (100, 250, 500)):
        raise CampaignError("selection count does not match 27 instances, 5 variants, and 45 runs per group")
    return found


def _group_counts(rows: list[dict]) -> dict[str, int]:
    counts = {"100": 0, "250": 0, "500": 0}
    for row in rows:
        key = str(row["n"])
        if key in counts:
            counts[key] += 1
    return counts


def _outcome_row(selection: Selection, destination: Path, outcome: ExportOutcome) -> dict:
    manifest = None
    raw = None
    try:
        manifest = _read_json(destination / "manifest.json")
        raw = _read_json(destination / "runs" / "run_01.json")
    except (OSError, json.JSONDecodeError, CampaignError):
        manifest = None
        raw = None
    completed = outcome.code == 0 and bool(manifest) and manifest.get("status") == "completed"
    counts = (manifest or {}).get("counts") or {}
    timing = (manifest or {}).get("timing") or {}
    phase = (raw or {}).get("evaluations_by_phase") or {}
    return {
        "status": "completed" if completed else "failed",
        "variant": selection.variant,
        "file": selection.file_name,
        "index": selection.index,
        "n": selection.n,
        "m": selection.m,
        "reference": selection.reference,
        "gap": None if raw is None else raw.get("gap"),
        "fitness": None if raw is None else raw.get("best_value"),
        "evaluations": None if raw is None else raw.get("evaluations"),
        "cycles_completed": None if raw is None else raw.get("cycles_completed"),
        "scouts": phase.get("scouts"),
        "recoveries": counts.get("n_recoveries"),
        "elapsed": timing.get("solve_seconds"),
        "scout_limit": None if raw is None else raw.get("scout_limit"),
        "output_path": str(destination),
    }


def _with_seed(row: dict, seed: int) -> dict:
    row["seed"] = seed
    return row


def _documents(campaign: Path, selections: list[Selection], settings: dict) -> tuple[dict, dict]:
    experiment = {
        "status": "running",
        "kind": "additive_variant_campaign",
        "runs_planned": PLANNED_RUNS,
        "runs_completed": 0,
        "runs_failed": 0,
        "superiority_claim": False,
        "one_run_per_selection": True,
        "repetitions": 1,
        "note": NOTE,
        "settings": settings,
        "groups": {"100": GROUP_RUNS, "250": GROUP_RUNS, "500": GROUP_RUNS},
        "selections": [
            {
                "file": item.file_name,
                "index": item.index,
                "variant": item.variant,
                "n": item.n,
                "m": item.m,
                "reference": item.reference,
                "output_path": str(destination_for(campaign, item)),
            }
            for item in selections
        ],
        "failed_selection": None,
    }
    summary = {
        "status": "running",
        "runs_planned": PLANNED_RUNS,
        "runs_completed": 0,
        "runs_failed": 0,
        "groups": {"100": 0, "250": 0, "500": 0},
        "superiority_claim": False,
        "note": NOTE,
        "failed_selection": None,
        "seed": settings["seed"],
        "population_size": settings["population_size"],
        "iterations": settings["iterations"],
        "max_evaluations": settings["max_evaluations"],
    }
    return experiment, summary


def _refresh(experiment: dict, summary: dict, rows: list[dict], failed: dict | None, status: str) -> None:
    completed = sum(1 for row in rows if row["status"] == "completed")
    failed_count = sum(1 for row in rows if row["status"] != "completed")
    experiment["runs_completed"] = completed
    experiment["runs_failed"] = failed_count
    experiment["status"] = status
    experiment["failed_selection"] = failed
    summary["runs_completed"] = completed
    summary["runs_failed"] = failed_count
    summary["status"] = status
    summary["failed_selection"] = failed
    summary["groups"] = _group_counts(rows)


def _persist(campaign: Path, experiment: dict, summary: dict, rows: list[dict]) -> None:
    _write_json(campaign / "experiment.json", experiment)
    _write_json(campaign / "campaign_summary.json", summary)
    _write_csv(campaign / "campaign_results.csv", rows)
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(str(row["n"]), []).append(row)
    for label, group_rows in grouped.items():
        _write_csv(campaign / label / "group_summary.csv", group_rows)


def _allocate(output_root: Path, timestamp: str | None) -> Path:
    _reject_local(output_root, "output root")
    root = Path(output_root)
    if root.exists() and not root.is_dir():
        raise CampaignError("output root must be a local directory")
    stamp = _safe_timestamp(timestamp or datetime.now().strftime("%Y%m%d_%H%M%S"))
    campaign = root / f"campaign_{stamp}"
    if campaign.exists():
        raise CampaignError(f"refusing to overwrite existing directory: {campaign}")
    campaign.mkdir(parents=True, exist_ok=False)
    return campaign


def run_campaign(
    instances_dir: Path,
    output_root: Path | None = None,
    *,
    seed: int = DEFAULT_SEED,
    population_size: int = DEFAULT_POPULATION_SIZE,
    iterations: int = DEFAULT_ITERATIONS,
    max_evaluations: int = DEFAULT_MAX_EVALUATIONS,
    timestamp: str | None = None,
) -> CampaignOutcome:
    """Validate every selected instance, then export one run at a time."""

    _validate_settings(seed, population_size, iterations, max_evaluations)
    selections = _selections(Path(instances_dir))
    root = DEFAULT_OUTPUT_ROOT if output_root is None else Path(output_root)
    campaign = _allocate(root, timestamp)
    settings = {
        "seed": seed,
        "population_size": population_size,
        "iterations": iterations,
        "max_evaluations": max_evaluations,
        "indices": list(INDICES),
        "variants": list(VARIANTS),
        "files": list(FILES),
    }
    experiment, summary = _documents(campaign, selections, settings)
    rows: list[dict] = []
    _persist(campaign, experiment, summary, rows)
    for selection in selections:
        destination = destination_for(campaign, selection)
        print(
            f"start variant={selection.variant} file={selection.file_name} index={selection.index} n={selection.n}",
            flush=True,
        )
        request = ExportRequest(
            instance=selection.path,
            instance_index=selection.index,
            seed=seed,
            population_size=population_size,
            iterations=iterations,
            max_evaluations=max_evaluations,
            output_root=root,
            variant=selection.variant,
            destination=destination,
        )
        try:
            outcome = export_single_run(request)
        except (ExportError, OSError, ValueError, RuntimeError) as exc:
            outcome = ExportOutcome(1, destination, f"{type(exc).__name__}: {exc}")
        row = _with_seed(_outcome_row(selection, destination, outcome), seed)
        rows.append(row)
        failed = None
        status = "running"
        if row["status"] != "completed":
            status = "failed"
            failed = {
                "file": selection.file_name,
                "index": selection.index,
                "variant": selection.variant,
                "n": selection.n,
                "output_path": str(destination),
            }
        _refresh(experiment, summary, rows, failed, status)
        _persist(campaign, experiment, summary, rows)
        if status == "failed":
            print(
                f"failed variant={selection.variant} file={selection.file_name} index={selection.index} n={selection.n} output={destination}",
                flush=True,
            )
            return CampaignOutcome(
                1,
                campaign,
                f"stopped on {selection.file_name} index {selection.index} variant {selection.variant}; solver was not rerun",
            )
        print(
            (
                f"completed variant={selection.variant} file={selection.file_name} "
                f"index={selection.index} n={selection.n} fitness={_cell(row['fitness'])} "
                f"evaluations={_cell(row['evaluations'])} cycles={_cell(row['cycles_completed'])} "
                f"output={destination}"
            ),
            flush=True,
        )
    if summary["runs_completed"] != PLANNED_RUNS or summary["runs_failed"] != 0:
        _refresh(experiment, summary, rows, summary.get("failed_selection"), "failed")
        _persist(campaign, experiment, summary, rows)
        return CampaignOutcome(1, campaign, "campaign incomplete; solver was not rerun")
    _refresh(experiment, summary, rows, None, "completed")
    _persist(campaign, experiment, summary, rows)
    return CampaignOutcome(0, campaign, f"completed {PLANNED_RUNS}/{PLANNED_RUNS} {campaign}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="woa_abc.campaign",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Run 135 sequential binary WOA-ABC exports: indices 0, 1, and 2 of "
            "mknapcb1.txt through mknapcb9.txt, for variants v0, v1, v2, v3, and v4. "
            "Loaded sizes form three groups of 45 (100, 250, and 500). "
            "Each selection is one run. This does not download data and claims no superiority."
        ),
    )
    parser.add_argument(
        "--instances-dir",
        type=Path,
        required=True,
        help="Local directory containing mknapcb1.txt through mknapcb9.txt.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory that receives a new campaign_<timestamp> folder.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="RNG seed.")
    parser.add_argument(
        "--population-size",
        type=int,
        default=DEFAULT_POPULATION_SIZE,
        help="Population size.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=DEFAULT_ITERATIONS,
        help="Hybrid-cycle horizon. A run may stop earlier at the evaluation cap.",
    )
    parser.add_argument(
        "--max-evaluations",
        type=int,
        default=DEFAULT_MAX_EVALUATIONS,
        help="Explicit shared evaluation cap. This does not require all cycles to run.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        code = exc.code
        return 0 if code is None else int(code)
    try:
        outcome = run_campaign(
            args.instances_dir,
            args.output_root,
            seed=args.seed,
            population_size=args.population_size,
            iterations=args.iterations,
            max_evaluations=args.max_evaluations,
        )
    except (CampaignError, ExportError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    stream = sys.stdout if outcome.code == 0 else sys.stderr
    print(outcome.message, file=stream)
    if outcome.code != 0 and outcome.output_dir is not None:
        print(f"output: {outcome.output_dir}", file=sys.stderr)
    return outcome.code


if __name__ == "__main__":
    raise SystemExit(main())
