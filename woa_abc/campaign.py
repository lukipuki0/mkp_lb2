"""Sequential campaign over selected OR-Library files and indices.

Defaults: all nine files, indices 0, 1, and 2; five recovery-clock variants
(135 runs) or four policies (108 runs). Results are grouped by the loaded
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
from .profiles import POLICIES, POLICY, check_policy


DEFAULT_SEED = 43
DEFAULT_POPULATION_SIZE = 30
DEFAULT_ITERATIONS = 1000
DEFAULT_MAX_EVALUATIONS = 180030
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "resultados"

INDICES = (0, 1, 2)
VARIANTS = hybrid.VARIANTS
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
    "reference_source",
    "gap",
    "fitness",
    "evaluations",
    "cycles_completed",
    "scouts",
    "profile_changes",
    "policy",
    "elapsed",
    "scout_limit",
    "output_path",
    "reaction",
    "recoveries",
)
NOTE = (
    "One run per selection. n=1 is not inferential evidence and claims no superiority. "
    "A header reference of 0 is the best known value of that same index in mkcbres.txt "
    "next to the instance file; without that entry it is unknown. It is never copied "
    "from another index."
)


class CampaignError(Exception):
    """Refused path, grouping, or settings. No solve was started for that refusal."""


@dataclass(frozen=True)
class Selection:
    file_name: str
    path: Path
    index: int
    variant: str | None
    n: int
    m: int
    reference: float
    reference_source: str
    policy: str | None = None

    @property
    def label(self) -> str:
        return self.variant or self.policy


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
        / selection.label
    )


def _validate_settings(
    seed: int,
    population_size: int,
    iterations: int,
    max_evaluations: int,
    policy: str | None = None,
    dtw_kirito: bool = False,
    reaction: str = "variant",
) -> None:
    hybrid._validate_seed(seed)
    build_hybrid_config(
        ExportRequest(
            instance=Path("."),
            seed=seed,
            population_size=population_size,
            iterations=iterations,
            max_evaluations=max_evaluations,
            reaction=reaction,
            variant="v0" if reaction == "variant" else None,
            policy=policy,
            dtw_kirito=dtw_kirito,
        )
    )


def _reject_local(value: object, label: str) -> None:
    try:
        _reject_url(str(value), label)
    except ExportError as exc:
        raise CampaignError(str(exc)) from exc


def _selection_inputs(files=None, indices=None) -> tuple[list[str], list[int]]:
    names = list(FILES if files is None else files)
    selected_indices = list(INDICES if indices is None else indices)
    if any(name not in FILES for name in names):
        raise CampaignError("files must be basenames from mknapcb1.txt through mknapcb9.txt")
    if len(set(names)) != len(names):
        raise CampaignError("files must not contain duplicates")
    if any(isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in selected_indices):
        raise CampaignError("indices must be nonnegative integers")
    if len(set(selected_indices)) != len(selected_indices):
        raise CampaignError("indices must not contain duplicates")
    return [name for name in FILES if name in names], sorted(selected_indices)


def _selections(
    instances_dir: Path, reaction: str = "variant", policy: str | None = None,
    *, files=None, indices=None,
) -> list[Selection]:
    files, indices = _selection_inputs(files, indices)
    _reject_local(instances_dir, "instances dir")
    directory = Path(instances_dir)
    if not directory.is_dir():
        raise CampaignError("instances dir must be an existing local directory")
    if reaction not in hybrid.REACTIONS:
        raise ValueError(hybrid.REACTION_ERROR)
    if reaction == 'variant' and policy is not None:
        raise ValueError(hybrid.VARIANT_MODE_ERROR)
    if policy is not None:
        check_policy(policy)
    members = VARIANTS if reaction == 'variant' else ((policy,) if policy else POLICIES)
    found: list[Selection] = []
    for name in files:
        path = directory / name
        if not path.is_file():
            raise CampaignError(f"missing instance file: {name}")
        expected = GROUP_BY_FILE[name]
        loaded: list[tuple[int, dict]] = []
        for index in indices:
            _instance, info = hybrid._load_local_instance(path, index)
            if info["n"] != expected:
                raise CampaignError(f"{name} index {index} loaded n={info['n']}; expected {expected}")
            loaded.append((index, info))
        for index, info in loaded:
            for member in members:
                found.append(
                    Selection(
                        file_name=name,
                        path=path,
                        index=index,
                        variant=member if reaction == 'variant' else None,
                        policy=member if reaction == 'policy' else None,
                        n=int(info["n"]),
                        m=int(info["m"]),
                        reference=float(info["reference"]),
                        reference_source=str(info["reference_source"]),
                    )
                )
    counts: dict[int, int] = {}
    for item in found:
        counts[item.n] = counts.get(item.n, 0) + 1
    expected_total = len(files) * len(indices) * len(members)
    if len(found) != expected_total or any(
        counts.get(n, 0) != sum(GROUP_BY_FILE[name] == n for name in files) * len(indices) * len(members)
        for n in (100, 250, 500)
    ):
        raise CampaignError("selection count does not match the selected reaction group")
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
    raw_config = ((raw or {}).get("metadata") or {}).get("config") or {}
    return {
        "status": "completed" if completed else "failed",
        "variant": selection.variant,
        "file": selection.file_name,
        "index": selection.index,
        "n": selection.n,
        "m": selection.m,
        "reference": selection.reference,
        "reference_source": selection.reference_source,
        "gap": None if raw is None else raw.get("gap"),
        "fitness": None if raw is None else raw.get("best_value"),
        "evaluations": None if raw is None else raw.get("evaluations"),
        "cycles_completed": None if raw is None else raw.get("cycles_completed"),
        "scouts": phase.get("scouts"),
        "profile_changes": counts.get("n_profile_changes"),
        "policy": raw_config.get("policy"),
        "elapsed": timing.get("solve_seconds"),
        "scout_limit": None if raw is None else raw.get("scout_limit"),
        "output_path": str(destination),
        "reaction": raw_config.get("reaction"),
        "recoveries": counts.get("n_recoveries"),
    }


def _with_seed(row: dict, seed: int) -> dict:
    row["seed"] = seed
    return row


def _documents(campaign: Path, selections: list[Selection], settings: dict) -> tuple[dict, dict]:
    experiment = {
        "status": "running",
        "kind": "separate_reaction_campaign",
        "runs_planned": len(selections),
        "runs_completed": 0,
        "runs_failed": 0,
        "superiority_claim": False,
        "one_run_per_selection": True,
        "repetitions": 1,
        "note": NOTE,
        "settings": settings,
        "groups": {str(n): sum(s.n == n for s in selections) for n in (100, 250, 500)},
        "selections": [
            {
                "file": item.file_name,
                "index": item.index,
                "variant": item.variant,
                "n": item.n,
                "m": item.m,
                "reference": item.reference,
                "reference_source": item.reference_source,
                "output_path": str(destination_for(campaign, item)),
            }
            for item in selections
        ],
        "failed_selection": None,
    }
    summary = {
        "status": "running",
        "runs_planned": len(selections),
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
        "policy": settings["policy"],
        "dtw_kirito": settings["dtw_kirito"],
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
    policy: str | None = None,
    dtw_kirito: bool = False,
    reaction: str = "variant",
    files=None,
    indices=None,
) -> CampaignOutcome:
    """Validate every selected instance, then export one run at a time."""

    _validate_settings(seed, population_size, iterations, max_evaluations, policy, dtw_kirito, reaction)
    files, indices = _selection_inputs(files, indices)
    selections = _selections(Path(instances_dir), reaction, policy, files=files, indices=indices)
    root = DEFAULT_OUTPUT_ROOT if output_root is None else Path(output_root)
    campaign = _allocate(root, timestamp)
    settings = {
        "seed": seed,
        "population_size": population_size,
        "iterations": iterations,
        "max_evaluations": max_evaluations,
        "indices": indices,
        "variants": list(VARIANTS) if reaction == 'variant' else [],
        "reaction": reaction,
        "files": files,
        "policy": policy,
        "dtw_kirito": dtw_kirito,
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
            reaction=reaction,
            policy=selection.policy,
            dtw_kirito=dtw_kirito,
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
    if summary["runs_completed"] != len(selections) or summary["runs_failed"] != 0:
        _refresh(experiment, summary, rows, summary.get("failed_selection"), "failed")
        _persist(campaign, experiment, summary, rows)
        return CampaignOutcome(1, campaign, "campaign incomplete; solver was not rerun")
    _refresh(experiment, summary, rows, None, "completed")
    _persist(campaign, experiment, summary, rows)
    return CampaignOutcome(0, campaign, f"completed {len(selections)}/{len(selections)} {campaign}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="woa_abc.campaign",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Run sequential WOA-ABC exports over selected files and indices. "
            "Defaults: indices 0, 1, and 2 of mknapcb1.txt through mknapcb9.txt; "
            "separate variants v0-v4 (135 runs, three groups of 45 for 100, 250, 500). "
            "With default files/indices, policy mode runs all four policies (108 runs), "
            "or --policy selects one (27 runs). "
            "Each selection is one run. This does not download data and claims no superiority."
        ),
    )
    parser.add_argument(
        "--instances-dir",
        type=Path,
        required=True,
        help="Local directory containing the selected instance files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory that receives a new campaign_<timestamp> folder.",
    )
    parser.add_argument(
        "--files", nargs="+", default=None,
        help="Instance basenames, without duplicates; default all mknapcb1.txt through mknapcb9.txt in that order.",
    )
    parser.add_argument(
        "--indices", nargs="+", type=int, default=None,
        help="Nonnegative instance indices, without duplicates, sorted ascending; default 0 1 2.",
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
    parser.add_argument('--reaction', choices=hybrid.REACTIONS, default='variant',
                        help='Separate campaign mode; default variant (135 runs with default files/indices).')
    parser.add_argument(
        "--policy",
        default=None,
        choices=POLICIES,
        help="Policy mode only: select one policy instead of the four-policy group.",
    )
    parser.add_argument(
        "--dtw-kirito",
        action="store_true",
        help="Binary policies only: keep one monitor and decide on every ready observation.",
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
            policy=args.policy,
            reaction=args.reaction,
            dtw_kirito=bool(args.dtw_kirito),
            files=args.files,
            indices=args.indices,
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
