"""Read-only summary of native-iteration hybrid-format results."""
import argparse
import csv
import json
import math
from pathlib import Path


def _blank(value):
    return value is None or str(value).strip() == ""


def _required_int(value, field):
    if _blank(value):
        raise ValueError(f"{field} is required")
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be an integer") from error


def _optional_int(value, field):
    if _blank(value):
        return None
    return _required_int(value, field)


def _optional_float(value, field):
    if _blank(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be a number") from error
    if not math.isfinite(number):
        return None
    return number


def _analyze_iteration_runs(campaign: Path) -> dict:
    successful, failed = [], []
    for path in sorted(campaign.glob("*/run_*/summary.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        row = {**row, "condition": path.parent.parent.name, "run_directory": str(path.parent)}
        (successful if row.get("status") == "success" else failed).append(row)
    if not successful and not failed:
        raise ValueError("Campaign has no run summaries")
    return {"schema_version": 2, "budget_unit": "native_iterations",
            "successful_runs": len(successful), "failed_runs": len(failed),
            "runs": successful, "failures": failed,
            "note": "Native epochs may overshoot requested iterations; no new plots or inferred evaluation counts"}


def _typed_metric(raw, case, condition):
    fitness = _optional_float(raw.get("final_fitness"), "final_fitness")
    return {
        "case": case, "condition": condition,
        "run": _optional_int(raw.get("run"), "run"),
        "seed": _required_int(raw.get("seed"), "seed"),
        "status": raw.get("status") or "",
        "fitness": fitness, "final_fitness": fitness,
        "iterations": _optional_int(raw.get("actual_iterations"), "actual_iterations"),
        "requested_iterations": _optional_int(raw.get("requested_iterations"), "requested_iterations"),
        "iteration_overshoot": _optional_int(raw.get("iteration_overshoot"), "iteration_overshoot"),
        "gap_pct": _optional_float(raw.get("gap_pct"), "gap_pct"),
        "elapsed_seconds": _optional_float(raw.get("elapsed_seconds"), "elapsed_seconds"),
        "n_blocks": _optional_int(raw.get("n_blocks"), "n_blocks"),
        "transitions": _optional_int(raw.get("transitions"), "transitions"),
        "error": raw.get("error") or None,
    }


def _inside_campaign(campaign: Path, relative: str):
    """Resolve a protocol path and reject absolute paths or escapes."""
    if not isinstance(relative, str) or relative.startswith(("/", "\\")) or ":" in relative:
        return None
    root = campaign.resolve()
    candidate = (campaign / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate


def _case_label(case):
    relative = case.get("relative_output_path")
    if isinstance(relative, str) and relative:
        return relative
    try:
        return f"items_{int(case['n'])}/mknapcb{int(case['family'])}_inst_{int(case['index']):02d}"
    except (KeyError, TypeError, ValueError):
        return relative if isinstance(relative, str) else ""


def _protocol_metric_files(campaign: Path, protocol: dict):
    """Prefer paths stored in the protocol, including a single-case output root."""
    found = []
    cases = protocol.get("cases")
    if not isinstance(cases, list):
        return found
    for case in cases:
        if not isinstance(case, dict):
            continue
        label = _case_label(case)
        conditions = case.get("conditions")
        if not isinstance(conditions, list):
            continue
        for condition in conditions:
            if not isinstance(condition, dict) or not isinstance(condition.get("relative_output_path"), str):
                continue
            directory = _inside_campaign(campaign, condition["relative_output_path"])
            if directory is None:
                continue
            path = directory / "runs_resultados.csv"
            if path.is_file():
                found.append((path, label, str(condition.get("condition") or "")))
    return found


def _legacy_metric_files(campaign: Path):
    """Schema-3 layout from before case/condition folders were flattened."""
    found = []
    for path in sorted(campaign.glob("items_*/mknapcb*/inst_*/*/runs_resultados.csv")):
        relative = path.relative_to(campaign)
        if len(relative.parts) < 5:
            raise ValueError("Campaign has no run summaries")
        found.append((path, "/".join(relative.parts[:3]), relative.parts[3]))
    return found


def _collect_metrics(identified):
    metrics, failures, cases = [], [], set()
    for path, case, condition in identified:
        cases.add(case)
        with path.open(encoding="utf-8", newline="") as handle:
            for raw in csv.DictReader(handle):
                metric = _typed_metric(raw, case, condition)
                metrics.append(metric)
                if metric["status"] == "failed":
                    failures.append(metric)
    return metrics, failures, cases, len(identified)


def _analyze_grouped(campaign: Path, protocol: dict) -> dict:
    """Read every runs_resultados.csv. Do not reduce a case to its winner."""
    identified = _protocol_metric_files(campaign, protocol) or _legacy_metric_files(campaign)
    if not identified:
        raise ValueError("Campaign has no run summaries")
    metrics, failures, cases, case_conditions = _collect_metrics(identified)
    if not metrics:
        raise ValueError("Campaign has no run summaries")
    return {
        "schema_version": 3, "budget_unit": "native_iterations", "kind": "grouped_mkp_campaign",
        "distinct_cases": len(cases), "case_conditions": case_conditions,
        "successful_runs": sum(metric["status"] == "success" for metric in metrics),
        "failed_runs": len(failures), "metrics": metrics, "failures": failures,
        "note": "Read-only grouped metrics for later within-case statistics; no inference and no new plots",
    }


def analyze_campaign(campaign: Path) -> dict:
    """Keep requested and actual iterations distinct; do not mix retired formats."""
    campaign = Path(campaign)
    protocol = json.loads((campaign / "protocol.json").read_text(encoding="utf-8"))
    version, budget = protocol.get("schema_version"), protocol.get("budget_unit")
    if version == 2 and budget == "native_iterations":
        return _analyze_iteration_runs(campaign)
    if version == 3 and budget == "native_iterations" and protocol.get("kind") == "grouped_mkp_campaign":
        return _analyze_grouped(campaign, protocol)
    raise ValueError("This is a retired evaluation campaign; use its original files without relabeling them as iterations")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True, help="Native-iteration campaign directory")
    args = parser.parse_args(argv)
    try:
        report = analyze_campaign(args.campaign)
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(report, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
