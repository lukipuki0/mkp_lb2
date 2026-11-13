"""Recover scalar ASOC DDTW reports from saved records; stdlib only, no solver imports."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import sys

POPULATION = ["PSO", "GWO", "WOA", "EHO", "ACO"]
BASELINES = POPULATION + ["ILS", "SA", "TS", "VNS"]
DECODED = ("wind_mw", "pv_mw", "n_el_units", "electrolyzer_mw", "battery_mw",
           "battery_duration_h", "battery_mwh", "total_h2_kg", "electrolyzer_cf")


def require(condition):
    if not condition:
        raise ValueError("invalid saved campaign")


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def recover(source: Path, output: Path):
    if output.exists() or output.is_symlink():
        raise FileExistsError("refusing to overwrite existing output destination")
    source = source.resolve()
    require(source not in output.resolve().parents)
    raw = {name: (source / name).read_bytes() for name in ("manifest.json", "runs.jsonl")}
    config = json.loads(raw["manifest.json"])["effective_configuration"]
    require(config["hybrid_population"] == POPULATION)
    require(config["n_runs"] == 31 and config["seed_base"] == 42
            and config["nominal_iterations"] == 1000 and config["modes"] == ["ddtw"])
    baselines = config["baselines"]
    require(isinstance(baselines, list) and len(set(baselines)) == len(baselines)
            and all(name in BASELINES for name in baselines))
    selected, seen = [], set()
    lines = raw["runs.jsonl"].splitlines(keepends=True)
    for line in lines:
        record = json.loads(line)
        run, method = record["run"], record["method"]
        require(type(run) is int and 1 <= run <= 31)
        hybrid = method == "Hybrid DDTW"
        require(hybrid or method in baselines)
        require(record["role"] == ("hybrid" if hybrid else "baseline")
                and record["mode"] == ("ddtw" if hybrid else "baseline"))
        offset = 0 if hybrid else 1000 * (BASELINES.index(method) + 1)
        require(type(record["seed"]) is int and record["seed"] == 42 + run + offset)
        require((method, run) not in seen)
        seen.add((method, run))
        require(type(record["feasible"]) is bool)
        require(all(finite(record[key]) for key in
                    ("objective", "physical_lcoe", "physical_lcoh", "physical_agsr", "elapsed_seconds")))
        require(record["result_elapsed_seconds"] is None or finite(record["result_elapsed_seconds"]))
        require(isinstance(record["solution"], list) and len(record["solution"]) == 4
                and all(finite(value) for value in record["solution"]))
        require(all(finite(record["decoded"][key]) for key in DECODED))
        require(record["nominal_iterations"] == 1000
                and type(record["actual_iterations"]) is int
                and record["actual_iterations"] >= 1000
                and record["overshoot_iterations"] == record["actual_iterations"] - 1000)
        if hybrid:
            selected.append((record, line))
    require(len(selected) == 31)
    selected.sort(key=lambda pair: pair[0]["run"])
    records = [pair[0] for pair in selected]
    values = [record["physical_lcoe"] for record in records]
    feasible = [record for record in records if record["feasible"]]
    summary = {
        "report_only": True, "original_campaign_completed": False,
        "sources": {name: {"path": str(source / name), "sha256": hashlib.sha256(data).hexdigest()}
                    for name, data in raw.items()},
        "source_record_count": len(lines), "ddtw_record_count": 31,
        "excluded_baseline_count": len(lines) - 31,
        "hybrid_population": config["hybrid_population"],
        "physical_lcoe": {"mean": statistics.mean(values), "sample_std": statistics.stdev(values),
                          "min": min(values), "max": max(values), "units": "CNY/kWh"},
        "feasible_count": len(feasible), "infeasible_count": 31 - len(feasible),
        "best_feasible_record": min(feasible, key=lambda r: (r["physical_lcoe"], r["run"])) if feasible else None,
        "unavailable_telemetry": ["convergence", "DDTW", "Gantt", "switches"],
        "note": "Original campaign not completed; scalar DDTW recovery only. No optimization, inference or telemetry reconstruction. Statistics include all 31 physical LCOEs; best is the minimum feasible LCOE, ties by run ID.",
    }
    # Separate derived report format: never import the optimizer-backed canonical writer.
    table = io.StringIO(newline="")
    writer = csv.DictWriter(table, fieldnames=list(records[0]))
    writer.writeheader()
    for record in records:
        writer.writerow({key: json.dumps(value, allow_nan=False) if isinstance(value, (dict, list)) else value
                         for key, value in record.items()})
    summary_json = json.dumps(summary, indent=2, allow_nan=False) + "\n"
    metrics = summary["physical_lcoe"]
    best = summary["best_feasible_record"]
    markdown = ("# ASOC DDTW scalar recovery\n\n" + summary["note"] + "\n\n"
                f"- DDTW: 31 records; excluded baselines: {len(lines) - 31}.\n"
                f"- Feasible: {len(feasible)}/31.\n"
                f"- Physical LCOE (CNY/kWh): mean {metrics['mean']:.17g}; sample std (ddof=1) {metrics['sample_std']:.17g}; min {metrics['min']:.17g}; max {metrics['max']:.17g}.\n"
                "- Unavailable telemetry: convergence, DDTW, Gantt, switches.\n"
                "- Source paths and SHA-256 hashes: see summary.json.\n")
    if best:
        markdown += (f"\n## Best feasible saved solution: run {best['run']}, seed {best['seed']}\n\n"
                     f"Physical LCOE: {best['physical_lcoe']}; LCOH: {best['physical_lcoh']} CNY/kg; AGSR: {best['physical_agsr']}.\n\n"
                     + "\n".join(f"- {key}: {value}" for key, value in best["decoded"].items()) + "\n")
    # Detect source changes during validation, before creating any report output.
    require(all((source / name).read_bytes() == data for name, data in raw.items()))
    output.mkdir(parents=True, exist_ok=False)
    (output / "runs.jsonl").write_bytes(b"".join(pair[1] for pair in selected))
    (output / "runs.csv").write_text(table.getvalue(), encoding="utf-8")
    (output / "summary.json").write_text(summary_json, encoding="utf-8")
    (output / "summary.md").write_text(markdown, encoding="utf-8")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Report-only ASOC DDTW recovery; never runs optimization.")
    parser.add_argument("--source", required=True, type=Path, help="Saved manifest.json and runs.jsonl; read-only.")
    parser.add_argument("--output", required=True, type=Path, help="Fresh report directory; existing destinations are rejected.")
    args = parser.parse_args(argv)
    try:
        summary = recover(args.source, args.output)
    except FileExistsError:
        print("error: refusing to overwrite existing output destination", file=sys.stderr)
        return 2
    except (ValueError, KeyError, TypeError, OverflowError):
        print("error: invalid saved campaign", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"Recovered 31 Hybrid DDTW records; excluded {summary['excluded_baseline_count']} baseline records.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
