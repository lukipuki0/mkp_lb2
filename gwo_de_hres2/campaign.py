"""
Multi-seed campaign of GWO-DE on HRES2: every signal x policy x seed.

Usage (from the repository root, with an interpreter that has pandas):
    python -m gwo_de_hres2.campaign --out gwo_de_hres2/results/campaign_full

Defaults: signals dtw and ddtw, seeds 1..31, the six policies of
gwo_de_dtw.CAMPAIGN_POLICIES and the settings of config.py, i.e.
2 x 6 x 31 = 372 runs. Every run calls control.run_search exactly as run.py
does. The HRES2 function (synthetic TMY weather) is built once and shared.

Everything is checked before the first write: unique seeds, pop >= 4,
iterations >= 1 and the output folder (absent or empty; default: new
campaign_<YYYYmmdd-HHMMSS> in gwo_de_hres2/results/). --dry-run stops after
these checks and prints the plan.

Layout of the campaign folder:
    campaign_config.json   settings, total/completed runs, status
    runs.csv / runs.jsonl  one row / record per run (no series), written after each run
    statistics.csv         one row per signal and policy, on `objective`
    tests.csv              two-sided Mann-Whitney U with Holm adjustment
    resumen.txt            Spanish readable summary
    <signal>/<policy>/best_run.json        full record of the best run
    <signal>/convergence_comparison.png    best run of each policy

Best run = lowest objective; ties go to the lower seed. Statistics use the
objective (physical LCOE when feasible, the HRES2 penalty otherwise; the
feasible count is reported). sd is the sample standard deviation (n - 1).
Test families, each Holm-adjusted on its own: per signal, every pair of
policies; dtw_vs_ddtw, each policy across both signals (only when both
signals run). A pair whose p-value is undefined (identical samples) gets
blank p columns and stays out of the Holm family.

A failing run stops the campaign (status failed, exit 1); finished rows stay
on disk. Summaries are written only when every run finished.

Spec: odd/tasks/gwo-de-hres2.md (S6-S8).
"""

import argparse
import csv
from datetime import datetime
from itertools import combinations
import json
import math
import os
from pathlib import Path
import statistics
import sys
import traceback

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import mannwhitneyu  # noqa: E402

from gwo_de_dtw.policies import CAMPAIGN_POLICIES  # noqa: E402

from . import config  # noqa: E402
from .control import run_search  # noqa: E402
from .run import RESULTS_DIR, load_function  # noqa: E402

SIGNALS = ("dtw", "ddtw")
SEEDS = tuple(range(1, 32))
SERIES_KEYS = ("best_series", "profile_series", "parameter_series", "intensity_series",
               "step_stats", "monitor_config")
RUN_FIELDS = ["signal", "policy", "seed", "objective", "feasible", "physical_lcoe",
              "physical_lcoh", "physical_agsr", "wind_mw", "n_el_units", "battery_mw",
              "battery_duration_h", "changes", "evaluations", "elapsed_seconds"]
STAT_FIELDS = ["signal", "policy", "runs", "feasible_runs", "best", "worst", "mean", "median",
               "sd", "best_seed", "mean_elapsed_seconds"]
TEST_FIELDS = ["family", "a", "b", "n_a", "n_b", "median_a", "median_b", "u", "p_value", "p_holm"]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m gwo_de_hres2.campaign",
                                description="Multi-seed GWO-DE campaign on HRES2 (WPEB).")
    p.add_argument("--signals", nargs="+", choices=SIGNALS, default=list(SIGNALS))
    p.add_argument("--policies", nargs="+", choices=CAMPAIGN_POLICIES,
                   default=list(CAMPAIGN_POLICIES))
    p.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS),
                   help="(default: 1..31)")
    p.add_argument("--pop-size", type=int, default=config.POP_SIZE)
    p.add_argument("--iterations", type=int, default=config.ITERATIONS)
    p.add_argument("--out", type=Path, default=None,
                   help="absent or empty folder (default: gwo_de_hres2/results/campaign_<timestamp>)")
    p.add_argument("--dry-run", action="store_true", help="validate and print the plan only")
    return p


def validate(parser, args) -> None:
    for name in ("signals", "policies", "seeds"):
        values = getattr(args, name)
        dup = sorted({v for v in values if values.count(v) > 1})
        if dup:
            parser.error(f"duplicate {name[:-1]}: {dup[0]}")
    if args.pop_size < 4:
        parser.error("--pop-size must be >= 4")
    if args.iterations < 1:
        parser.error("--iterations must be >= 1")
    if args.out is None:
        args.out = Path(RESULTS_DIR) / f"campaign_{datetime.now():%Y%m%d-%H%M%S}"
    if args.out.exists() and (not args.out.is_dir() or any(args.out.iterdir())):
        parser.exit(2, f"error: output directory is not empty: {args.out}\n")


def holm(pvalues):
    """Holm step-down adjusted p-values, in input order, capped at 1."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _blank(value):
    return "" if value is None else value


def run_row(rec: dict) -> dict:
    d = rec["decoded"]
    return {"signal": rec["signal"], "policy": rec["policy"], "seed": rec["seed"],
            "objective": rec["objective"], "feasible": rec["feasible"],
            "physical_lcoe": _blank(rec["physical_lcoe"]),
            "physical_lcoh": _blank(rec["physical_lcoh"]),
            "physical_agsr": _blank(rec["physical_agsr"]),
            "wind_mw": _blank(d.get("wind_mw")), "n_el_units": _blank(d.get("n_el_units")),
            "battery_mw": _blank(d.get("battery_mw")),
            "battery_duration_h": _blank(d.get("battery_duration_h")),
            "changes": len(rec["changes"]), "evaluations": rec["evaluations"],
            "elapsed_seconds": rec["elapsed_seconds"]}


def is_better(rec: dict, best) -> bool:
    if best is None:
        return True
    if rec["objective"] != best["objective"]:
        return rec["objective"] < best["objective"]
    return rec["seed"] < best["seed"]


def group_statistics(signal, policy, rows, best) -> dict:
    values = [r["objective"] for r in rows]
    return {"signal": signal, "policy": policy, "runs": len(rows),
            "feasible_runs": sum(1 for r in rows if r["feasible"] is True),
            "best": min(values), "worst": max(values), "mean": statistics.mean(values),
            "median": statistics.median(values),
            "sd": statistics.stdev(values) if len(values) > 1 else "",
            "best_seed": best["seed"],
            "mean_elapsed_seconds": statistics.mean(r["elapsed_seconds"] for r in rows)}


def _family(name, pairs) -> list:
    rows = []
    for a_name, b_name, a, b in pairs:
        row = {"family": name, "a": a_name, "b": b_name, "n_a": len(a), "n_b": len(b),
               "median_a": statistics.median(a), "median_b": statistics.median(b),
               "u": "", "p_value": "", "p_holm": ""}
        res = mannwhitneyu(a, b, alternative="two-sided")
        if math.isfinite(res.pvalue):
            row["u"], row["p_value"] = float(res.statistic), float(res.pvalue)
        rows.append(row)
    defined = [r for r in rows if r["p_value"] != ""]
    for row, adj in zip(defined, holm([r["p_value"] for r in defined])):
        row["p_holm"] = adj
    return rows


def pairwise_tests(values: dict, signals, policies) -> list:
    """values[(signal, policy)] -> objectives in seed order."""
    rows = []
    for s in signals:
        rows += _family(s, [(a, b, values[(s, a)], values[(s, b)])
                            for a, b in combinations(policies, 2)])
    if len(signals) == 2:
        s1, s2 = signals
        rows += _family(f"{s1}_vs_{s2}", [(p, p, values[(s1, p)], values[(s2, p)])
                                          for p in policies])
    return rows


def plot_convergence(records, path: Path, signal: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for rec in records:
        series = rec["best_series"]
        ax.plot(range(1, len(series) + 1), series,
                label=f"{rec['policy']} (seed {rec['seed']}, {series[-1]:.5f})")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best-so-far LCOE (CNY/kWh, minimization)")
    ax.set_title(f"GWO-DE on HRES2, {signal.upper()}: best run of each policy")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def render_resumen(args, stats, tests) -> str:
    lines = ["Campa\u00f1a GWO-DE sobre HRES2 (WPEB)",
             f"Se\u00f1ales: {', '.join(args.signals)} | semillas: {len(args.seeds)} | "
             f"N={args.pop_size} | generaciones={args.iterations}",
             "Objetivo: LCOE (CNY/kWh) si la soluci\u00f3n es factible; penalizaci\u00f3n "
             "100 + 10*AGSR si no. Menor es mejor.",
             "Ranking por mejor objetivo; en empate, la semilla menor. sd muestral (n - 1).", ""]
    for signal in args.signals:
        lines.append(f"== {signal.upper()} ==")
        rows = sorted((s for s in stats if s["signal"] == signal),
                      key=lambda s: (s["best"], s["best_seed"]))
        for rank, s in enumerate(rows, 1):
            sd = "no aplica" if s["sd"] == "" else f"{s['sd']:.6f}"
            lines.append(f"{rank}. {s['policy']:<22s} mejor={s['best']:.6f} (semilla "
                         f"{s['best_seed']}) mediana={s['median']:.6f} media={s['mean']:.6f} "
                         f"sd={sd} factibles={s['feasible_runs']}/{s['runs']}")
        lines.append("")
    lines.append("Mann-Whitney bilateral con ajuste de Holm por familia (alfa 0.05).")
    for family in dict.fromkeys(t["family"] for t in tests):
        rows = [t for t in tests if t["family"] == family]
        sig = [t for t in rows if t["p_holm"] != "" and t["p_holm"] < 0.05]
        lines.append(f"Familia {family}: {len(sig)} de {len(rows)} pares significativos.")
        for t in sig:
            lines.append(f"  {t['a']} vs {t['b']}: p_holm={t['p_holm']:.4g} "
                         f"(medianas {t['median_a']:.6f} vs {t['median_b']:.6f})")
    return "\n".join(lines) + "\n"


def _write_csv(path: Path, fields, rows) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate(parser, args)
    total = len(args.signals) * len(args.policies) * len(args.seeds)
    print(f"GWO-DE HRES2 campaign: {len(args.signals)} signals x {len(args.policies)} policies "
          f"x {len(args.seeds)} seeds = {total} runs, N={args.pop_size} T={args.iterations} "
          f"-> {args.out}")
    if args.dry_run:
        return 0

    root = args.out
    root.mkdir(parents=True, exist_ok=True)
    state = {"signals": args.signals, "policies": args.policies, "seeds": args.seeds,
             "pop_size": args.pop_size, "iterations": args.iterations,
             "monitor": {s: config.monitor_config(s == "ddtw").__dict__ for s in args.signals},
             "objective": "HRES2Function._evaluate (LCOE, penalty if infeasible)",
             "weather": "generate_synthetic_tmy_8760",
             "total_runs": total, "completed_runs": 0, "status": "running",
             "started": datetime.now().isoformat(timespec="seconds")}
    _write_json(root / "campaign_config.json", state)

    rows_by_group, best_by_group = {}, {}
    try:
        func = load_function()
        with open(root / "runs.csv", "w", newline="", encoding="utf-8") as csv_fh, \
                open(root / "runs.jsonl", "w", encoding="utf-8") as jsonl_fh:
            writer = csv.DictWriter(csv_fh, fieldnames=RUN_FIELDS)
            writer.writeheader()
            csv_fh.flush()
            for signal in args.signals:
                monitor_cfg = config.monitor_config(signal == "ddtw")
                for policy in args.policies:
                    key = (signal, policy)
                    rows_by_group[key] = []
                    for seed in args.seeds:
                        rec = run_search(func, monitor_cfg, policy=policy,
                                         pop_size=args.pop_size, iterations=args.iterations,
                                         seed=seed)
                        row = run_row(rec)
                        writer.writerow(row)
                        csv_fh.flush()
                        jsonl_fh.write(json.dumps({k: v for k, v in rec.items()
                                                   if k not in SERIES_KEYS}) + "\n")
                        jsonl_fh.flush()
                        rows_by_group[key].append(row)
                        if is_better(rec, best_by_group.get(key)):
                            best_by_group[key] = rec
                        state["completed_runs"] += 1
                        _write_json(root / "campaign_config.json", state)
                        print(f"  [{state['completed_runs']}/{total}] {signal} {policy} "
                              f"seed={seed} objective={rec['objective']:.6f} "
                              f"feasible={rec['feasible']} time={rec['elapsed_seconds']:.1f}s")
    except Exception:
        state["status"] = "failed"
        _write_json(root / "campaign_config.json", state)
        traceback.print_exc()
        return 1

    stats = []
    for (signal, policy), rows in rows_by_group.items():
        best = best_by_group[(signal, policy)]
        _write_json(root / signal / policy / "best_run.json", best)
        stats.append(group_statistics(signal, policy, rows, best))
    for signal in args.signals:
        plot_convergence([best_by_group[(signal, p)] for p in args.policies],
                         root / signal / "convergence_comparison.png", signal)
    values = {k: [r["objective"] for r in rows] for k, rows in rows_by_group.items()}
    tests = pairwise_tests(values, args.signals, args.policies)
    _write_csv(root / "statistics.csv", STAT_FIELDS, stats)
    _write_csv(root / "tests.csv", TEST_FIELDS, tests)
    (root / "resumen.txt").write_text(render_resumen(args, stats, tests), encoding="utf-8")
    state["status"] = "completed"
    state["finished"] = datetime.now().isoformat(timespec="seconds")
    _write_json(root / "campaign_config.json", state)
    print(f"Completed {state['completed_runs']}/{total} runs -> {root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
