"""Domain-separated, seed-paired pilot sensitivity reports (lower metric is better)."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, wilcoxon


def holm_adjust(pvalues: list[float]) -> list[float]:
    """Holm step-down adjusted p-values in the original comparison order."""
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in pvalues):
        raise ValueError("p-values must be finite and between zero and one")
    order = sorted(range(len(pvalues)), key=pvalues.__getitem__)
    adjusted = [0.0] * len(pvalues)
    previous = 0.0
    for rank, index in enumerate(order):
        previous = max(previous, min(1.0, (len(order) - rank) * pvalues[index]))
        adjusted[index] = previous
    return adjusted


def _group_key(row: dict) -> tuple:
    return (row["domain"], row["case"]["id"], row["mode"], row["budget"],
            row.get("max_epoch_iters"))


def _valid_metric(row: dict) -> bool:
    value = row.get("metric")
    return (row.get("status") == "success" and row.get("feasible") is True
            and isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def _mean_interval(values: np.ndarray, seed: int, resamples: int) -> tuple:
    """Unadjusted percentile bootstrap of the mean, over paired seed differences."""
    if len(values) < 2:
        return None, None
    rng = np.random.default_rng(seed)
    means = np.mean(rng.choice(values, (resamples, len(values)), replace=True), axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def build_reports(records: list[dict], planned_tasks: list[dict] | None = None,
                  resamples: int = 2000) -> tuple[list[dict], list[dict]]:
    """Compare alternatives to same-mode baseline within a case and budget.

    Quality inference is conditional on BOTH solutions being feasible. Feasibility
    rates and missing/failed pairs are reported separately, not hidden as ties.
    Holm families cover all cases/configurations in a domain/mode/budget/metric.
    No cross-domain or cross-case raw objectives are pooled.
    """
    if not isinstance(resamples, int) or resamples < 100:
        raise ValueError("resamples must be an integer >= 100")
    configurations = defaultdict(dict)
    for task in planned_tasks or records:
        key = _group_key(task)
        configurations[key].setdefault(task["config_id"], task)
    observed = defaultdict(dict)
    for record in records:
        group = _group_key(record)
        identity = (group, record["config_id"])
        seed = record["seed"]
        if seed in observed[identity]:
            raise ValueError("duplicate configuration/case/seed result")
        if record["config_id"] not in configurations[group]:
            raise ValueError("result does not belong to the planned configurations")
        observed[identity][seed] = record

    planned = defaultdict(set)
    for task in planned_tasks or records:
        planned[(_group_key(task), task["config_id"])].add(task["seed"])

    summaries, comparisons = [], []
    for group, configs in sorted(configurations.items(), key=lambda item: str(item[0])):
        domain, case_id, mode, budget, cap = group
        baselines = [cfg for cfg, row in configs.items() if row["is_baseline"]]
        if len(baselines) != 1:
            raise ValueError("each case/mode/budget must have exactly one baseline")
        baseline_id = baselines[0]
        base = observed[(group, baseline_id)]
        for config_id, task in configs.items():
            runs = observed[(group, config_id)]
            success = [r for r in runs.values() if r.get("status") == "success"]
            valid = [r for r in success if _valid_metric(r)]
            values = np.asarray([r["metric"] for r in valid], dtype=float)
            names = {r["metric_name"] for r in success}
            if len(names) > 1:
                raise ValueError("metric name changes within a configuration")
            expected_names = {"mkp": "gap_pct", "cec": "optimum_error",
                              "hres": "lcoe_cny_per_kwh"}
            metric_name = next(iter(names), expected_names[domain])
            if metric_name != expected_names[domain]:
                raise ValueError("unexpected domain metric")
            common = dict(domain=domain, case_id=case_id, mode=mode, budget=budget,
                          max_epoch_iters=cap, metric_name=metric_name)
            summaries.append(dict(
                **common, config_id=config_id, factor=task["factor"], level=task["level"],
                is_baseline=task["is_baseline"], n_planned=len(planned[(group, config_id)]),
                n_success=len(success), n_failed=len(runs) - len(success),
                n_missing=len(planned[(group, config_id)] - runs.keys()), n_metric=len(values),
                feasibility_rate=(sum(r.get("feasible") is True for r in success) /
                                  len(success)) if success else None,
                mean=float(np.mean(values)) if len(values) else None,
                median=float(np.median(values)) if len(values) else None,
                std=float(np.std(values, ddof=1)) if len(values) > 1 else None,
                mean_elapsed_seconds=float(np.mean([r["elapsed_seconds"] for r in success]))
                if success else None,
                mean_actual_iters=float(np.mean([r["actual_iters"] for r in success]))
                if success else None,
            ))
            if config_id == baseline_id:
                continue
            matched = sorted(base.keys() & runs.keys())
            pairs = [s for s in matched if _valid_metric(base[s]) and _valid_metric(runs[s])]
            difference = np.array([base[s]["metric"] - runs[s]["metric"] for s in pairs])
            digest = hashlib.sha256(repr((group, config_id)).encode()).hexdigest()
            low, high = _mean_interval(difference, int(digest[:8], 16), resamples)
            nonzero = difference[difference != 0]
            pvalue, rbc = None, None
            if len(difference) >= 2:
                if len(nonzero) == 0:
                    pvalue, rbc = 1.0, 0.0
                else:
                    pvalue = float(wilcoxon(difference, alternative="two-sided",
                                           zero_method="wilcox", method="auto").pvalue)
                    ranks = rankdata(abs(nonzero))
                    rbc = float(np.sum(ranks * np.sign(nonzero)) / np.sum(ranks))
            complete = (set(pairs) == planned[(group, config_id)] ==
                        planned[(group, baseline_id)])
            comparisons.append(dict(
                **common, baseline_config_id=baseline_id, config_id=config_id,
                factor=task["factor"], level=task["level"], n_matched=len(matched),
                n_pairs=len(pairs), n_excluded=len(matched) - len(pairs),
                paired_seeds=pairs, coverage_complete=complete,
                status="insufficient_pairs" if len(pairs) < 2 else
                ("complete" if complete else "partial_or_conditional"),
                mean_improvement=float(np.mean(difference)) if len(pairs) else None,
                median_improvement=float(np.median(difference)) if len(pairs) else None,
                mean_improvement_ci_low=low, mean_improvement_ci_high=high,
                rank_biserial=rbc, p_raw=pvalue, p_holm=None, significant_holm=False,
                family_size=None, quality_scope="paired_feasible_only",
            ))

    families = defaultdict(list)
    for row in comparisons:
        family = (row["domain"], row["mode"], row["budget"], row["max_epoch_iters"],
                  row["metric_name"])
        families[family].append(row)
    for family in families.values():
        # Untested hypotheses still count in the planned family (conservative p=1).
        adjusted = holm_adjust([r["p_raw"] if r["p_raw"] is not None else 1.0 for r in family])
        for row, p in zip(family, adjusted):
            row["family_size"] = len(family)
            if row["p_raw"] is not None:
                row["p_holm"] = p
                row["significant_holm"] = p < 0.05
    return summaries, comparisons


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v
                             for k, v in row.items()})


def _plots(directory: Path, summaries: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    groups = defaultdict(list)
    for row in summaries:
        if row["mean"] is not None:
            groups[(row["domain"], row["case_id"], row["mode"], row["budget"])].append(row)
    directory.mkdir(exist_ok=True)
    for key, rows in groups.items():
        fig, ax = plt.subplots(figsize=(max(7, len(rows) * 0.6), 4.5))
        labels = ["baseline" if r["is_baseline"] else f'{r["factor"]}={r["level"]}' for r in rows]
        ax.plot(range(len(rows)), [r["mean"] for r in rows], "o")
        ax.set_xticks(range(len(rows)), labels, rotation=65, ha="right")
        ax.set_ylabel(f'{rows[0]["metric_name"]} (feasible runs; lower is better)')
        ax.set_title(" / ".join(map(str, key)) + " — pilot sensitivity")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        filename = hashlib.sha256(repr(key).encode()).hexdigest()[:12] + ".png"
        fig.savefig(directory / filename, dpi=150)
        plt.close(fig)


def analyze_campaign(directory: Path, make_plots: bool = True,
                     resamples: int = 2000) -> tuple[list[dict], list[dict]]:
    """Read canonical atomic records, not potentially interrupted CSV exports."""
    from sensitivity.design import expand_tasks
    from sensitivity.run import load_records
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    tasks = expand_tasks(manifest["protocol"])
    summaries, comparisons = build_reports(load_records(directory), tasks, resamples)
    _write_csv(directory / "summary.csv", summaries)
    _write_csv(directory / "statistics.csv", comparisons)
    if make_plots:
        _plots(directory / "figures", summaries)
    settings = {"quality_scope": "paired_feasible_only", "difference": "baseline - alternative",
                "positive_effect": "alternative improves", "test": "Wilcoxon two-sided auto",
                "holm_family": "domain/mode/budget/epoch-cap/metric; all planned cases/alternatives",
                "bootstrap": "paired percentile mean CI, 95%, unadjusted",
                "resamples": resamples, "numpy": np.__version__,
                "warning": "Pilot/smoke output is not evidence of general robustness. Missing or "
                "infeasible runs can bias conditional comparisons. No equivalence inference."}
    (directory / "analysis_settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    return summaries, comparisons


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--resamples", type=int, default=2000)
    args = parser.parse_args()
    summaries, comparisons = analyze_campaign(args.campaign, not args.no_plots, args.resamples)
    print(f"Wrote {len(summaries)} summaries and {len(comparisons)} baseline comparisons.")


if __name__ == "__main__":
    main()
