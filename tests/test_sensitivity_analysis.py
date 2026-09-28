"""Functional checks for seed pairing and transparent pilot inference."""
import unittest

from sensitivity.analysis import build_reports, holm_adjust


def row(seed, config="base", value=10.0, domain="cec", case="F1", mode="dtw", **overrides):
    result = dict(domain=domain, case={"id": case}, mode=mode, budget=1000,
                  max_epoch_iters=None, config_id=config, seed=seed,
                  is_baseline=config == "base", factor="baseline" if config == "base" else "window",
                  level=None if config == "base" else 20, status="success", feasible=True,
                  metric=value, metric_name={"cec": "optimum_error", "mkp": "gap_pct",
                                            "hres": "lcoe_cny_per_kwh"}[domain],
                  elapsed_seconds=1.0, actual_iters=1000)
    result.update(overrides)
    return result


class AnalysisTests(unittest.TestCase):
    def test_holm_order_monotone(self):
        self.assertEqual(holm_adjust([0.03, 0.01, 0.04]), [0.06, 0.03, 0.06])
        self.assertEqual(holm_adjust([]), [])
        with self.assertRaises(ValueError):
            holm_adjust([float("nan")])

    def test_paired_improvement_and_reproducible_interval(self):
        records = [row(s, value=10 + s) for s in range(8)]
        records += [row(s, "alt", value=8 + s) for s in reversed(range(8))]
        _, comparisons = build_reports(records, resamples=100)
        result = comparisons[0]
        self.assertEqual(result["mean_improvement"], 2)
        self.assertEqual(result["rank_biserial"], 1)
        self.assertEqual(result["n_pairs"], 8)
        self.assertTrue(result["coverage_complete"])
        self.assertLess(result["p_holm"], 0.05)
        self.assertEqual(result["mean_improvement_ci_low"], 2)
        self.assertEqual(build_reports(records, resamples=100)[1], comparisons)

    def test_all_zero_is_no_detected_difference_not_equivalence(self):
        records = [row(s, cfg) for s in range(3) for cfg in ("base", "alt")]
        _, comparisons = build_reports(records, resamples=100)
        self.assertEqual(comparisons[0]["p_holm"], 1)
        self.assertEqual(comparisons[0]["rank_biserial"], 0)
        self.assertFalse(comparisons[0]["significant_holm"])

    def test_feasibility_failures_and_missing_pairs_are_explicit(self):
        tasks = [row(s, cfg, domain="hres") for s in range(4) for cfg in ("base", "alt")]
        records = tasks[:-2]
        records[3] = dict(records[3], feasible=False, metric=None)
        records[5] = dict(records[5], status="failed", metric=None)
        summaries, comparisons = build_reports(records, tasks, resamples=100)
        result = comparisons[0]
        self.assertEqual(result["n_pairs"], 1)
        self.assertEqual(result["n_excluded"], 2)
        self.assertEqual(result["status"], "insufficient_pairs")
        self.assertIsNone(result["p_raw"])
        alt = next(r for r in summaries if r["config_id"] == "alt")
        self.assertEqual(alt["n_failed"], 1)
        self.assertEqual(alt["n_missing"], 1)
        self.assertEqual(alt["feasibility_rate"], 0.5)

    def test_domains_and_modes_are_not_pooled(self):
        records = [row(s, cfg, domain=domain, mode=mode)
                   for domain in ("cec", "mkp") for mode in ("dtw", "ddtw")
                   for s in range(2) for cfg in ("base", "alt")]
        summaries, comparisons = build_reports(records, resamples=100)
        self.assertEqual(len(summaries), 8)
        self.assertEqual(len(comparisons), 4)
        self.assertTrue(all(r["family_size"] == 1 for r in comparisons))

    def test_planned_missing_comparisons_still_count_in_holm_family(self):
        tasks = [row(s, cfg, case=case) for s in range(3)
                 for cfg in ("base", "alt") for case in ("F1", "F2")]
        records = [r for r in tasks if r["case"]["id"] == "F1"]
        summaries, comparisons = build_reports(records, tasks, resamples=100)
        self.assertTrue(all(r["family_size"] == 2 for r in comparisons))
        self.assertTrue(any(r["n_missing"] == 3 for r in summaries))
        self.assertIsNone(next(r for r in comparisons if r["case_id"] == "F2")["p_holm"])

    def test_duplicate_seed_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            build_reports([row(43), row(43)], resamples=100)

    def test_unknown_gap_not_substituted_by_objective(self):
        records = [row(s, cfg, domain="mkp", metric=None)
                   for s in range(2) for cfg in ("base", "alt")]
        summaries, comparisons = build_reports(records, resamples=100)
        self.assertTrue(all(r["mean"] is None for r in summaries))
        self.assertIsNone(comparisons[0]["p_raw"])


if __name__ == "__main__":
    unittest.main()
