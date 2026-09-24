from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from new_woa_abc.reporting.statistics import (
    adjust_pvalues_holm,
    save_global_statistical_summary,
    save_paired_statistical_analysis,
)


class TestStatisticalAnalysis(unittest.TestCase):
    def _rows(self) -> list[dict]:
        offsets = {
            "M0_no_dtw": 0.0,
            "M1_fire_d2": -2.0,
            "M2_fire_3cond": 2.0,
            "M4_d2_continuous": -1.5,
            "M5_sigmoid_delta": 1.5,
            "M6_hysteresis_woa_abc": -1.0,
            "M8_four_state_woa_abc": 1.0,
        }
        return [
            {
                "variant": variant,
                "seed": seed,
                "best_cost": 100.0 + seed + offset,
                "optimum_reached": variant == "M1_fire_d2",
            }
            for variant, offset in offsets.items()
            for seed in range(42, 52)
        ]

    def test_holm_step_down(self) -> None:
        adjusted = adjust_pvalues_holm([0.01, 0.03, 0.04])
        self.assertEqual([round(value, 8) for value in adjusted], [0.03, 0.06, 0.06])

    def test_full_paired_analysis_and_global_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            analysis = save_paired_statistical_analysis(
                directory,
                self._rows(),
                metric="best_cost",
                metric_label="Costo",
                title="Problema sintético",
                minimize=True,
            )
            self.assertEqual(analysis["n_runs"], 10)
            self.assertEqual(analysis["n_variants"], 7)
            by_variant = {row["variant"]: row for row in analysis["table"]}
            self.assertEqual(by_variant["M1_fire_d2"]["relation_vs_control"], "better")
            self.assertEqual(by_variant["M2_fire_3cond"]["relation_vs_control"], "worse")
            self.assertGreater(by_variant["M1_fire_d2"]["rank_biserial"], 0.0)
            self.assertEqual(by_variant["M1_fire_d2"]["optimum_hits"], 10)
            for filename in (
                "analisis_estadistico_pvalues.csv",
                "analisis_estadistico_pvalues.json",
                "analisis_estadistico_pvalues.md",
                "boxplot_estadistico.png",
                "boxplot_estadistico.pdf",
            ):
                self.assertTrue((directory / filename).is_file(), filename)

            result = save_global_statistical_summary(
                directory,
                [("P1", analysis), ("P2", analysis)],
                title="Sintético",
            )
            self.assertFalse(result["raw_objectives_combined"])
            self.assertTrue((directory / "analisis_estadistico_global.csv").is_file())
            self.assertTrue((directory / "ranking_estadistico_global.png").is_file())

    def test_maximization_direction(self) -> None:
        rows = [
            {"variant": variant, "seed": seed, "profit": 100 + seed + offset}
            for variant, offset in (
                ("M0_no_dtw", 0),
                ("M1_fire_d2", 2),
                ("M2_fire_3cond", -2),
            )
            for seed in range(10)
        ]
        with tempfile.TemporaryDirectory() as temporary:
            analysis = save_paired_statistical_analysis(
                Path(temporary),
                rows,
                metric="profit",
                metric_label="Beneficio",
                title="Maximización",
                minimize=False,
            )
        by_variant = {row["variant"]: row for row in analysis["table"]}
        self.assertEqual(by_variant["M1_fire_d2"]["relation_vs_control"], "better")
        self.assertEqual(by_variant["M2_fire_3cond"]["relation_vs_control"], "worse")

    def test_requires_exactly_paired_seeds(self) -> None:
        rows = [
            {"variant": "M0_no_dtw", "seed": 1, "value": 1.0},
            {"variant": "M0_no_dtw", "seed": 2, "value": 2.0},
            {"variant": "M1_fire_d2", "seed": 1, "value": 1.0},
            {"variant": "M1_fire_d2", "seed": 3, "value": 2.0},
        ]
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "semillas no están emparejadas"):
                save_paired_statistical_analysis(
                    Path(temporary),
                    rows,
                    metric="value",
                    metric_label="Valor",
                    title="Inválido",
                )


if __name__ == "__main__":
    unittest.main()
