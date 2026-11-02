"""Additive v0, v1 and v4 regressions for the shared-population WOA-ABC hybrid.

v0 is the unchanged baseline. Each later variant changes one control only and
keeps the stagnation reaction.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import numpy as np

import woa_abc.solver as hybrid
from dtw_stagnation import StagnationConfig
from mkp_core.problem import MKPInstance
from woa_abc.solver import (
    PHASE_EMPLOYED,
    PHASE_ONLOOKERS,
    PHASE_WOA,
    HybridConfig,
    _abc_step,
    main,
    solve,
)

VARIANT_ERROR = "variant must be one of v0, v1, v4"
CLI_VARIANT_ERROR = f"error: {VARIANT_ERROR}\n"


def _instance() -> MKPInstance:
    return MKPInstance(
        n=3,
        m=1,
        valor_optimo=9.0,
        p=np.array([5.0, 4.0, 3.0]),
        r=np.array([[2.0, 2.0, 2.0]]),
        b=np.array([4.0]),
    )


def _config(**overrides: object) -> HybridConfig:
    values: dict[str, object] = {
        "population_size": 2,
        "iterations": 4,
        "max_evaluations": 10_000,
        "use_stagnation": False,
    }
    values.update(overrides)
    return HybridConfig(**values)  # type: ignore[arg-type]


def _strip_variant(payload: dict) -> dict:
    config = payload["metadata"]["config"]
    config.pop("variant", None)
    return payload


class TestVariantSelection(unittest.TestCase):
    def test_default_and_explicit_v0_match_and_omit_variant_metadata(self) -> None:
        inst = _instance()
        omitted = solve(inst, _config(), seed=5)
        explicit = solve(inst, _config(variant="v0"), seed=5)

        self.assertEqual(omitted.scout_limit, 2 * inst.n)
        self.assertNotIn("variant", omitted.metadata["config"])
        self.assertNotIn("variant", explicit.metadata["config"])
        self.assertEqual(asdict(omitted), asdict(explicit))
        self.assertEqual(HybridConfig().variant, "v0")

    def test_invalid_variant_is_rejected_without_mutating_other_fields(self) -> None:
        for value in ("v2", "v3", "v5", "V1", "", None, True, 1):
            with self.subTest(value=value):
                with self.assertRaises(ValueError) as caught:
                    _config(variant=value)
                self.assertEqual(str(caught.exception), VARIANT_ERROR)

    def test_nonbaseline_metadata_names_only_the_selected_variant(self) -> None:
        inst = _instance()
        baseline = solve(inst, _config(scout_limit=100), seed=2)
        for variant in ("v1", "v4"):
            with self.subTest(variant=variant):
                result = solve(inst, _config(variant=variant, scout_limit=100), seed=2)
                self.assertEqual(result.metadata["config"]["variant"], variant)
                stripped = _strip_variant(asdict(result))
                if variant == "v4":
                    self.assertEqual(stripped["metadata"], asdict(baseline)["metadata"])
                    self.assertEqual(stripped["scout_limit"], baseline.scout_limit)
                else:
                    self.assertEqual(stripped, asdict(baseline))


class TestV1ScoutLimit(unittest.TestCase):
    def test_v1_default_scout_limit_is_100_and_matches_explicit_v0(self) -> None:
        inst = _instance()
        v1 = solve(
            inst,
            _config(variant="v1", population_size=3, iterations=2, max_evaluations=None),
            seed=4,
        )
        v0 = solve(
            inst,
            _config(
                variant="v0",
                population_size=3,
                iterations=2,
                max_evaluations=None,
                scout_limit=100,
            ),
            seed=4,
        )

        self.assertEqual(v1.scout_limit, 100)
        self.assertNotEqual(v1.scout_limit, 3 * inst.n)
        self.assertEqual(v1.metadata["config"]["max_evaluations"], 3 * (1 + 7 * 2))
        self.assertEqual(v1.metadata["config"]["variant"], "v1")
        self.assertEqual(_strip_variant(asdict(v1)), asdict(v0))
        self.assertIsNone(_config(variant="v1").scout_limit)

    def test_explicit_scout_limit_still_overrides_v1(self) -> None:
        inst = _instance()
        config = _config(variant="v1", scout_limit=1, iterations=3)
        before = config.scout_limit
        v1 = solve(inst, config, seed=6)
        v0 = solve(inst, _config(variant="v0", scout_limit=1, iterations=3), seed=6)

        self.assertEqual(before, 1)
        self.assertEqual(config.scout_limit, 1)
        self.assertEqual(v1.scout_limit, 1)
        self.assertGreater(v1.evaluations_by_phase["scouts"], 0)
        self.assertEqual(_strip_variant(asdict(v1)), asdict(v0))


class TestV4SingleCoordinate(unittest.TestCase):
    def test_v4_perturbs_one_coordinate_in_both_abc_phases_and_keeps_acceptance(self) -> None:
        inst = _instance()
        config = _config(variant="v4", population_size=2, v_max=6.0)
        state = hybrid._RunState(
            inst=inst,
            config=config,
            rng=np.random.default_rng(0),
            positions=np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=float),
            bits=[[1, 0, 0], [0, 1, 0]],
            fitness=[1.0, 4.0],
            trials=np.array([2, 5], dtype=int),
            max_evaluations=20,
            scout_limit=9,
        )
        state.by_phase = {phase: 0 for phase in hybrid.PHASES}

        class _Script:
            def __init__(self, draws: list[int]) -> None:
                self._draws = iter(draws)

            def integers(self, low: int, high: int) -> int:
                del low, high
                return next(self._draws)

            def uniform(self, low: float, high: float) -> float:
                self.low = low
                self.high = high
                return 1.0

        captured: list[tuple[str, np.ndarray]] = []

        def fake_lb2(run, position, bits, phase):
            captured.append((phase, np.array(position, copy=True)))
            return list(bits), 0.0

        state.rng = _Script([0, 2])
        with patch.object(hybrid, "_lb2_pair", fake_lb2):
            _abc_step(state, 0, 1.5, PHASE_EMPLOYED)
        self.assertEqual(captured[0][0], PHASE_EMPLOYED)
        np.testing.assert_allclose(captured[0][1], [0.0, 0.0, -3.0])
        self.assertEqual(state.bits[0], [1, 0, 0])
        self.assertEqual(int(state.trials[0]), 3)
        np.testing.assert_allclose(state.positions[0], [0.0, 0.0, 0.0])
        self.assertEqual(state.rng.low, -1.5)
        self.assertEqual(state.rng.high, 1.5)

        def improved(run, position, bits, phase):
            captured.append((phase, np.array(position, copy=True)))
            return [1, 1, 0], 9.0

        state.rng = _Script([0, 0])
        with patch.object(hybrid, "_lb2_pair", improved):
            _abc_step(state, 1, 1.0, PHASE_ONLOOKERS)
        self.assertEqual(captured[1][0], PHASE_ONLOOKERS)
        np.testing.assert_allclose(captured[1][1], [2.0, 2.0, 3.0])
        self.assertEqual(state.bits[1], [1, 1, 0])
        self.assertEqual(state.fitness[1], 9.0)
        self.assertEqual(int(state.trials[1]), 0)
        np.testing.assert_allclose(state.positions[1], [2.0, 2.0, 3.0])

    def test_v4_solve_keeps_rho_budget_and_full_lb2_while_abc_changes_one_coordinate(self) -> None:
        inst = _instance()
        baseline = solve(inst, _config(variant="v0", iterations=3), seed=2)
        captured: list[tuple[str, np.ndarray, np.ndarray]] = []
        real_pair = hybrid._lb2_pair
        real_l1 = hybrid._calcular_L1
        l1_lengths: list[int] = []

        def spy_pair(state, position, bits, phase):
            captured.append(
                (phase, np.array(position, copy=True), np.array(state.positions, copy=True))
            )
            return real_pair(state, position, bits, phase)

        def spy_l1(position, *args, **kwargs):
            l1_lengths.append(int(len(position)))
            return real_l1(position, *args, **kwargs)

        with patch.object(hybrid, "_lb2_pair", spy_pair), patch.object(hybrid, "_calcular_L1", spy_l1):
            result = solve(inst, _config(variant="v4", iterations=3), seed=2)

        self.assertEqual(result.metadata["config"]["variant"], "v4")
        self.assertEqual(result.scout_limit, 2 * inst.n)
        self.assertEqual(baseline.scout_limit, 2 * inst.n)
        self.assertEqual(
            [(record.profile, record.a, record.rho, record.G3) for record in result.history],
            [(record.profile, record.a, record.rho, record.G3) for record in baseline.history],
        )
        self.assertEqual(sum(result.evaluations_by_phase.values()), result.evaluations)
        self.assertGreater(result.evaluations_by_phase[PHASE_EMPLOYED], 0)
        self.assertGreater(result.evaluations_by_phase[PHASE_ONLOOKERS], 0)
        self.assertTrue(l1_lengths)
        self.assertTrue(all(length == inst.n for length in l1_lengths))

        abc = [item for item in captured if item[0] in {PHASE_EMPLOYED, PHASE_ONLOOKERS}]
        woa = [item for item in captured if item[0] == PHASE_WOA]
        self.assertTrue(abc)
        self.assertTrue(woa)
        single_changes = 0
        for _phase, proposal, positions in abc:
            self.assertEqual(len(proposal), inst.n)
            closest = min(
                int(np.count_nonzero(~np.isclose(proposal, row))) for row in positions
            )
            self.assertLessEqual(closest, 1)
            if closest == 1:
                single_changes += 1
        self.assertGreater(single_changes, 0)
        self.assertTrue(
            any(
                min(int(np.count_nonzero(~np.isclose(proposal, row))) for row in positions) >= 2
                for _phase, proposal, positions in woa
            )
        )


class TestVariantsKeepTheReaction(unittest.TestCase):
    def test_v1_and_v4_change_profile_on_the_same_cycles_as_v0(self) -> None:
        # Every repair fills the knapsack, so the archived best is constant
        # whatever the variant draws.
        constant = MKPInstance(
            n=3,
            m=1,
            valor_optimo=6.0,
            p=np.array([1.0, 2.0, 3.0]),
            r=np.array([[1.0, 1.0, 1.0]]),
            b=np.array([10.0]),
        )
        monitor = dict(window=4, band=1, min_slope=2.0, plateau_max=2, patience=2, adapt_thresholds=False)
        results = {}
        for variant in ("v0", "v1", "v4"):
            config = _config(
                variant=variant,
                iterations=8,
                scout_limit=100,
                use_stagnation=True,
                stag_cfg=StagnationConfig(**monitor),
            )
            results[variant] = solve(constant, config, seed=1)
        for variant, result in results.items():
            with self.subTest(variant=variant):
                self.assertEqual(result.profile_changes, [4])
                self.assertEqual(
                    [record.profile for record in result.history],
                    ["exploit"] * 5 + ["explore"] * 3,
                )


class TestVariantCommandLine(unittest.TestCase):
    def _file(self, directory: str) -> Path:
        path = Path(directory) / "tiny.dat"
        path.write_text("1\n3 1 9\n5 4 3\n2 2 2\n4\n", encoding="utf-8")
        return path

    def test_solver_help_and_default_v0_cli_match_the_omitted_flag(self) -> None:
        stdout = StringIO()
        with redirect_stdout(stdout):
            self.assertEqual(main(["--help"]), 0)
        self.assertIn("--variant", stdout.getvalue())
        self.assertIn("v0", stdout.getvalue())
        self.assertNotIn("v2", stdout.getvalue())
        self.assertNotIn("v3", stdout.getvalue())

        with tempfile.TemporaryDirectory() as directory:
            path = self._file(directory)
            argv = [
                "--instance",
                str(path),
                "--seed",
                "1",
                "--population-size",
                "2",
                "--iterations",
                "2",
                "--max-evaluations",
                "40",
                "--no-stagnation",
            ]
            omitted = StringIO()
            explicit = StringIO()
            with redirect_stdout(omitted):
                omitted_code = main(argv)
            with redirect_stdout(explicit):
                explicit_code = main([*argv, "--variant", "v0"])

        self.assertEqual(omitted_code, 0)
        self.assertEqual(explicit_code, 0)
        self.assertEqual(omitted.getvalue(), explicit.getvalue())
        payload = json.loads(omitted.getvalue())
        self.assertNotIn("variant", payload["metadata"]["config"])
        self.assertEqual(payload["metadata"]["config"]["scout_limit"], 6)

    def test_solver_rejects_an_invalid_variant_without_solving(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._file(directory)
            stdout = StringIO()
            stderr = StringIO()
            with patch.object(hybrid, "solve", side_effect=AssertionError("solver invoked")):
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    code = main(
                        [
                            "--instance",
                            str(path),
                            "--seed",
                            "1",
                            "--variant",
                            "v9",
                        ]
                    )
        self.assertEqual(code, 1)
        self.assertEqual(stdout.getvalue(), "")
        self.assertEqual(stderr.getvalue(), CLI_VARIANT_ERROR)

    def test_benchmark_variant_cli_rejects_cleanly_and_names_nonbaseline_exports(self) -> None:
        import woa_abc.benchmark as bench

        help_text = bench.build_parser().format_help()
        self.assertIn("--variant", help_text)
        self.assertNotIn("--runs", help_text)
        args = bench.build_parser().parse_args(["--instance", "instancias/mknapcb1.txt"])
        self.assertEqual(args.variant, "v0")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stdout = StringIO()
            stderr = StringIO()
            with patch.object(bench, "export_single_run", side_effect=AssertionError("export invoked")):
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    code = bench.main(
                        [
                            "--instance",
                            "https://example.test/mknapcb1.txt",
                            "--variant",
                            "nope",
                            "--output-root",
                            str(root),
                        ]
                    )
            self.assertEqual(code, 1)
            self.assertEqual(stdout.getvalue(), "")
            self.assertEqual(stderr.getvalue(), CLI_VARIANT_ERROR)
            self.assertEqual(list(root.iterdir()), [])

            path = root / "tiny.txt"
            path.write_text("1\n3 1 9\n5 4 3\n2 2 2\n4\n", encoding="utf-8")
            outcome = bench.export_single_run(
                bench.ExportRequest(
                    instance=path,
                    seed=3,
                    population_size=2,
                    iterations=2,
                    max_evaluations=40,
                    output_root=root / "out",
                    timestamp="variantv1",
                    variant="v1",
                )
            )
            self.assertEqual(outcome.code, 0, outcome.message)
            raw = json.loads((outcome.output_dir / "runs" / "run_01.json").read_text(encoding="utf-8"))
            manifest = json.loads((outcome.output_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(raw["metadata"]["config"]["variant"], "v1")
            self.assertEqual(manifest["effective_config"]["variant"], "v1")
            self.assertEqual(raw["scout_limit"], 100)
            self.assertEqual(manifest["effective_config"]["scout_limit"], 100)
            self.assertEqual(manifest["runs"], 1)

            with self.assertRaises(ValueError) as caught:
                bench.export_single_run(
                    bench.ExportRequest(
                        instance=path,
                        output_root=root / "rejected",
                        timestamp="bad",
                        variant="v9",
                    )
                )
            self.assertEqual(str(caught.exception), VARIANT_ERROR)
            self.assertFalse((root / "rejected").exists())


if __name__ == "__main__":
    unittest.main()
