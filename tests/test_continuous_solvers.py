"""Regression coverage for supported solvers after retiring broken WOA-ABC glue."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import random
import subprocess
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SUPPORTED = ("GA", "PSO", "GWO", "WOA", "EHO", "ACO", "ABC")


class ContinuousSolversTests(unittest.TestCase):
    def test_public_imports_no_longer_require_retired_package(self):
        result = subprocess.run(
            [sys.executable, "-B", "-c", "import continuous_benchmark.mh; import continuous_benchmark.orchestrator; print('continuous benchmark import OK')"],
            cwd=ROOT, capture_output=True, text=True, timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "continuous benchmark import OK\n")
        self.assertEqual(result.stderr, "")

    def test_all_seven_existing_standalone_solvers_still_optimize(self):
        from continuous_benchmark.funciones_cec2022 import ContinuousFunction
        from continuous_benchmark.orchestrator import ejecutar_mh_standalone, MH_POBLACIONAL_DISPONIBLES
        self.assertEqual(set(MH_POBLACIONAL_DISPONIBLES), set(SUPPORTED))
        sphere = ContinuousFunction("sphere", lambda x: float(np.sum(x * x)), -5, 5, 0, 10)
        python_state, numpy_state = random.getstate(), np.random.get_state()
        try:
            for solver in SUPPORTED:
                with self.subTest(solver=solver):
                    random.seed(17)
                    np.random.seed(17)
                    stdout = io.StringIO()
                    with redirect_stdout(stdout):
                        result = ejecutar_mh_standalone(sphere, solver, max_iters=1)
                    self.assertEqual(stdout.getvalue(), "")
                    solution = np.asarray(result.mejor_solucion)
                    self.assertEqual(solution.shape, (10,))
                    self.assertTrue(np.all((solution >= -5) & (solution <= 5)))
                    self.assertTrue(np.isfinite(result.mejor_valor))
                    self.assertAlmostEqual(result.mejor_valor, sphere.func(solution))
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)

    def test_retired_names_rejected_without_random_state_mutation(self):
        from continuous_benchmark import orchestrator
        from dtw_stagnation import StagnationConfig
        py_state, np_state = random.getstate(), np.random.get_state()
        for name in ("WOA-ABC", "WOA_ABC", "WOAABC"):
            expected = f"MH poblacional no soportada: '{name}'. Opciones válidas: {list(SUPPORTED)}"
            with self.subTest(name=name):
                with self.assertRaises(ValueError) as error:
                    orchestrator.ejecutar_mh_standalone(None, name, max_iters=1)
                self.assertEqual(str(error.exception), expected)
                with self.assertRaises(ValueError) as error:
                    orchestrator._ejecutar_mh(name, None, None, StagnationConfig(), "random", 0, False)
                self.assertEqual(str(error.exception), expected)
                self.assertEqual(random.getstate(), py_state)
                after = np.random.get_state()
                self.assertEqual(after[0], np_state[0])
                np.testing.assert_array_equal(after[1], np_state[1])
                self.assertEqual(after[2:], np_state[2:])

    def test_obsolete_bridge_launchers_and_source_inventory_removed(self):
        from sensitivity.run import SOURCE_DIRS
        self.assertNotIn("woa_abc", SOURCE_DIRS)
        for relative in ("continuous_benchmark/mh/woa_abc.py", "run_woa_abc_cec.sh", "run_woa_abc_hres2.sh"):
            self.assertFalse((ROOT / relative).exists(), relative)
        # Individual solvers are explicitly preserved.
        for relative in ("continuous_benchmark/mh/woa.py", "continuous_benchmark/mh/abc.py",
                         "hybrid_mkp/mh/woa.py", "hybrid_mkp/mh/abc.py"):
            self.assertTrue((ROOT / relative).is_file(), relative)


if __name__ == "__main__":
    unittest.main()
