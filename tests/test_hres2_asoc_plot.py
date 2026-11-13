"""Saved-telemetry rendering only; no optimizer execution."""
import contextlib
import csv
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from matplotlib.figure import Figure

ENTRY = Path(__file__).resolve().parents[1] / "HRES2-H2/plot_asoc_ddtw_trace.py"


class TracePlotTests(unittest.TestCase):
    def test_saved_alignment_segmentation_labels_and_preservation(self):
        spec = importlib.util.spec_from_file_location("trace_plot", ENTRY)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as scratch:
            source = Path(scratch)
            csv_path = source / "historial_dtw.csv"
            csv_path.write_text("iteracion,epoch,mh,lcoe_best,dtw_delta\n"
                                "1,1,PSO,0.4,\n2,1,PSO,0.3,0.2\n"
                                "3,2,GWO,0.25,-0.1\n4,2,GWO,0.2,\n", encoding="utf-8")
            for name in ("fitness_convergence.png", "dtw_delta.png"):
                (source / name).write_bytes(b"original")
            before = {p.name: p.read_bytes() for p in source.iterdir()}
            plots = []
            savefig = Figure.savefig
            def capture(fig, *args, **kwargs):
                ax = fig.axes[0]
                plots.append((ax.get_title(), ax.get_ylabel(),
                    [(list(line.get_xdata()), list(line.get_ydata())) for line in ax.lines if line.get_linestyle() != "--"]))
                return savefig(fig, *args, **kwargs)
            stdout, stderr = io.StringIO(), io.StringIO()
            with patch.object(Figure, "savefig", capture), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                self.assertEqual(module.main(["--source", str(source)]), 0)
            self.assertEqual(stderr.getvalue(), "")
            self.assertEqual(stdout.getvalue(), "Rendered ddtw_convergence.png and ddtw_delta.png from 4 saved iterations.\n")
            self.assertIn("DDTW", plots[0][0])
            self.assertIn("DDTW", plots[1][0])
            self.assertEqual(plots[1][1], "DDTW Delta")
            self.assertEqual(plots[0][2], [([0, 1], [0.4, 0.3]), ([2, 3], [0.25, 0.2])])
            self.assertEqual(plots[1][2], [([1], [0.2]), ([1, 2], [0.2, -0.1])])
            for name, raw in before.items():
                self.assertEqual((source / name).read_bytes(), raw)
            for name in ("ddtw_convergence.png", "ddtw_delta.png"):
                self.assertTrue((source / name).read_bytes().startswith(b"\x89PNG"))
            saved = {p.name: p.read_bytes() for p in source.iterdir()}
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(module.main(["--source", str(source)]), 2)
            self.assertEqual(saved, {p.name: p.read_bytes() for p in source.iterdir()})

    def test_invalid_or_empty_telemetry_leaves_no_output(self):
        spec = importlib.util.spec_from_file_location("trace_plot_invalid", ENTRY)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for row in ("2,1,PSO,0.4,0.2", "1,1,ABC,0.4,0.2", "1,1,PSO,nan,0.2", "1,1,PSO,0.4,", "1,1,PSO,0.4,inf"):
            with self.subTest(row=row), tempfile.TemporaryDirectory() as scratch:
                source = Path(scratch)
                path = source / "historial_dtw.csv"
                path.write_text("iteracion,epoch,mh,lcoe_best,dtw_delta\n" + row + "\n")
                before = path.read_bytes()
                stdout, stderr = io.StringIO(), io.StringIO()
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    self.assertEqual(module.main(["--source", str(source)]), 2)
                self.assertEqual(stdout.getvalue(), "")
                self.assertEqual(stderr.getvalue(), "error: invalid or insufficient saved telemetry\n")
                self.assertEqual(list(source.iterdir()), [path])
                self.assertEqual(path.read_bytes(), before)
