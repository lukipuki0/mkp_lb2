"""Tests for the grouped output of one four-policy run and the convergence comparison.

Spec: odd/tasks/gwo-de-dtw-hybrid.md
"""
import contextlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from gwo_de_dtw import CAMPAIGN_POLICIES, POLICIES  # noqa: E402

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
POLICY_FILES = sorted([
    "result.json", "fitness_convergence.png", "dtw_delta.png", "parameters.png",
    "temporal_diagnostics.png", "temporal_diagnostics.pdf", "temporal_series.npz",
    "temporal_series.csv", "temporal_metadata.json",
])
COMPARISON_FILES = ["convergence_comparison.pdf", "convergence_comparison.png"]
HEADER = ("instance=mknapcb1.txt[0] binarization=lb2 signal=ddtw W=10 pop=8 "
          "iterations=25 seed=3")
DEFAULT_NAME = re.compile(r"^mknapcb1_0_s3_\d{8}-\d{6}(_\d+)?$")


def snapshot(folder):
    """Every file below `folder` with its bytes."""
    return {str(p.relative_to(folder)): p.read_bytes()
            for p in sorted(Path(folder).rglob("*")) if p.is_file()}


class GroupedCliTests(unittest.TestCase):
    ARGS = ("--instance", str(ROOT / "instancias" / "mknapcb1.txt"), "--pop", "8",
            "--iterations", "25", "--window", "10", "--seed", "3")

    def run_cli(self, *args, cwd):
        env = dict(os.environ, PYTHONPATH=str(ROOT))
        return subprocess.run([sys.executable, "-B", "-m", "gwo_de_dtw.run", *args],
                              cwd=cwd, env=env, capture_output=True, text=True, timeout=300)

    def assert_group(self, res, group, shown=None):
        """One folder per policy (record + plots) plus the comparison PNG/PDF.

        `shown` is the group path as printed (default: `group`).
        """
        shown = group if shown is None else shown
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertEqual(res.stderr, "")
        self.assertEqual(sorted(p.name for p in group.iterdir()),
                         sorted(list(CAMPAIGN_POLICIES) + COMPARISON_FILES))
        lines = res.stdout.splitlines()
        self.assertEqual(lines[0], HEADER)
        self.assertEqual(len(lines), 1 + len(CAMPAIGN_POLICIES) * 5 + 2)
        for i, policy in enumerate(CAMPAIGN_POLICIES):
            folder = group / policy
            block = lines[1 + 5 * i:6 + 5 * i]
            self.assertTrue(block[0].startswith(f"policy={policy} best="), block[0])
            self.assertTrue(all(line.startswith("  [plot] ") and f"'{shown / policy}" in line
                                for line in block[1:]), block)
            self.assertEqual(sorted(p.name for p in folder.iterdir()), POLICY_FILES)
            data = json.loads((folder / "result.json").read_text())
            self.assertEqual(data["policy"], policy)
            self.assertIs(data["dtw_kirito"], False)
            self.assertEqual(len(data["best_series"]), 25)
        self.assertEqual(lines[-2], "  [plot] convergence_comparison.png -> "
                                    f"'{shown / 'convergence_comparison.png'}'")
        self.assertEqual(lines[-1], f"output={shown}")
        self.assertEqual((group / "convergence_comparison.png").read_bytes()[:8], PNG_SIGNATURE)
        self.assertEqual((group / "convergence_comparison.pdf").read_bytes()[:5], b"%PDF-")

    def test_out_json_groups_the_four_policies_in_its_stem_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            res = self.run_cli(*self.ARGS, "--out", str(Path(tmp) / "x.json"), cwd=tmp)
            self.assert_group(res, Path(tmp) / "x")
            self.assertEqual([p.name for p in Path(tmp).iterdir()], ["x"])

    def test_out_directory_and_existing_empty_directory_are_used_as_is(self):
        with tempfile.TemporaryDirectory() as tmp:
            group = Path(tmp) / "group"
            group.mkdir()
            res = self.run_cli(*self.ARGS, "--out", str(group), cwd=tmp)
            self.assert_group(res, group)
            self.assertEqual([p.name for p in Path(tmp).iterdir()], ["group"])

    def test_without_out_writes_a_new_timestamp_group_in_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = Path(tmp) / "gwo_de_dtw" / "results"
            first = self.run_cli(*self.ARGS, cwd=tmp)
            groups = list(results.iterdir())
            self.assertEqual(len(groups), 1)
            self.assertRegex(groups[0].name, DEFAULT_NAME)
            # Paths are printed relative to the working directory, as given.
            shown = Path("gwo_de_dtw") / "results"
            self.assert_group(first, groups[0], shown / groups[0].name)
            before = snapshot(groups[0])

            second = self.run_cli(*self.ARGS, cwd=tmp)
            self.assertEqual(second.returncode, 0, second.stderr)
            new = [p for p in results.iterdir() if p != groups[0]]
            self.assertEqual(len(new), 1)
            self.assertRegex(new[0].name, DEFAULT_NAME)
            self.assert_group(second, new[0], shown / new[0].name)
            self.assertEqual(snapshot(groups[0]), before)

    def test_nonempty_destination_is_rejected_before_running(self):
        with tempfile.TemporaryDirectory() as tmp:
            group = Path(tmp) / "x"
            (group / "binary_simple").mkdir(parents=True)
            (group / "binary_simple" / "result.json").write_text('{"old": true}')
            (group / "notes.txt").write_text("keep")
            before = snapshot(tmp)
            for out in (group, Path(tmp) / "x.json"):
                with self.subTest(out=out.name):
                    res = self.run_cli(*self.ARGS, "--out", str(out), cwd=tmp)
                    self.assertEqual(res.returncode, 2)
                    self.assertEqual(res.stdout, "")
                    self.assertIn(f"error: output directory is not empty: {group}",
                                  res.stderr)
                    self.assertEqual(snapshot(tmp), before)
                    self.assertEqual(sorted(p.name for p in Path(tmp).iterdir()), ["x"])

    def test_destination_that_is_a_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "x"
            target.write_text("keep")
            res = self.run_cli(*self.ARGS, "--out", str(target), cwd=tmp)
            self.assertEqual(res.returncode, 2)
            self.assertEqual(res.stdout, "")
            self.assertIn(f"error: output path is not a directory: {target}", res.stderr)
            self.assertEqual(target.read_text(), "keep")
            self.assertEqual([p.name for p in Path(tmp).iterdir()], ["x"])

    def test_help_documents_the_group(self):
        from gwo_de_dtw.run import build_parser
        action = next(a for a in build_parser()._actions if a.dest == "out")
        self.assertIn("convergence_comparison", action.help)
        self.assertIn("gwo_de_dtw/results", action.help)


def record(policy, series, optimum=100.0, instance="demo[0]"):
    return {"policy": policy, "best_series": list(series), "known_optimum": optimum,
            "instance": instance}


class ComparisonPlotTests(unittest.TestCase):
    SERIES = {
        "binary_simple": [10.0, 20.0, 20.0, 30.0],
        "binary_complex": [12.0, 12.0, 25.0, 40.0],
        "continuous_simple": [5.0, 15.0, 35.0, 35.0],
        "continuous_complex": [8.0, 9.0, 10.0, 50.0],
    }

    def records(self, **kwargs):
        return [record(policy, self.SERIES[policy], **kwargs) for policy in POLICIES]

    def test_title_omits_txt_extension_without_changing_record(self):
        from gwo_de_dtw.plots import convergence_comparison_figure
        record = {"policy": "binary_simple", "best_series": [1, 2],
                  "instance": "mknapcb1.txt[0]"}
        fig = convergence_comparison_figure([record])
        try:
            self.assertEqual(fig.axes[0].get_title(),
                             "Convergence of one policy - mknapcb1[0]")
            self.assertEqual(record["instance"], "mknapcb1.txt[0]")
        finally:
            import matplotlib.pyplot as plt
            plt.close(fig)

    def test_public_exports(self):
        import gwo_de_dtw.plots as plots
        self.assertIn("plot_convergence_comparison", plots.__all__)
        self.assertIn("convergence_comparison_figure", plots.__all__)

    def test_four_raw_best_so_far_lines_on_generations(self):
        from gwo_de_dtw.plots import convergence_comparison_figure
        fig = convergence_comparison_figure(self.records())
        try:
            ax = fig.axes[0]
            lines = [line for line in ax.get_lines() if line.get_label() in POLICIES]
            self.assertEqual([line.get_label() for line in lines], list(POLICIES))
            for line, policy in zip(lines, POLICIES):
                self.assertEqual(list(line.get_xdata()), [1, 2, 3, 4])
                self.assertEqual(list(line.get_ydata()), self.SERIES[policy])
            colors = [line.get_color() for line in lines]
            self.assertEqual(len(set(colors)), 4)
            self.assertEqual(colors, ["#0072B2", "#E69F00", "#009E73", "#CC79A7"])
            refs = [line for line in ax.get_lines() if line not in lines]
            self.assertEqual(len(refs), 1)
            self.assertEqual(list(refs[0].get_ydata()), [100.0, 100.0])
            self.assertEqual(refs[0].get_label(), "Best known (100)")
            self.assertEqual(ax.get_xlabel(), "Generation")
            self.assertEqual(ax.get_ylabel(), "Best-so-far fitness (maximization)")
            self.assertEqual([t.get_text() for t in ax.get_legend().get_texts()],
                             list(POLICIES) + ["Best known (100)"])
            self.assertIn("demo[0]", ax.get_title())
        finally:
            plt.close(fig)

    def test_reference_only_when_all_share_a_positive_known_optimum(self):
        from gwo_de_dtw.plots import convergence_comparison_figure
        mixed = self.records()
        mixed[0]["known_optimum"] = 90.0
        for records in (mixed, self.records(optimum=0.0)):
            fig = convergence_comparison_figure(records)
            try:
                self.assertEqual([line.get_label() for line in fig.axes[0].get_lines()],
                                 list(POLICIES))
            finally:
                plt.close(fig)

    def test_default_labels_ignore_seed(self):
        from gwo_de_dtw.plots import convergence_comparison_figure
        records = self.records(instance="mknapcb1.txt[0]")
        for seed, res in enumerate(records, start=3):
            res["seed"] = seed
        fig = convergence_comparison_figure(records)
        try:
            ax = fig.axes[0]
            self.assertEqual(ax.get_title(), "Convergence of the four policies - mknapcb1[0]")
            self.assertEqual([t.get_text() for t in ax.get_legend().get_texts()],
                             list(POLICIES) + ["Best known (100)"])
        finally:
            plt.close(fig)

    def test_seed_labels_name_the_seed_of_each_best_run(self):
        from gwo_de_dtw.plots import comparison
        records = self.records(instance="mknapcb1.txt[0]")
        for res, seed in zip(records, (7, 31, 1, 16)):
            res["seed"] = seed
        labels = ["binary_simple (seed 7)", "binary_complex (seed 31)",
                  "continuous_simple (seed 1)", "continuous_complex (seed 16)"]
        fig = comparison.convergence_comparison_figure(records, seed_labels=True)
        try:
            ax = fig.axes[0]
            self.assertEqual(ax.get_title(), "Best run of each policy - mknapcb1[0]")
            self.assertEqual([t.get_text() for t in ax.get_legend().get_texts()],
                             labels + ["Best known (100)"])
            for line, policy in zip(ax.get_lines(), POLICIES):
                self.assertEqual(list(line.get_ydata()), self.SERIES[policy])
        finally:
            plt.close(fig)
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(comparison, "convergence_comparison_figure",
                                   wraps=comparison.convergence_comparison_figure) as spy, \
                    contextlib.redirect_stdout(io.StringIO()):
                comparison.plot_convergence_comparison(records, tmp, seed_labels=True)
            spy.assert_called_once_with(records, seed_labels=True)
            self.assertEqual(sorted(p.name for p in Path(tmp).iterdir()), COMPARISON_FILES)
        del records[1]["seed"]
        with self.assertRaisesRegex(ValueError, "record 1 has no seed"):
            comparison.convergence_comparison_figure(records, seed_labels=True)

    def test_record_without_policy_is_rejected(self):
        from gwo_de_dtw.plots import convergence_comparison_figure
        records = self.records()
        del records[2]["policy"]
        with self.assertRaisesRegex(ValueError, "policy"):
            convergence_comparison_figure(records)

    def test_saves_png_and_pdf_and_closes_the_figure(self):
        from gwo_de_dtw.plots import plot_convergence_comparison
        open_before = plt.get_fignums()
        with tempfile.TemporaryDirectory() as tmp:
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                paths = plot_convergence_comparison(self.records(), tmp)
            png, pdf = Path(tmp) / COMPARISON_FILES[1], Path(tmp) / COMPARISON_FILES[0]
            self.assertEqual(paths, [str(png), str(pdf)])
            self.assertEqual(sorted(p.name for p in Path(tmp).iterdir()), COMPARISON_FILES)
            self.assertGreater(png.stat().st_size, 8)
            self.assertEqual(png.read_bytes()[:8], PNG_SIGNATURE)
            self.assertGreater(pdf.stat().st_size, 5)
            self.assertEqual(pdf.read_bytes()[:5], b"%PDF-")
            self.assertEqual(out.getvalue(),
                             f"  [plot] convergence_comparison.png -> '{png}'\n")
        self.assertEqual(plt.get_fignums(), open_before)


if __name__ == "__main__":
    unittest.main()
