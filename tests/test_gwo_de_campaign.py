"""Tests for the multi-seed campaign runner (gwo_de_dtw.campaign).

Most tests patch campaign.run_hybrid with a deterministic runner that copies a
small real record and sets the fitness of each (policy, seed); one smoke test
runs the real hybrid with a tiny budget.

Spec: odd/tasks/gwo-de-dtw-hybrid.md
"""
import contextlib
import copy
import csv
import io
import json
import os
from pathlib import Path
import re
import statistics
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

from gwo_de_dtw import campaign  # noqa: E402
from gwo_de_dtw.policies import CAMPAIGN_POLICIES as POLICIES  # noqa: E402
from gwo_de_dtw.controller import run_hybrid as real_run_hybrid  # noqa: E402
from gwo_de_dtw.problem import cargar_instancia  # noqa: E402
from gwo_de_dtw.run import _to_json  # noqa: E402
from dtw_stagnation import StagnationConfig  # noqa: E402

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
BEST_RUN_FILES = sorted([
    "result.json", "fitness_convergence.png", "dtw_delta.png", "parameters.png",
    "temporal_diagnostics.png", "temporal_diagnostics.pdf", "temporal_series.npz",
    "temporal_series.csv", "temporal_metadata.json",
])
INSTANCE_FILES = sorted(list(POLICIES) + [
    "convergence_comparison.pdf", "convergence_comparison.png", "runs.csv", "statistics.csv"])
RUN_FIELDS = ["file", "index", "policy", "seed", "status", "best_fitness", "known_optimum",
              "gap_pct", "evaluations", "generations", "changes", "elapsed_s"]
STAT_FIELDS = ["file", "index", "policy", "runs", "best_fitness", "worst_fitness",
               "mean_fitness", "median_fitness", "std_fitness", "best_seed",
               "known_optimum", "best_gap_pct", "mean_gap_pct", "mean_elapsed_s"]
SMALL = ("--pop", "4", "--iterations", "12", "--window", "5")
RUN_LINE = re.compile(r"^\[(\d+)/(\d+)\] (\S+)\[(\d+)\] seed=(\d+) policy=(\S+) "
                      r"best=(\S+) gap=(\S+) elapsed=\d+\.\d{2}s$")


def snapshot(folder):
    """Every path below `folder`, with the bytes of files."""
    return {str(p.relative_to(folder)): (p.read_bytes() if p.is_file() else None)
            for p in sorted(Path(folder).rglob("*"))}


def write_instances(folder, name="mknapcb1.txt", count=2):
    """Tiny OR-Library file: `count` instances with 6 items and 2 constraints.

    Instance k has p[0] = 10 + k, so the fake runner can tell them apart.
    """
    lines = [str(count)]
    for k in range(count):
        lines += ["6 2 0", " ".join(str(10 * (j + 1) + k) for j in range(6)),
                  "3 5 2 4 6 1", "2 2 5 3 1 4", "9 8"]
    (Path(folder) / name).write_text("\n".join(lines) + "\n")


def write_references(folder, values=(300.0, 310.0)):
    """mkcbres.txt with the best known value of mknapcb1 (5.100) instances 0, 1, ..."""
    (Path(folder) / "mkcbres.txt").write_text(
        "".join(f"5.100-{i:02d} {v:g}\n" for i, v in enumerate(values)))


def read_csv(path):
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        return reader.fieldnames, list(reader)


class FakeRunner:
    """Stands in for run_hybrid: a real tiny record per policy, fitness from a table."""

    bases = {}

    def __init__(self, fitness, fail=None):
        self.fitness = fitness      # (index, policy, seed) -> float
        self.fail = fail            # (index, policy, seed) that raises
        self.calls = []
        self.records = {}

    def __call__(self, inst, cfg, **kw):
        index = int(inst["p"][0]) - 10
        key = (index, kw["policy"], kw["seed"])
        self.calls.append(key)
        if key == self.fail:
            raise RuntimeError("boom")
        if kw["policy"] not in self.bases:
            self.bases[kw["policy"]] = real_run_hybrid(inst, cfg, **dict(kw, seed=0))
        res = copy.deepcopy(self.bases[kw["policy"]])
        fitness = float(self.fitness(*key))
        optimum = inst["optimo"]
        res.update(seed=kw["seed"], best_fitness=fitness, known_optimum=optimum,
                   gap_pct=(100.0 * (optimum - fitness) / optimum) if optimum > 0 else None,
                   fake_tag=f"{index}-{kw['policy']}-{kw['seed']}")
        self.records[key] = copy.deepcopy(res)
        return res


class CampaignTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.inst_dir = self.tmp / "inst"
        self.inst_dir.mkdir()
        write_instances(self.inst_dir)
        self.out = self.tmp / "camp"

    def tearDown(self):
        self._tmp.cleanup()

    def run_main(self, runner, *args, wrap_plots=False, skip_run_plots=False):
        """campaign.main with the patched runner; returns (code, stdout, stderr, spies).

        skip_run_plots=True replaces save_all_plots (per best run) to save time.
        """
        out, err = io.StringIO(), io.StringIO()
        spies = {}
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(campaign, "run_hybrid", runner))
            if skip_run_plots:
                stack.enter_context(mock.patch.object(campaign, "save_all_plots",
                                                      return_value=[]))
            if wrap_plots:
                spies["save"] = stack.enter_context(mock.patch.object(
                    campaign, "save_all_plots", wraps=campaign.save_all_plots))
                spies["compare"] = stack.enter_context(mock.patch.object(
                    campaign, "plot_convergence_comparison",
                    wraps=campaign.plot_convergence_comparison))
            stack.enter_context(contextlib.redirect_stdout(out))
            stack.enter_context(contextlib.redirect_stderr(err))
            try:
                code = campaign.main(["--instances-dir", str(self.inst_dir), *args])
            except SystemExit as exc:
                code = exc.code
        return code, out.getvalue(), err.getvalue(), spies

    def expected_record(self, runner, key, instance):
        data = json.loads(json.dumps(runner.records[key], default=_to_json))
        data.update(instance=instance, signal="ddtw")
        return data


class DefaultsAndPreflightTests(CampaignTestCase):
    def test_defaults_come_from_config(self):
        args = campaign.build_parser().parse_args([])
        self.assertEqual(args.files, [f"mknapcb{i}.txt" for i in range(1, 10)])
        self.assertEqual(args.indices, [0, 1])
        self.assertEqual(args.seeds, list(range(1, 32)))
        self.assertEqual((args.pop, args.iterations, args.window, args.signal,
                          args.binarization, args.dtw_kirito, args.out, args.dry_run),
                         (30, 1000, 40, "ddtw", "lb2", False, None, False))
        self.assertEqual(args.instances_dir, Path("instancias"))

    def test_dry_run_counts_3348_runs_without_running_or_writing(self):
        env = dict(os.environ, PYTHONPATH=str(ROOT))
        res = subprocess.run(
            [sys.executable, "-B", "-m", "gwo_de_dtw.campaign", "--dry-run",
             "--instances-dir", str(ROOT / "instancias"), "--out", str(self.out)],
            cwd=self.tmp, env=env, capture_output=True, text=True, timeout=120)
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertEqual(res.stderr, "")
        self.assertEqual(res.stdout.splitlines(), [
            "campaign files=9 indices=2 seeds=31 policies=6 runs=3348 binarization=lb2 "
            "signal=ddtw W=40 pop=30 iterations=1000",
            "dry-run: selection and output validated, nothing executed or written",
        ])
        self.assertEqual(sorted(p.name for p in self.tmp.iterdir()), ["inst"])

    def test_rejected_inputs_exit_2_before_any_run_or_write(self):
        nonempty = self.tmp / "old"
        nonempty.mkdir()
        (nonempty / "summary.csv").write_text("keep")
        a_file = self.tmp / "afile"
        a_file.write_text("keep")
        cases = [
            (("--files", "../mknapcb1.txt"),
             "invalid instance file name (basename only): ../mknapcb1.txt"),
            (("--files", "sub/mknapcb1.txt"),
             "invalid instance file name (basename only): sub/mknapcb1.txt"),
            (("--files", ".."), "invalid instance file name (basename only): .."),
            (("--files", "missing.txt"),
             f"instance file not found: {self.inst_dir / 'missing.txt'}"),
            (("--files", "mknapcb1.txt", "mknapcb1.txt"),
             "duplicate instance file: mknapcb1.txt"),
            (("--files",), "argument --files: expected at least one argument"),
            (("--files", "mknapcb1.txt", "--indices", "2"),
             "index 2 out of range for mknapcb1.txt (2 instances)"),
            (("--files", "mknapcb1.txt", "--indices", "-1"),
             "index -1 out of range for mknapcb1.txt (2 instances)"),
            (("--files", "mknapcb1.txt", "--indices", "0", "0"), "duplicate index: 0"),
            (("--files", "mknapcb1.txt", "--indices"),
             "argument --indices: expected at least one argument"),
            (("--files", "mknapcb1.txt", "--indices", "0", "--seeds", "-1"),
             "seeds must be nonnegative: -1"),
            (("--files", "mknapcb1.txt", "--indices", "0", "--seeds", "3", "3"),
             "duplicate seed: 3"),
            (("--files", "mknapcb1.txt", "--indices", "0", "--seeds"),
             "argument --seeds: expected at least one argument"),
            (("--files", "mknapcb1.txt", "--indices", "0", "--pop", "3"),
             "--pop must be >= 4 (DE/rand/1 needs three other individuals)"),
            (("--files", "mknapcb1.txt", "--indices", "0", "--iterations", "0"),
             "--iterations must be positive"),
            (("--files", "mknapcb1.txt", "--indices", "0", "--window", "0"),
             "--window must be positive"),
            (("--files", "mknapcb1.txt", "--indices", "0", "--g2", "100"),
             "--g2 must be below v_max=100 (otherwise the LB2 slope changes sign)"),
            (("--files", "mknapcb1.txt", "--indices", "0", "--out", str(nonempty)),
             f"output directory is not empty: {nonempty}"),
            (("--files", "mknapcb1.txt", "--indices", "0", "--out", str(a_file)),
             f"output path is not a directory: {a_file}"),
        ]
        (self.inst_dir / "notes").mkdir()
        before = snapshot(self.tmp)
        for args, message in cases:
            with self.subTest(args=args):
                runner = FakeRunner(lambda *k: 1.0)
                code, out, err, _ = self.run_main(runner, *args)
                self.assertEqual(code, 2)
                self.assertEqual(out, "")
                self.assertIn(f"error: {message}\n", err)
                self.assertEqual(runner.calls, [])
                self.assertEqual(snapshot(self.tmp), before)

    def test_help_documents_layout_and_dry_run(self):
        text = campaign.build_parser().format_help()
        for word in ("--dry-run", "campaign_config.json", "resumen.txt", "summary.csv",
                     "--instances-dir", "best_run"):
            self.assertIn(word, text)


class CampaignOutputTests(CampaignTestCase):
    def test_31_seed_ties_pick_the_lowest_seed_and_unknown_reference_is_blank(self):
        tied = {7, 12, 30}
        runner = FakeRunner(lambda i, p, s: 50.0 if s in tied else float(s % 5))
        seeds = [str(s) for s in range(31, 0, -1)]     # reverse input order
        code, out, err, _ = self.run_main(runner, "--files", "mknapcb1.txt", "--indices", "0",
                                          "--seeds", *seeds, *SMALL, "--out", str(self.out))
        self.assertEqual(code, 0, err)
        self.assertEqual(err, "")
        self.assertEqual(len(runner.calls), 186)
        self.assertEqual(sorted(p.name for p in self.out.iterdir()),
                         ["campaign_config.json", "mknapcb1", "resumen.txt", "statistics.csv",
                          "summary.csv"])
        inst = self.out / "mknapcb1" / "inst_00"
        self.assertEqual([p.name for p in (self.out / "mknapcb1").iterdir()], ["inst_00"])
        self.assertEqual(sorted(p.name for p in inst.iterdir()), INSTANCE_FILES)
        for policy in POLICIES:
            self.assertEqual([p.name for p in (inst / policy).iterdir()], ["best_run"])
            self.assertEqual(sorted(p.name for p in (inst / policy / "best_run").iterdir()),
                             BEST_RUN_FILES)
            data = json.loads((inst / policy / "best_run" / "result.json").read_text())
            self.assertEqual(data, self.expected_record(runner, (0, policy, 7),
                                                        "mknapcb1.txt[0]"))
        self.assertEqual((inst / "convergence_comparison.png").read_bytes()[:8], PNG_SIGNATURE)
        self.assertEqual((inst / "convergence_comparison.pdf").read_bytes()[:5], b"%PDF-")

        fields, rows = read_csv(inst / "runs.csv")
        self.assertEqual(fields, RUN_FIELDS)
        self.assertEqual(len(rows), 186)
        self.assertEqual([(r["seed"], r["policy"]) for r in rows[:len(POLICIES) + 1]],
                         [("31", p) for p in POLICIES] + [("30", "binary_simple")])
        for row in rows:
            self.assertEqual((row["file"], row["index"], row["status"]),
                             ("mknapcb1.txt", "0", "ok"))
            self.assertEqual(row["known_optimum"], "")
            self.assertEqual(row["gap_pct"], "")
            self.assertGreaterEqual(float(row["elapsed_s"]), 0.0)
        self.assertEqual(read_csv(self.out / "summary.csv"), (fields, rows))

        fields, stats = read_csv(inst / "statistics.csv")
        self.assertEqual(fields, STAT_FIELDS)
        self.assertEqual([r["policy"] for r in stats], list(POLICIES))
        for row in stats:
            self.assertEqual((row["runs"], row["best_fitness"], row["best_seed"]),
                             ("31", "50.0", "7"))
            self.assertEqual((row["known_optimum"], row["best_gap_pct"], row["mean_gap_pct"]),
                             ("", "", ""))
        self.assertEqual(read_csv(self.out / "statistics.csv"), (fields, stats))

        config = json.loads((self.out / "campaign_config.json").read_text())
        self.assertEqual((config["status"], config["completed_runs"], config["total_runs"]),
                         ("completed", 186, 186))
        self.assertIsNone(config["error"])
        self.assertEqual(config["selection"]["seeds"], list(range(31, 0, -1)))
        self.assertEqual(config["selection"]["policies"], list(POLICIES))
        self.assertEqual(config["config"]["pop"], 4)
        self.assertIs(config["config"]["dtw_kirito"], False)

        lines = out.splitlines()
        self.assertEqual(lines[0], "campaign files=1 indices=1 seeds=31 policies=6 runs=186 "
                                   "binarization=lb2 signal=ddtw W=5 pop=4 iterations=12")
        run_lines = [RUN_LINE.match(line) for line in lines if line.startswith("[")]
        self.assertEqual(len(run_lines), 186)
        self.assertTrue(all(run_lines))
        self.assertEqual(run_lines[0].groups()[:8],
                         ("1", "186", "mknapcb1.txt", "0", "31", "binary_simple", "1", "n/a"))
        self.assertEqual(lines[-2], "status=completed runs=186/186")
        self.assertEqual(lines[-1], f"output={self.out}")

        resumen = (self.out / "resumen.txt").read_text(encoding="utf-8")
        self.assertIn("Brecha de la semilla ganadora: no disponible", resumen)
        self.assertNotIn("Brecha de la semilla ganadora: 0", resumen)
        self.assertNotIn("brecha 0", resumen)
        tied = (7, 12, 30)
        ranked = list(tied) + [seed for fitness in (4, 3, 2, 1, 0)
                               for seed in range(1, 32)
                               if seed not in tied and seed % 5 == fitness]
        self.assertEqual(len(ranked), 31)
        start = resumen.index(
            "  Ranking de semillas (mejor a peor; por fitness, no por la media):")
        rank_lines = resumen[start:].splitlines()[1:32]
        self.assertEqual([line.split()[2] for line in rank_lines], [str(s) for s in ranked])
        self.assertTrue(all(line.endswith("brecha no disponible") for line in rank_lines))
        mean = f"{statistics.mean(50.0 if s in tied else float(s % 5) for s in range(1, 32)):.4f}"
        self.assertIn("\n".join(
            f"  {place}. {policy}  mejor 50.0000  semilla 7  brecha no disponible  media {mean}"
            for place, policy in enumerate(POLICIES, 1)), resumen)

    def test_winner_seeds_differ_statistics_and_only_winners_are_exported(self):
        write_references(self.inst_dir)
        winners = {"binary_simple": 4, "binary_complex": 1,
                   "continuous_simple": 5, "continuous_complex": 2,
                   "vanilla_exploitation": 3, "vanilla_exploration": 3}
        policy_rank = {p: k for k, p in enumerate(POLICIES)}

        def fitness(index, policy, seed):
            if seed == winners[policy]:
                return 280.0 + index + policy_rank[policy]
            return 100.0 + 10 * seed + policy_rank[policy] + index

        runner = FakeRunner(fitness)
        seeds = ("3", "5", "1", "4", "2")
        code, out, err, spies = self.run_main(
            runner, "--files", "mknapcb1.txt", "--indices", "1", "0", "--seeds", *seeds,
            *SMALL, "--out", str(self.out), wrap_plots=True)
        self.assertEqual(code, 0, err)
        self.assertEqual(len(runner.calls), 60)

        saved = [(c.args[0]["fake_tag"], Path(c.args[1])) for c in spies["save"].call_args_list]
        expected_saved = []
        for index in (1, 0):
            for policy in POLICIES:
                expected_saved.append(
                    (f"{index}-{policy}-{winners[policy]}",
                     self.out / "mknapcb1" / f"inst_{index:02d}" / policy / "best_run"))
        self.assertEqual(saved, expected_saved)
        compare = spies["compare"].call_args_list
        self.assertEqual(len(compare), 2)
        for call, index in zip(compare, (1, 0)):
            self.assertEqual([r["fake_tag"] for r in call.args[0]],
                             [f"{index}-{p}-{winners[p]}" for p in POLICIES])
            self.assertEqual(Path(call.args[1]), self.out / "mknapcb1" / f"inst_{index:02d}")
            self.assertEqual(call.kwargs, {"seed_labels": True})

        _, summary = read_csv(self.out / "summary.csv")
        self.assertEqual([(r["index"], r["seed"]) for r in summary[::len(POLICIES)]],
                         [(i, s) for i in ("1", "0") for s in seeds])
        _, root_stats = read_csv(self.out / "statistics.csv")
        self.assertEqual([(r["index"], r["policy"]) for r in root_stats],
                         [(i, p) for i in ("1", "0") for p in POLICIES])
        for index, optimum in ((0, 300.0), (1, 310.0)):
            inst = self.out / "mknapcb1" / f"inst_{index:02d}"
            _, rows = read_csv(inst / "runs.csv")
            self.assertEqual(rows, [r for r in summary if r["index"] == str(index)])
            for row in rows:
                f = fitness(index, row["policy"], int(row["seed"]))
                self.assertEqual(float(row["best_fitness"]), f)
                self.assertEqual(float(row["known_optimum"]), optimum)
                self.assertEqual(float(row["gap_pct"]), 100.0 * (optimum - f) / optimum)
            _, stats = read_csv(inst / "statistics.csv")
            self.assertEqual(stats, [r for r in root_stats if r["index"] == str(index)])
            for row, policy in zip(stats, POLICIES):
                values = [fitness(index, policy, int(s)) for s in seeds]
                gaps = [100.0 * (optimum - v) / optimum for v in values]
                self.assertEqual(row["runs"], "5")
                self.assertEqual(row["best_seed"], str(winners[policy]))
                self.assertEqual(float(row["best_fitness"]), max(values))
                self.assertEqual(float(row["worst_fitness"]), min(values))
                self.assertEqual(float(row["mean_fitness"]), statistics.mean(values))
                self.assertEqual(float(row["median_fitness"]), statistics.median(values))
                self.assertEqual(float(row["std_fitness"]), statistics.stdev(values))
                self.assertEqual(float(row["known_optimum"]), optimum)
                self.assertEqual(float(row["best_gap_pct"]), min(gaps))
                self.assertEqual(float(row["mean_gap_pct"]), statistics.mean(gaps))
                self.assertGreaterEqual(float(row["mean_elapsed_s"]), 0.0)
                best = inst / policy / "best_run" / "result.json"
                self.assertEqual(json.loads(best.read_text()),
                                 self.expected_record(runner, (index, policy, winners[policy]),
                                                      f"mknapcb1.txt[{index}]"))

    def test_single_seed_std_is_blank(self):
        runner = FakeRunner(lambda i, p, s: 42.0)
        code, _, err, _ = self.run_main(runner, "--files", "mknapcb1.txt", "--indices", "0",
                                        "--seeds", "9", *SMALL, "--out", str(self.out),
                                        skip_run_plots=True)
        self.assertEqual(code, 0, err)
        _, stats = read_csv(self.out / "statistics.csv")
        self.assertEqual([(r["runs"], r["std_fitness"], r["best_seed"]) for r in stats],
                         [("1", "", "9")] * len(POLICIES))
        resumen = (self.out / "resumen.txt").read_text(encoding="utf-8")
        self.assertIn("Desviaci\u00f3n est\u00e1ndar muestral: no aplica", resumen)
        self.assertNotIn("Desviaci\u00f3n est\u00e1ndar muestral: 0", resumen)
        self.assertIn("Brecha de la semilla ganadora: no disponible", resumen)
        self.assertNotIn("Brecha de la semilla ganadora: 0", resumen)
        self.assertIn("\n".join(
            f"  {place}. {policy}  mejor 42.0000  semilla 9  brecha no disponible  media 42.0000"
            for place, policy in enumerate(POLICIES, 1)), resumen)

    def test_failure_stops_with_failed_status_and_keeps_finished_results(self):
        runner = FakeRunner(lambda i, p, s: float(s), fail=(1, "continuous_simple", 2))
        code, out, err, _ = self.run_main(runner, "--files", "mknapcb1.txt", "--indices", "0",
                                          "1", "--seeds", "1", "2", *SMALL,
                                          "--out", str(self.out), skip_run_plots=True)
        self.assertEqual(code, 1)
        self.assertEqual(err, "error: campaign failed at mknapcb1.txt[1] "
                              "policy=continuous_simple seed=2: RuntimeError: boom\n")
        self.assertEqual(len(runner.calls), 21)
        self.assertEqual(out.splitlines()[-2:], ["status=failed runs=20/24",
                                                 f"output={self.out}"])
        config = json.loads((self.out / "campaign_config.json").read_text())
        self.assertEqual((config["status"], config["completed_runs"], config["total_runs"]),
                         ("failed", 20, 24))
        self.assertEqual(config["error"], "mknapcb1.txt[1] policy=continuous_simple seed=2: "
                                          "RuntimeError: boom")
        _, summary = read_csv(self.out / "summary.csv")
        self.assertEqual(len(summary), 21)
        self.assertEqual([r["status"] for r in summary], ["ok"] * 20 + ["failed"])
        failed = summary[-1]
        self.assertEqual((failed["index"], failed["policy"], failed["seed"]),
                         ("1", "continuous_simple", "2"))
        self.assertEqual([failed[k] for k in RUN_FIELDS[5:11]], [""] * 6)
        done = self.out / "mknapcb1" / "inst_00"
        self.assertEqual(sorted(p.name for p in done.iterdir()), INSTANCE_FILES)
        partial = self.out / "mknapcb1" / "inst_01"
        self.assertEqual([p.name for p in partial.iterdir()], ["runs.csv"])
        _, rows = read_csv(partial / "runs.csv")
        self.assertEqual(rows, summary[12:])
        _, stats = read_csv(self.out / "statistics.csv")
        self.assertEqual([r["index"] for r in stats], ["0"] * len(POLICIES))
        resumen = (self.out / "resumen.txt").read_text(encoding="utf-8")
        self.assertIn("Estado: fallida", resumen)
        self.assertIn("Progreso: 1/2 instancias completadas, 20/24 corridas", resumen)
        self.assertIn("Instancia mknapcb1.txt [0]", resumen)
        self.assertNotIn("Instancia mknapcb1.txt [1]", resumen)
        self.assertIn("Error: mknapcb1.txt[1] policy=continuous_simple seed=2: "
                      "RuntimeError: boom", resumen)
        self.assertIn("Incluye solo las instancias completadas antes de detener la campa\u00f1a.",
                      resumen)
        self.assertFalse(any(".tmp" in p.name for p in self.out.rglob("*")))

    def test_default_output_is_a_new_timestamp_folder_under_results(self):
        runner = FakeRunner(lambda i, p, s: 1.0)
        cwd = os.getcwd()
        os.chdir(self.tmp)
        try:
            first = self.run_main(runner, "--files", "mknapcb1.txt", "--indices", "0",
                                  "--seeds", "1", *SMALL, skip_run_plots=True)
            second = self.run_main(runner, "--files", "mknapcb1.txt", "--indices", "0",
                                   "--seeds", "1", *SMALL, skip_run_plots=True)
        finally:
            os.chdir(cwd)
        results = self.tmp / "gwo_de_dtw" / "results"
        names = sorted(p.name for p in results.iterdir())
        self.assertEqual(len(names), 2)
        for (code, out, err, _), name in zip((first, second), names):
            self.assertEqual(code, 0, err)
            self.assertRegex(name, r"^campaign_\d{8}-\d{6}(_\d+)?$")
        shown = {first[1].splitlines()[-1], second[1].splitlines()[-1]}
        self.assertEqual(shown, {f"output={Path('gwo_de_dtw') / 'results' / n}" for n in names})

    def _run_resumen_fixture(self, runner):
        write_references(self.inst_dir, values=(300.0,))
        ticks = {"n": 0.0}

        def clock():
            ticks["n"] += 0.25
            return ticks["n"]

        with mock.patch.object(campaign.time, "perf_counter", clock), \
                mock.patch.object(campaign, "plot_convergence_comparison", return_value=[]):
            return self.run_main(
                runner, "--files", "mknapcb1.txt", "--indices", "1", "0",
                "--seeds", "3", "1", "2", *SMALL, "--dtw-kirito", "--out", str(self.out),
                skip_run_plots=True)

    def test_resumen_txt_exact_sections_order_ties_gap_progressive_and_csv(self):
        runner = ResumenSnapshotRunner(self.out)
        code, _, err, _ = self._run_resumen_fixture(runner)
        self.assertEqual(code, 0, err)
        self.assertEqual(err, "")
        self.assertFalse(runner.progressive_tmp)
        progressive = expected_resumen(
            "en curso", "1/2 instancias completadas, 18/36 corridas",
            "Incluye solo las instancias completadas hasta ahora.", (1,))
        final = expected_resumen(
            "completada", "2/2 instancias completadas, 36/36 corridas",
            "Incluye todas las instancias completadas.", (1, 0))
        self.assertEqual(runner.progressive, progressive)
        self.assertEqual((self.out / "resumen.txt").read_text(encoding="utf-8"), final)
        self.assertFalse(any(".tmp" in p.name for p in self.out.rglob("*")))
        self.assertNotIn("Instancia mknapcb1.txt [0]", progressive)
        self.assertIn("no disponible", progressive)
        self.assertNotIn("brecha 0", final)

        fields, rows = read_csv(self.out / "summary.csv")
        self.assertEqual(fields, RUN_FIELDS)
        expected_rows = [(str(index), str(seed), policy)
                         for index in (1, 0) for seed in (3, 1, 2) for policy in POLICIES]
        self.assertEqual([(r["index"], r["seed"], r["policy"]) for r in rows], expected_rows)
        for row in rows:
            fitness = FITNESS_TABLE[(int(row["index"]), row["policy"])][int(row["seed"])]
            self.assertEqual(float(row["best_fitness"]), fitness)
            self.assertEqual(row["status"], "ok")
            if row["index"] == "1":
                self.assertEqual((row["known_optimum"], row["gap_pct"]), ("", ""))
            else:
                self.assertEqual(float(row["known_optimum"]), 300.0)
                self.assertEqual(float(row["gap_pct"]), 100.0 * (300.0 - fitness) / 300.0)
        fields, stats = read_csv(self.out / "statistics.csv")
        self.assertEqual(fields, STAT_FIELDS)
        self.assertEqual([(r["index"], r["policy"]) for r in stats],
                         [(str(index), policy) for index in (1, 0) for policy in POLICIES])
        winners = {1: {"binary_simple": "1", "binary_complex": "2",
                       "continuous_simple": "1", "continuous_complex": "1",
                       "vanilla_exploitation": "1", "vanilla_exploration": "1"},
                   0: {"binary_simple": "2", "binary_complex": "1",
                       "continuous_simple": "2", "continuous_complex": "2",
                       "vanilla_exploitation": "3", "vanilla_exploration": "3"}}
        for row in stats:
            index, policy = int(row["index"]), row["policy"]
            values = [FITNESS_TABLE[(index, policy)][seed] for seed in (3, 1, 2)]
            self.assertEqual(row["best_seed"], winners[index][policy])
            self.assertEqual(float(row["best_fitness"]), max(values))
            self.assertEqual(float(row["worst_fitness"]), min(values))
            self.assertEqual(float(row["mean_fitness"]), statistics.mean(values))
            self.assertEqual(float(row["std_fitness"]), statistics.stdev(values))
            if index == 1:
                self.assertEqual((row["known_optimum"], row["best_gap_pct"], row["mean_gap_pct"]),
                                 ("", "", ""))
            else:
                gaps = [100.0 * (300.0 - value) / 300.0 for value in values]
                self.assertEqual(float(row["best_gap_pct"]), min(gaps))
                self.assertEqual(float(row["mean_gap_pct"]), statistics.mean(gaps))

    def test_resumen_interrupted_rewrite_keeps_prior_text_and_no_temp(self):
        runner = ResumenSnapshotRunner(self.out)
        real_write_text = Path.write_text

        def interrupted_write(self, data, *args, **kwargs):
            if self.name == "resumen.txt.tmp":
                interrupted_write.n += 1
                if interrupted_write.n >= 2:
                    real_write_text(self, "PARTIAL", encoding="utf-8")
                    raise OSError("interrupted")
            return real_write_text(self, data, *args, **kwargs)

        interrupted_write.n = 0
        with mock.patch.object(Path, "write_text", interrupted_write):
            code, _, err, _ = self._run_resumen_fixture(runner)
        self.assertEqual(code, 1)
        self.assertIn("interrupted", err)
        progressive = expected_resumen(
            "en curso", "1/2 instancias completadas, 18/36 corridas",
            "Incluye solo las instancias completadas hasta ahora.", (1,))
        text = (self.out / "resumen.txt").read_text(encoding="utf-8")
        self.assertEqual(runner.progressive, progressive)
        self.assertEqual(text, progressive)
        self.assertNotIn("PARTIAL", text)
        self.assertFalse(any(".tmp" in p.name for p in self.out.rglob("*")))
        config = json.loads((self.out / "campaign_config.json").read_text(encoding="utf-8"))
        self.assertEqual(config["status"], "failed")
        _, rows = read_csv(self.out / "summary.csv")
        self.assertEqual(len(rows), 36)
        self.assertTrue(all(row["status"] == "ok" for row in rows))


RESUMEN_RULES = (
    "El ranking ordena las mejores corridas: mayor fitness primero y, en empate, "
    "la semilla menor.\n"
    "Si el fitness y la semilla tambi\u00e9n empatan, se conserva el orden de los modos "
    "en la campa\u00f1a.\n"
    "La media se muestra, pero no determina el ranking.\n"
    "La desviaci\u00f3n est\u00e1ndar es muestral (n - 1). Con una sola corrida: no aplica.\n"
    "Si la referencia es desconocida, la brecha es no disponible (no cero)."
)

INSTANCE_TEXT = {
    1: """Instancia mknapcb1.txt [1]
--------------------------
Modo binary_simple
  Mejor fitness: 50.0000
  Peor fitness: 40.0000
  Media: 43.3333
  Desviaci\u00f3n est\u00e1ndar muestral: 5.7735
  Semilla ganadora: 1
  Brecha de la semilla ganadora: no disponible
  Tiempo medio: 0.2500 s
  Ranking de semillas (mejor a peor; por fitness, no por la media):
    1. semilla 1  fitness 50.0000  brecha no disponible
    2. semilla 2  fitness 40.0000  brecha no disponible
    3. semilla 3  fitness 40.0000  brecha no disponible

Modo binary_complex
  Mejor fitness: 70.0000
  Peor fitness: 30.0000
  Media: 56.6667
  Desviaci\u00f3n est\u00e1ndar muestral: 23.0940
  Semilla ganadora: 2
  Brecha de la semilla ganadora: no disponible
  Tiempo medio: 0.2500 s
  Ranking de semillas (mejor a peor; por fitness, no por la media):
    1. semilla 2  fitness 70.0000  brecha no disponible
    2. semilla 3  fitness 70.0000  brecha no disponible
    3. semilla 1  fitness 30.0000  brecha no disponible

Modo continuous_simple
  Mejor fitness: 70.0000
  Peor fitness: 10.0000
  Media: 33.3333
  Desviaci\u00f3n est\u00e1ndar muestral: 32.1455
  Semilla ganadora: 1
  Brecha de la semilla ganadora: no disponible
  Tiempo medio: 0.2500 s
  Ranking de semillas (mejor a peor; por fitness, no por la media):
    1. semilla 1  fitness 70.0000  brecha no disponible
    2. semilla 3  fitness 20.0000  brecha no disponible
    3. semilla 2  fitness 10.0000  brecha no disponible

Modo continuous_complex
  Mejor fitness: 70.0000
  Peor fitness: 5.0000
  Media: 33.3333
  Desviaci\u00f3n est\u00e1ndar muestral: 33.2916
  Semilla ganadora: 1
  Brecha de la semilla ganadora: no disponible
  Tiempo medio: 0.2500 s
  Ranking de semillas (mejor a peor; por fitness, no por la media):
    1. semilla 1  fitness 70.0000  brecha no disponible
    2. semilla 2  fitness 25.0000  brecha no disponible
    3. semilla 3  fitness 5.0000  brecha no disponible

Modo vanilla_exploitation
  Mejor fitness: 15.0000
  Peor fitness: 10.0000
  Media: 12.3333
  Desviaci\u00f3n est\u00e1ndar muestral: 2.5166
  Semilla ganadora: 1
  Brecha de la semilla ganadora: no disponible
  Tiempo medio: 0.2500 s
  Ranking de semillas (mejor a peor; por fitness, no por la media):
    1. semilla 1  fitness 15.0000  brecha no disponible
    2. semilla 2  fitness 12.0000  brecha no disponible
    3. semilla 3  fitness 10.0000  brecha no disponible

Modo vanilla_exploration
  Mejor fitness: 9.0000
  Peor fitness: 7.0000
  Media: 8.0000
  Desviaci\u00f3n est\u00e1ndar muestral: 1.0000
  Semilla ganadora: 1
  Brecha de la semilla ganadora: no disponible
  Tiempo medio: 0.2500 s
  Ranking de semillas (mejor a peor; por fitness, no por la media):
    1. semilla 1  fitness 9.0000  brecha no disponible
    2. semilla 3  fitness 8.0000  brecha no disponible
    3. semilla 2  fitness 7.0000  brecha no disponible""",
    0: """Instancia mknapcb1.txt [0]
--------------------------
Modo binary_simple
  Mejor fitness: 200.0000
  Peor fitness: 180.0000
  Media: 193.3333
  Desviaci\u00f3n est\u00e1ndar muestral: 11.5470
  Semilla ganadora: 2
  Brecha de la semilla ganadora: 33.33%
  Tiempo medio: 0.2500 s
  Ranking de semillas (mejor a peor; por fitness, no por la media):
    1. semilla 2  fitness 200.0000  brecha 33.33%
    2. semilla 3  fitness 200.0000  brecha 33.33%
    3. semilla 1  fitness 180.0000  brecha 40.00%

Modo binary_complex
  Mejor fitness: 200.0000
  Peor fitness: 150.0000
  Media: 170.0000
  Desviaci\u00f3n est\u00e1ndar muestral: 26.4575
  Semilla ganadora: 1
  Brecha de la semilla ganadora: 33.33%
  Tiempo medio: 0.2500 s
  Ranking de semillas (mejor a peor; por fitness, no por la media):
    1. semilla 1  fitness 200.0000  brecha 33.33%
    2. semilla 3  fitness 160.0000  brecha 46.67%
    3. semilla 2  fitness 150.0000  brecha 50.00%

Modo continuous_simple
  Mejor fitness: 140.0000
  Peor fitness: 100.0000
  Media: 120.0000
  Desviaci\u00f3n est\u00e1ndar muestral: 20.0000
  Semilla ganadora: 2
  Brecha de la semilla ganadora: 53.33%
  Tiempo medio: 0.2500 s
  Ranking de semillas (mejor a peor; por fitness, no por la media):
    1. semilla 2  fitness 140.0000  brecha 53.33%
    2. semilla 3  fitness 120.0000  brecha 60.00%
    3. semilla 1  fitness 100.0000  brecha 66.67%

Modo continuous_complex
  Mejor fitness: 135.0000
  Peor fitness: 130.0000
  Media: 131.6667
  Desviaci\u00f3n est\u00e1ndar muestral: 2.8868
  Semilla ganadora: 2
  Brecha de la semilla ganadora: 55.00%
  Tiempo medio: 0.2500 s
  Ranking de semillas (mejor a peor; por fitness, no por la media):
    1. semilla 2  fitness 135.0000  brecha 55.00%
    2. semilla 1  fitness 130.0000  brecha 56.67%
    3. semilla 3  fitness 130.0000  brecha 56.67%

Modo vanilla_exploitation
  Mejor fitness: 90.0000
  Peor fitness: 80.0000
  Media: 85.0000
  Desviaci\u00f3n est\u00e1ndar muestral: 5.0000
  Semilla ganadora: 3
  Brecha de la semilla ganadora: 70.00%
  Tiempo medio: 0.2500 s
  Ranking de semillas (mejor a peor; por fitness, no por la media):
    1. semilla 3  fitness 90.0000  brecha 70.00%
    2. semilla 2  fitness 85.0000  brecha 71.67%
    3. semilla 1  fitness 80.0000  brecha 73.33%

Modo vanilla_exploration
  Mejor fitness: 70.0000
  Peor fitness: 60.0000
  Media: 65.0000
  Desviaci\u00f3n est\u00e1ndar muestral: 5.0000
  Semilla ganadora: 3
  Brecha de la semilla ganadora: 76.67%
  Tiempo medio: 0.2500 s
  Ranking de semillas (mejor a peor; por fitness, no por la media):
    1. semilla 3  fitness 70.0000  brecha 76.67%
    2. semilla 2  fitness 65.0000  brecha 78.33%
    3. semilla 1  fitness 60.0000  brecha 80.00%""",
}

RANK_TEXT = {
    1: """Instancia mknapcb1.txt [1]
  1. continuous_simple  mejor 70.0000  semilla 1  brecha no disponible  media 33.3333
  2. continuous_complex  mejor 70.0000  semilla 1  brecha no disponible  media 33.3333
  3. binary_complex  mejor 70.0000  semilla 2  brecha no disponible  media 56.6667
  4. binary_simple  mejor 50.0000  semilla 1  brecha no disponible  media 43.3333
  5. vanilla_exploitation  mejor 15.0000  semilla 1  brecha no disponible  media 12.3333
  6. vanilla_exploration  mejor 9.0000  semilla 1  brecha no disponible  media 8.0000""",
    0: """Instancia mknapcb1.txt [0]
  1. binary_complex  mejor 200.0000  semilla 1  brecha 33.33%  media 170.0000
  2. binary_simple  mejor 200.0000  semilla 2  brecha 33.33%  media 193.3333
  3. continuous_simple  mejor 140.0000  semilla 2  brecha 53.33%  media 120.0000
  4. continuous_complex  mejor 135.0000  semilla 2  brecha 55.00%  media 131.6667
  5. vanilla_exploitation  mejor 90.0000  semilla 3  brecha 70.00%  media 85.0000
  6. vanilla_exploration  mejor 70.0000  semilla 3  brecha 76.67%  media 65.0000""",
}


def expected_resumen(estado, progreso, alcance, instance_keys):
    """Exact Spanish resumen.txt for the deterministic two-instance fixture."""
    header = (
        "Resumen de la campa\u00f1a GWO-DE\n"
        "============================\n"
        f"Estado: {estado}\n"
        f"Progreso: {progreso}\n"
        "Error: ninguno\n"
        "Archivos: mknapcb1.txt\n"
        "\u00cdndices: 1, 0\n"
        "Semillas: 3, 1, 2\n"
        "Modos: binary_simple, binary_complex, continuous_simple, continuous_complex, vanilla_exploitation, vanilla_exploration\n"
        "Binarizaci\u00f3n: lb2\n"
        "Se\u00f1al: ddtw\n"
        "Poblaci\u00f3n: 4\n"
        "Iteraciones: 12\n"
        "Ventana W: 5\n"
        "Banda: 0\n"
        "Meseta m\u00e1xima: 15\n"
        "Paciencia: 8\n"
        "G1: 0.5\n"
        "G2: 0.5\n"
        "G3 explotaci\u00f3n: 0.1\n"
        "G3 exploraci\u00f3n: 0.1\n"
        "DTW Kirito: s\u00ed"
    )
    final = (
        "Ranking final de los modos\n"
        "==========================\n"
        "Orden por la mejor corrida de cada modo, dentro de cada instancia: mayor fitness "
        "y, en empate, semilla menor.\n"
        "No se mezclan fitness de instancias distintas. La media se muestra y no determina "
        "el orden.\n"
        f"{alcance}"
    )
    parts = [header, RESUMEN_RULES]
    parts.extend(INSTANCE_TEXT[key] for key in instance_keys)
    parts.append(final)
    parts.extend(RANK_TEXT[key] for key in instance_keys)
    return "\n\n".join(parts) + "\n"


FITNESS_TABLE = {
    (1, "binary_simple"): {3: 40.0, 1: 50.0, 2: 40.0},
    (1, "binary_complex"): {3: 70.0, 1: 30.0, 2: 70.0},
    (1, "continuous_simple"): {3: 20.0, 1: 70.0, 2: 10.0},
    (1, "continuous_complex"): {3: 5.0, 1: 70.0, 2: 25.0},
    (1, "vanilla_exploitation"): {3: 10.0, 1: 15.0, 2: 12.0},
    (1, "vanilla_exploration"): {3: 8.0, 1: 9.0, 2: 7.0},
    (0, "binary_simple"): {3: 200.0, 1: 180.0, 2: 200.0},
    (0, "binary_complex"): {3: 160.0, 1: 200.0, 2: 150.0},
    (0, "continuous_simple"): {3: 120.0, 1: 100.0, 2: 140.0},
    (0, "continuous_complex"): {3: 130.0, 1: 130.0, 2: 135.0},
    (0, "vanilla_exploitation"): {3: 90.0, 1: 80.0, 2: 85.0},
    (0, "vanilla_exploration"): {3: 70.0, 1: 60.0, 2: 65.0},
}


class ResumenSnapshotRunner(FakeRunner):
    """Reads resumen.txt when the second instance starts, after the first rewrite."""

    def __init__(self, out):
        super().__init__(lambda index, policy, seed: FITNESS_TABLE[(index, policy)][seed])
        self.out = out
        self.progressive = None
        self.progressive_tmp = None

    def __call__(self, inst, cfg, **kw):
        index = int(inst["p"][0]) - 10
        if index == 0 and self.progressive is None:
            path = self.out / "resumen.txt"
            self.progressive = path.read_text(encoding="utf-8")
            self.progressive_tmp = (self.out / "resumen.txt.tmp").exists()
        return super().__call__(inst, cfg, **kw)


class RealSmokeTests(unittest.TestCase):
    def test_tiny_real_campaign_matches_direct_run_hybrid(self):
        instances = ROOT / "instancias"
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "smoke"
            out, err = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                code = campaign.main(["--instances-dir", str(instances), "--files",
                                      "mknapcb1.txt", "--indices", "0", "--seeds", "2", "1",
                                      "--pop", "6", "--iterations", "12", "--window", "5",
                                      "--out", str(out_dir)])
            self.assertEqual(code, 0, err.getvalue())
            inst = cargar_instancia(str(instances / "mknapcb1.txt"), 0)
            cfg = campaign.monitor_config(campaign.build_parser().parse_args(
                ["--window", "5"]))
            self.assertEqual(cfg, StagnationConfig(
                window=5, band=0, min_slope=2.0, plateau_max=15, patience=8, use_ddtw=True,
                adapt_thresholds=True, p_low=40.0, p_high=60.0, improvement_tol=0.0))
            _, rows = read_csv(out_dir / "mknapcb1" / "inst_00" / "runs.csv")
            direct = {}
            for row in rows:
                res = real_run_hybrid(inst, cfg, pop_size=6, iterations=12,
                                      seed=int(row["seed"]), policy=row["policy"])
                direct[(row["policy"], int(row["seed"]))] = res
                self.assertEqual(float(row["best_fitness"]), res["best_fitness"])
                self.assertEqual(float(row["known_optimum"]), 24381.0)
                self.assertEqual(float(row["gap_pct"]), res["gap_pct"])
                self.assertEqual(int(row["evaluations"]), res["evaluations"])
                self.assertEqual(int(row["generations"]), 12)
                self.assertEqual(int(row["changes"]), len(res["changes"]))
            for policy in POLICIES:
                winner = min((-direct[(policy, s)]["best_fitness"], s) for s in (1, 2))[1]
                data = json.loads((out_dir / "mknapcb1" / "inst_00" / policy / "best_run"
                                   / "result.json").read_text())
                self.assertEqual(data["seed"], winner)
                self.assertEqual(data["best_series"], direct[(policy, winner)]["best_series"])
                self.assertEqual(data["instance"], "mknapcb1.txt[0]")


if __name__ == "__main__":
    unittest.main()
