"""Checks of the single-job Slurm campaign scripts (no Slurm needed)."""

import os
from pathlib import Path
import re
import shutil
import subprocess
import unittest

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "resultados"
BASH = shutil.which("bash")

FAMILIES = {"100": (1, 4, 7), "250": (2, 5, 8), "500": (3, 6, 9)}
SIZE_OF_FAMILY = {family: size for size, families in FAMILIES.items() for family in families}
SCRIPTS = {
    "gwo_de": REPO / "run_gwo_de_hpc.sh",
    "woa_abc": REPO / "run_woa_abc_hpc.sh",
}
MODULES = {"gwo_de": "gwo_de_dtw.campaign", "woa_abc": "woa_abc.seed_campaign"}
SEEDS = "--seeds " + " ".join(str(seed) for seed in range(1, 32))

# Stands in for the campaign: records its arguments and writes the layout the real
# campaign leaves in --out for one file and one index.
FAKE_PYTHON = r'''#!/bin/bash
printf "%s\n" "$*" >> "$FAKE_PYTHON_LOG"
args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
    case "${args[i]}" in
        --files) file=${args[i+1]} ;;
        --indices) index=${args[i+1]} ;;
        --out) out=${args[i+1]} ;;
    esac
done
FAIL_CHECK
inst="$out/${file%.txt}/inst_$(printf %02d "$index")"
mkdir -p "$inst/binary_simple/best_run"
printf "file,index\n%s,%s\n" "$file" "$index" > "$inst/runs.csv"
printf "file,policy\n%s,binary_simple\n" "$file" > "$inst/statistics.csv"
cp "$inst/runs.csv" "$out/summary.csv"
cp "$inst/statistics.csv" "$out/statistics.csv"
printf '{"status": "completed"}\n' > "$out/campaign_config.json"
printf "resumen\n" > "$out/resumen.txt"
exit 0
'''


def posix(path: Path) -> str:
    """Path as the bash under test sees it (/c/... on Git Bash)."""
    return subprocess.run([BASH, "-c", 'cd "$1" && pwd -P', "_", str(path)],
                          capture_output=True, text=True, check=True).stdout.strip()


def bash_path() -> str:
    return subprocess.run([BASH, "-c", 'printf "%s" "$PATH"'], capture_output=True,
                          text=True, check=True).stdout


def run(script: Path, *args: str, env: dict | None = None) -> subprocess.CompletedProcess:
    full_env = dict(os.environ)
    for name in [key for key in full_env if key.startswith("SLURM_")]:
        del full_env[name]
    full_env.update(env or {})
    return subprocess.run([BASH, str(script), *args], capture_output=True, text=True,
                          encoding="utf-8", cwd=REPO, env=full_env, timeout=120)


def results_listing() -> set:
    return {p.name for p in RESULTS.iterdir()} if RESULTS.is_dir() else set()


@unittest.skipIf(BASH is None, "bash is not available")
class HpcScriptTest(unittest.TestCase):

    def test_header_matches_run_parallel_hpc(self):
        for name, script in SCRIPTS.items():
            with self.subTest(script=name):
                header = [line.strip() for line in script.read_text(encoding="utf-8").splitlines()
                          if line.startswith("#SBATCH")]
                self.assertEqual(header, [
                    f"#SBATCH --job-name={name}",
                    "#SBATCH --partition=CPU",
                    "#SBATCH --nodes=1",
                    "#SBATCH --ntasks=1",
                    "#SBATCH --cpus-per-task=9",
                    "#SBATCH --threads-per-core=1",
                    "#SBATCH --mem=32G",
                    f"#SBATCH --output=./.hpc/logs/{name}_%j.log",
                    f"#SBATCH --error=./.hpc/errors/{name}_%j.error",
                    "#SBATCH --qos=long",
                ])
                self.assertNotIn(b"\r", script.read_bytes())

    def test_dry_run_plans_groups_in_order_with_first_three_indices(self):
        for name, script in SCRIPTS.items():
            with self.subTest(script=name):
                before = results_listing()
                result = run(script, "--dry-run")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stderr, "")
                groups = re.findall(r"^items_(\d+): 9 instancias en paralelo$", result.stdout, re.MULTILINE)
                self.assertEqual(groups, ["100", "250", "500"])
                plan = re.findall(r"^items_(\d+) task (\d): mknapcb(\d)\.txt index (\d)$",
                                  result.stdout, re.MULTILINE)
                expected = [(size, str(task), str(FAMILIES[size][task // 3]), str(task % 3))
                            for size in ("100", "250", "500") for task in range(9)]
                self.assertEqual(plan, expected)
                self.assertIn(f"Resultados: {posix(RESULTS)}/{name}_<SLURM_JOB_ID>/mknapcb<F>/inst_<II>/",
                              result.stdout)
                self.assertEqual(results_listing(), before, "dry run must not write")

    def test_dry_run_shows_the_python_command_of_each_hybrid(self):
        for name, script in SCRIPTS.items():
            with self.subTest(script=name):
                result = run(script, "--dry-run")
                self.assertEqual(result.returncode, 0, result.stderr)
                line = next(l for l in result.stdout.splitlines() if "--files mknapcb4.txt --indices 1" in l)
                self.assertIn(f"-m {MODULES[name]}", line)
                self.assertIn(SEEDS, line)
                self.assertTrue(line.endswith(f"/{name}_<SLURM_JOB_ID>/.staging/mknapcb4_inst_01/campaign"), line)
                self.assertNotIn("--policy ", line)

    def test_help_and_refusals(self):
        for name, script in SCRIPTS.items():
            with self.subTest(script=name):
                help_result = run(script, "--help")
                self.assertEqual(help_result.returncode, 0)
                self.assertIn(f"sbatch run_{name}_hpc.sh", help_result.stdout)
                self.assertIn("--dry-run", help_result.stdout)

                extra = run(script, "--dry-run", "otro")
                self.assertEqual(extra.returncode, 2)
                self.assertIn("Error:", extra.stderr)
                self.assertEqual(extra.stdout, "")

                before = results_listing()
                outside = run(script)
                self.assertEqual(outside.returncode, 2)
                self.assertIn("Error:", outside.stderr)
                self.assertIn("sbatch", outside.stderr)
                self.assertEqual(results_listing(), before)

    def _fake_job(self, name: str, fail_on: str | None = None) -> tuple[dict, Path, Path]:
        """Fake conda.sh and a fake python3; returns env, call log, experiment folder."""
        job_id = str(900000 + os.getpid() % 100000)
        experiment = RESULTS / f"{name}_{job_id}"
        self.assertFalse(experiment.exists(), f"refusing to reuse {experiment}")
        # Remove only what this test created; real results under resultados/ stay.
        self.addCleanup(shutil.rmtree, experiment, True)
        scratch = RESULTS / "hpc_test_tmp" / f"{name}_{os.getpid()}"
        shutil.rmtree(scratch, ignore_errors=True)
        (scratch / "bin").mkdir(parents=True)
        self.addCleanup(self._remove_if_empty, scratch.parent)
        self.addCleanup(shutil.rmtree, scratch, True)
        log = scratch / "python_calls.txt"
        fail_check = (f'[[ "$*" == *"{fail_on}"* ]] && {{ echo "falla simulada" >&2; exit 7; }}'
                      if fail_on else ":")
        fake_python = scratch / "bin" / "python3"
        fake_python.write_text(FAKE_PYTHON.replace("FAIL_CHECK", fail_check),
                               encoding="utf-8", newline="\n")
        fake_python.chmod(0o755)
        conda_sh = scratch / "conda.sh"
        # An unset variable inside conda.sh must not kill the job.
        conda_sh.write_text('conda() { : "${CONDA_UNSET_PROBE}"; }\n: "${CONDA_UNSET_PROBE2}"\n',
                            encoding="utf-8", newline="\n")
        env = {
            "SLURM_JOB_ID": job_id, "SLURM_SUBMIT_DIR": posix(REPO),
            "MKP_CONDA_SH": posix(scratch) + "/conda.sh", "FAKE_PYTHON_LOG": str(log),
            "PATH": posix(scratch / "bin") + ":" + bash_path(),
        }
        return env, log, experiment

    @staticmethod
    def _remove_if_empty(path: Path) -> None:
        try:
            path.rmdir()
        except OSError:
            pass

    @staticmethod
    def _instance_of(call: str) -> tuple[int, int]:
        match = re.search(r"--files mknapcb(\d)\.txt --indices (\d)", call)
        return int(match.group(1)), int(match.group(2))

    def test_job_writes_experiment_instance_mode_layout(self):
        for name, script in SCRIPTS.items():
            with self.subTest(script=name):
                env, log, experiment = self._fake_job(name)
                result = run(script, env=env)
                self.assertEqual(result.returncode, 0, result.stderr)
                calls = log.read_text(encoding="utf-8").splitlines()
                self.assertEqual(len(calls), 27)
                sizes = [SIZE_OF_FAMILY[self._instance_of(call)[0]] for call in calls]
                self.assertEqual(sizes, sorted(sizes, key=int), "groups must not overlap")
                for call in calls:
                    self.assertIn(f"-m {MODULES[name]}", call)
                    self.assertIn(SEEDS, call)
                self.assertEqual(sorted(p.name for p in experiment.iterdir()),
                                 [f"mknapcb{family}" for family in range(1, 10)])
                for family in range(1, 10):
                    self.assertEqual(sorted(p.name for p in (experiment / f"mknapcb{family}").iterdir()),
                                     ["inst_00", "inst_01", "inst_02"])
                    for index in range(3):
                        inst = experiment / f"mknapcb{family}" / f"inst_{index:02d}"
                        self.assertEqual((inst / "runs.csv").read_text(encoding="utf-8"),
                                         f"file,index\nmknapcb{family}.txt,{index}\n")
                        for kept in ("statistics.csv", "campaign_config.json", "run.log"):
                            self.assertTrue((inst / kept).is_file(), inst / kept)
                        self.assertTrue((inst / "binary_simple" / "best_run").is_dir())
                        if name == "gwo_de":
                            self.assertTrue((inst / "resumen.txt").is_file())
                self.assertIn(f"Resultados: {posix(experiment)}", result.stdout)

                again = run(script, env=env)
                self.assertEqual(again.returncode, 2)
                self.assertIn("ya existe", again.stderr)
                self.assertEqual(len(log.read_text(encoding="utf-8").splitlines()), 27)

    def test_failed_instance_keeps_staging_and_others_continue(self):
        for name, script in SCRIPTS.items():
            with self.subTest(script=name):
                env, log, experiment = self._fake_job(name, fail_on="mknapcb4.txt --indices 1")
                result = run(script, env=env)
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertIn("Instancias fallidas (1): mknapcb4/inst_01", result.stderr)
                self.assertIn("código 7", result.stderr)
                self.assertEqual(len(log.read_text(encoding="utf-8").splitlines()), 27)
                self.assertFalse((experiment / "mknapcb4" / "inst_01").exists())
                staging = experiment / ".staging" / "mknapcb4_inst_01"
                self.assertIn("falla simulada", (staging / "run.log").read_text(encoding="utf-8"))
                self.assertEqual(sorted(p.name for p in (experiment / ".staging").iterdir()),
                                 ["mknapcb4_inst_01"])
                self.assertTrue((experiment / "mknapcb9" / "inst_02" / "runs.csv").is_file())


if __name__ == "__main__":
    unittest.main()
