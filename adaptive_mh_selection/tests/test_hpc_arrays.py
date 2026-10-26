"""Public Bash contracts; all scheduler, activation and optimizer calls are fakes."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
BASH = shutil.which("bash")
SCRIPTS = ("submit_adaptive_mkp_arrays.sh", "run_adaptive_mkp_array.sh")
THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")


@unittest.skipUnless(BASH, "Bash is required (Git Bash on Windows or Bash on Linux)")
class HPCArrayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="mkp arrays ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for script in SCRIPTS:
            shutil.copyfile(ROOT / script, self.root / script)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.shell_root = subprocess.run(
            [BASH, "-c", 'cd -- "$1" && pwd -P', "bash", str(self.root)],
            check=True, capture_output=True, text=True, timeout=20,
        ).stdout.strip()
        self.env = os.environ.copy()
        for key in tuple(self.env):
            if key.startswith("SLURM_") or key.startswith("MKP_"):
                self.env.pop(key)
        self.env.update(PATH=str(self.bin) + os.pathsep + self.env["PATH"],
                        TEST_STATE=self.shell_root,
                        MKP_CONDA_SH=self.shell_root + "/conda.sh")
        self.write("conda.sh", 'conda() { printf "%s\\n" "$*" >> "$TEST_STATE/activation"; }\n')
        self.write("bin/sbatch", r'''#!/bin/bash
set -eu
n=0
[[ ! -f "$TEST_STATE/count" ]] || read -r n < "$TEST_STATE/count"
n=$((n + 1))
printf '%s\n' "$n" > "$TEST_STATE/count"
printf '%s\n' "$@" > "$TEST_STATE/sbatch.$n"
for arg in "$@"; do
    case "$arg" in
        --output=*|--error=*) [[ -d "$(dirname -- "${arg#*=}")" ]] || exit 91 ;;
    esac
done
[[ "${FAIL_AT:-0}" != "$n" ]] || { echo 'fake scheduler failure' >&2; exit 37; }
printf '%s;mockcluster\n' "$((7000 + n))"
''')
        self.write("bin/python3", r'''#!/bin/bash
set -eu
printf '%s\n' "$@" > "$TEST_STATE/python.args"
for key in OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS VECLIB_MAXIMUM_THREADS NUMEXPR_NUM_THREADS PYTHONUNBUFFERED MPLBACKEND MPLCONFIGDIR; do
    printf '%s=%s\n' "$key" "${!key}"
done > "$TEST_STATE/python.env"
[[ -d "$MPLCONFIGDIR" ]] || exit 92
while (( $# )); do
    if [[ "$1" == --output ]]; then mkdir -- "$2"; break; fi
    shift
done
exit "${PYTHON_EXIT:-0}"
''')

    def write(self, name, content):
        path = self.root / name
        path.write_text(content, encoding="utf-8", newline="\n")
        path.chmod(0o755)

    def call(self, script, *args, env=None):
        return subprocess.run([BASH, str(self.root / script), *args], cwd=self.root,
                              env=self.env if env is None else env,
                              capture_output=True, text=True, timeout=30)

    def submit(self, *args, **kwargs):
        return self.call(SCRIPTS[0], *args, **kwargs)

    def batch(self, tag="study"):
        return self.root / "resultados" / "adaptive_mh_selection" / ("batch_" + tag)

    def snapshot(self):
        return {str(p.relative_to(self.root)): p.read_bytes() if p.is_file() else None
                for p in self.root.rglob("*")}

    def reject(self, invoke):
        before = self.snapshot()
        result = invoke()
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertTrue(result.stderr.strip())
        self.assertEqual(result.stdout, "")
        self.assertEqual(self.snapshot(), before)

    def worker(self, size="100", task="0", **kwargs):
        env = dict(self.env, SLURM_JOB_ID="7001", SLURM_ARRAY_JOB_ID="7000",
                   SLURM_ARRAY_TASK_ID=task)
        env.update(kwargs.pop("extra_env", {}))
        return self.call(SCRIPTS[1], self.shell_root, size, self.shell_root + "/batch",
                         env=env, **kwargs)

    def test_all_27_mappings_and_private_outputs(self):
        (self.root / "batch").mkdir()
        outputs, caches, cases = set(), set(), set()
        for size, families in ((100, (1, 4, 7)), (250, (2, 5, 8)), (500, (3, 6, 9))):
            for task in range(9):
                with self.subTest(size=size, task=task):
                    result = self.worker(str(size), str(task))
                    self.assertEqual(result.returncode, 0, result.stderr)
                    args = (self.root / "python.args").read_text().splitlines()
                    family, index = families[task // 3], task % 3
                    output = f"{self.shell_root}/batch/items_{size}/mknapcb{family}_inst_{index:02d}"
                    reservation = (f"{self.shell_root}/batch/.reservations/items_{size}/"
                                   f"mknapcb{family}_inst_{index:02d}")
                    self.assertEqual(args, ["-B", "-u", "-m", "adaptive_mh_selection.campaign",
                                           "--families", str(family), "--indices", str(index),
                                           "--runs", "31", "--seed", "42", "--iterations", "1000",
                                           "--selectors", "adaptive", "--modes", "ddtw",
                                           "--selection-trace", "--output", output])
                    self.assertNotIn("/campaign", output)
                    settings = dict(line.split("=", 1) for line in
                                    (self.root / "python.env").read_text().splitlines())
                    for key in (*THREADS, "PYTHONUNBUFFERED"):
                        self.assertEqual(settings[key], "1")
                    self.assertEqual(settings["MPLBACKEND"], "Agg")
                    self.assertEqual(settings["MPLCONFIGDIR"], reservation + "/mplconfig")
                    self.assertNotEqual(Path(settings["MPLCONFIGDIR"]).parent.as_posix(), output)
                    self.assertTrue((self.root / "batch/.reservations" / f"items_{size}"
                                     / f"mknapcb{family}_inst_{index:02d}" / "mplconfig").is_dir())
                    self.assertTrue((self.root / "batch" / f"items_{size}"
                                     / f"mknapcb{family}_inst_{index:02d}").is_dir())
                    outputs.add(output)
                    caches.add(settings["MPLCONFIGDIR"])
                    cases.add((family, index))
        self.assertEqual(len(outputs), 27)
        self.assertEqual(len(caches), 27)
        self.assertEqual(cases, {(f, i) for f in range(1, 10) for i in range(3)})
        self.assertEqual((self.root / "activation").read_text().splitlines(), ["activate mkp_env"] * 27)
        self.reject(lambda: self.worker())

    def test_submit_three_arrays_and_record_ids(self):
        result = self.submit("study")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.root / "count").read_text().strip(), "3")
        logs = []
        for number, size in enumerate((100, 250, 500), 1):
            args = (self.root / f"sbatch.{number}").read_text().splitlines()
            self.assertIn("--parsable", args)
            self.assertIn("--array=0-8%9", args)
            self.assertIn("--chdir=" + self.shell_root, args)
            self.assertEqual(args[-4:], [self.shell_root + "/" + SCRIPTS[1], self.shell_root,
                                         str(size), self.shell_root + "/resultados/adaptive_mh_selection/batch_study"])
            for flag, directory, suffix in (("--output=", "logs", "out"), ("--error=", "errors", "err")):
                name = next(arg[len(flag):] for arg in args if arg.startswith(flag))
                self.assertEqual(name, f"{args[-1]}/{directory}/study_items_{size}_%A_%a.{suffix}")
                logs.append(name)
        self.assertEqual(len(set(logs)), 6)
        self.assertEqual((self.batch() / "jobs.tsv").read_text().splitlines(),
                         ["size\tarray_job_id", "100\t7001;mockcluster", "250\t7002;mockcluster", "500\t7003;mockcluster"])
        self.reject(lambda: self.submit("study"))
        self.reject(lambda: self.submit("--dry-run", "study"))

    def test_dry_run_is_three_plans_without_effects(self):
        before = self.snapshot()
        for args in (("--dry-run", "safe.tag-1"), ("--dry-run",)):
            result = self.submit(*args)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(len(result.stdout.splitlines()), 3)
            self.assertEqual(result.stdout.count("--array=0-8%9"), 3)
            self.assertEqual(result.stdout.count("sbatch --parsable"), 3)
            self.assertEqual(self.snapshot(), before)
        # Force lookup failure even on a host where sbatch happens to be installed.
        no_scheduler = dict(self.env, BASH_ENV=self.shell_root + "/no_scheduler.sh")
        self.write("no_scheduler.sh", 'command() { return 1; }\n')
        result = self.submit("--dry-run", "offline", env=no_scheduler)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.reject(lambda: self.submit("offline", env=no_scheduler))

    def test_invalid_submit_arguments_have_no_effects(self):
        for args in (("",), ("..",), ("../escape",), ("bad/tag",), ("bad tag",),
                     ("--unknown",), ("one", "two"), ("--dry-run", "x", "y")):
            with self.subTest(args=args):
                self.reject(lambda: self.submit(*args))

    def test_worker_rejects_invalid_context_before_activation(self):
        (self.root / "batch").mkdir()
        for size in ("240", "0100", "", "../100"):
            self.reject(lambda: self.worker(size=size))
        for task in ("", "-1", "9", "00", "1x", "1+1"):
            self.reject(lambda: self.worker(task=task))
        for key in ("SLURM_ARRAY_JOB_ID", "SLURM_JOB_ID"):
            for value in ("", "oops"):
                self.reject(lambda: self.worker(extra_env={key: value}))
        for args in ((), (self.shell_root, "100"), ("relative", "100", self.shell_root),
                     (self.shell_root, "100", "relative"),
                     (self.shell_root, "100", self.shell_root, "extra")):
            self.reject(lambda: self.call(SCRIPTS[1], *args))
        output = self.root / "batch/items_100/mknapcb1_inst_00"
        output.mkdir(parents=True)
        self.reject(lambda: self.worker())
        shutil.rmtree(output)
        reservation = self.root / "batch/.reservations/items_100/mknapcb1_inst_00"
        reservation.mkdir(parents=True)
        self.reject(lambda: self.worker())

    def test_partial_submission_preserves_failure_and_accepted_ids(self):
        result = self.submit("partial", env=dict(self.env, FAIL_AT="2"))
        self.assertEqual(result.returncode, 37)
        self.assertIn("fake scheduler failure", result.stderr)
        self.assertIn("7001;mockcluster", result.stderr)
        self.assertEqual((self.root / "count").read_text().strip(), "2")
        self.assertFalse((self.root / "sbatch.3").exists())
        self.assertEqual((self.batch("partial") / "jobs.tsv").read_text().splitlines(),
                         ["size\tarray_job_id", "100\t7001;mockcluster"])
        self.reject(lambda: self.submit("partial"))

    def test_worker_propagates_python_failure_without_resuming(self):
        (self.root / "batch").mkdir()
        result = self.worker(extra_env={"PYTHON_EXIT": "23"})
        self.assertEqual(result.returncode, 23)
        self.reject(lambda: self.worker())

    def test_resource_directives_and_lf(self):
        text = (ROOT / SCRIPTS[1]).read_text()
        for directive in ("partition=CPU", "qos=normal", "nodes=1", "ntasks=1",
                          "cpus-per-task=1", "mem=4G", "time=24:00:00"):
            self.assertIn("#SBATCH --" + directive + "\n", text)
        self.assertIn('${HOME}/miniconda3/etc/profile.d/conda.sh', text)
        for script in SCRIPTS:
            self.assertNotIn(b"\r", (ROOT / script).read_bytes())


if __name__ == "__main__":
    unittest.main()
