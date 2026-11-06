"""Public shell checks for the variant Slurm array; no cluster required."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'run_woa_abc_variants_hpc.sh'
BASH = shutil.which('bash')


@unittest.skipUnless(BASH, 'bash unavailable')
class VariantSlurmTest(unittest.TestCase):
    def run_script(self, *args, env=None):
        clean = {k: v for k, v in os.environ.items() if not k.startswith('SLURM_')}
        clean.update(env or {})
        return subprocess.run([BASH, str(SCRIPT), *args], cwd=ROOT,
                              env=clean, capture_output=True, text=True, timeout=20)

    def test_plan(self):
        result = self.run_script('--dry-run')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stderr, '')
        commands = [line for line in result.stdout.splitlines() if ' -m woa_abc.variant_seed_campaign ' in line]
        self.assertEqual(len(commands), 27)
        for family in range(1, 10):
            for index in range(3):
                selected = [s for s in commands if f'--files mknapcb{family}.txt --indices {index} ' in s]
                self.assertEqual(len(selected), 1)
                for command in selected:
                    self.assertIn('--seeds ' + ' '.join(map(str, range(1, 32))), command)
                    self.assertNotIn('seed_01', command)
                    self.assertIn('--max-evaluations 180030', command)
                    self.assertNotIn('--scout-limit', command)
        self.assertIn('#SBATCH --array=0-26%9', SCRIPT.read_text())

    def test_help_and_refusals(self):
        self.assertEqual(self.run_script('--help').returncode, 0)
        for args, env in [(('bad',), {}), ((), {}),
                          ((), {'SLURM_ARRAY_JOB_ID': '123', 'SLURM_ARRAY_TASK_ID': '27'})]:
            result = self.run_script(*args, env=env)
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertEqual(result.stdout, '')
            self.assertIn('Error:', result.stderr)

    def test_execution_and_failure_preserve_evidence(self):
        # Fake submission repo and all generated outputs stay outside the worktree.
        with tempfile.TemporaryDirectory(prefix='variant-test-') as directory:
            scratch = Path(directory)
            canonical = subprocess.run([BASH, '-c', 'cd "$1" && pwd -P', '_', directory],
                                       capture_output=True, text=True, check=True).stdout.strip()
            (scratch / 'woa_abc').mkdir()
            (scratch / 'woa_abc/variant_seed_campaign.py').write_text('# stub repo\n')
            (scratch / 'instancias').mkdir()
            (scratch / 'conda.sh').write_text('conda() { export PATH="' + canonical + '/bin:$PATH"; }\n')
            (scratch / 'bin').mkdir()
            python = scratch / 'bin/python3'
            python.write_text('#!/bin/bash\nprintf "%s\\n" "$*" >> "' + canonical + '/calls"\n'
                              'mkdir -p "' + canonical + '/resultados/woa_abc_variants_123/mknapcb1/inst_00"\n'
                              'echo simulated > "' + canonical + '/resultados/woa_abc_variants_123/mknapcb1/inst_00/run.log"\n'
                              'echo simulated >&2; exit 7\n')
            python.chmod(0o755)
            job = '123'
            output = scratch / 'resultados' / f'woa_abc_variants_{job}'
            self.assertFalse(output.exists())
            env = {'SLURM_ARRAY_JOB_ID': job, 'SLURM_ARRAY_TASK_ID': '0',
                   'SLURM_SUBMIT_DIR': canonical, 'MKP_CONDA_SH': canonical + '/conda.sh'}
            result = self.run_script(env=env)
            self.assertEqual(result.returncode, 7, result.stderr)
            calls = (scratch / 'calls').read_text().splitlines()
            self.assertEqual(len(calls), 1, 'one campaign invocation per instance')
            self.assertIn('--files mknapcb1.txt --indices 0', calls[0])
            self.assertIn('simulated', (output / 'mknapcb1/inst_00/run.log').read_text())
            again = self.run_script(env=env)
            self.assertEqual(again.returncode, 2)
            self.assertEqual((scratch / 'calls').read_text().splitlines(), calls)
            python.write_text('#!/bin/bash\nprintf "%s\\n" "$*" >> "' + canonical + '/calls"\nexit 0\n')
            env['SLURM_ARRAY_TASK_ID'] = '1'
            success = self.run_script(env=env)
            self.assertEqual(success.returncode, 0, success.stderr)
            self.assertEqual(success.stderr, '')
            self.assertIn('Completadas 31 semillas x 5 variantes', success.stdout)
            self.assertEqual(len((scratch / 'calls').read_text().splitlines()), 2)
            self.assertFalse(list(output.rglob('seed_*')))


if __name__ == '__main__':
    unittest.main()
