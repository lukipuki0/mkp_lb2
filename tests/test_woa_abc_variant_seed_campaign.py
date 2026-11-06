"""Bounded public campaign checks; all results live in system temporary folders."""
import contextlib
import csv
import io
import json
from pathlib import Path
import statistics
import tempfile
import unittest
from unittest.mock import patch

from woa_abc import variant_seed_campaign as campaign
from woa_abc.benchmark import ExportOutcome, EXPECTED_ARTIFACTS


class VariantSeedsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.inputs = self.root / 'inputs'
        self.inputs.mkdir()
        # Valid, tiny synthetic OR-Library selection with required family item count.
        (self.inputs / 'mknapcb1.txt').write_text(
            '1\n100 1 100\n' + '1 ' * 100 + '\n' + '1 ' * 100 + '\n50\n')
        self.out = self.root / 'results'

    def invoke(self, *extra):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = campaign.main(['--instances-dir', str(self.inputs), '--files', 'mknapcb1.txt',
                                  '--indices', '0', '--out', str(self.out), *extra])
        return code, stdout.getvalue(), stderr.getvalue()

    def fake_export(self, request):
        dest = request.destination
        (dest / 'runs').mkdir(parents=True)
        value = 100 if request.seed in (1, 31) else request.seed
        (dest / 'runs/run_01.json').write_text(json.dumps({'best_value': value,
            'gap': 100 - value, 'evaluations': 30, 'cycles_completed': 1,
            'metadata': {'seed': request.seed, 'config': {'reaction': 'variant', 'variant': request.variant}}}))
        (dest / 'manifest.json').write_text(json.dumps({'status': 'completed',
            'counts': {'n_profile_changes': 0, 'n_recoveries': 2}, 'timing': {'solve_seconds': 0.5}}))
        return ExportOutcome(0, dest)

    def test_all_rows_statistics_winner_tie_and_layout(self):
        def comparison(raws, folder, *args):
            self.assertEqual([r['metadata']['seed'] for r in raws], [1] * 5)
            for ext in ('png', 'pdf'):
                (folder / f'convergence_comparison.{ext}').write_text('plot')
        with patch.object(campaign, 'export_single_run', side_effect=self.fake_export) as exporter, \
                patch.object(campaign, 'plot_comparison', side_effect=comparison):
            code, stdout, stderr = self.invoke('--seeds', '31', *map(str, range(1, 31)))
        self.assertEqual((code, stderr), (0, ''))
        self.assertIn('completed 155/155', stdout)
        self.assertEqual(exporter.call_count, 155)
        self.assertTrue(all(c.kwargs == {} and c.args[0].scout_limit is None for c in exporter.call_args_list))
        folder = self.out / 'mknapcb1/inst_00'
        self.assertEqual({p.name for p in folder.iterdir()}, {
            'campaign_config.json', 'run.log', 'resumen.txt', 'runs.csv', 'statistics.csv',
            'convergence_comparison.png', 'convergence_comparison.pdf', 'v0', 'v1', 'v2', 'v3', 'v4'})
        with (folder / 'runs.csv').open() as fh:
            rows = list(csv.DictReader(fh))
        self.assertEqual(len(rows), 155)
        self.assertEqual({r['variant'] for r in rows}, {f'v{i}' for i in range(5)})
        with (folder / 'statistics.csv').open() as fh:
            stats = list(csv.DictReader(fh))
        values = [100, *range(2, 31), 100]
        self.assertEqual(len(stats), 5)
        for row in stats:
            self.assertEqual(int(row['best_seed']), 1)
            for key, expected in [('best_fitness', 100), ('worst_fitness', 2),
                                  ('mean_fitness', statistics.mean(values)),
                                  ('median_fitness', statistics.median(values)),
                                  ('std_fitness', statistics.stdev(values))]:
                self.assertAlmostEqual(float(row[key]), expected)
            raw = json.loads((folder / row['variant'] / 'best_run/runs/run_01.json').read_text())
            self.assertEqual(raw['metadata']['seed'], 1)
        self.assertFalse(list(self.out.rglob('seed_*')))
        config = json.loads((folder / 'campaign_config.json').read_text())
        self.assertEqual((config['status'], config['completed_runs']), ('completed', 155))
        before = (folder / 'runs.csv').read_bytes()
        self.assertEqual(self.invoke()[0], 2)
        self.assertEqual((folder / 'runs.csv').read_bytes(), before)

    def test_invalid_and_dry_run_leave_no_output(self):
        for args in [('--seeds', '1', '1'), ('--seeds', '-1'), ('--indices', '3'),
                     ('--population-size', '0'), ('--files', '../mknapcb1.txt'), ('--policy', 'binary_simple')]:
            code, stdout, stderr = self.invoke(*args)
            self.assertEqual(code, 2)
            self.assertEqual(stdout, '')
            self.assertIn('error:', stderr)
            self.assertFalse(self.out.exists())
        code, stdout, stderr = self.invoke('--dry-run')
        self.assertEqual((code, stderr), (0, ''))
        self.assertIn('runs=155', stdout)
        self.assertFalse(self.out.exists())

    def test_failure_keeps_evidence_and_stops(self):
        def fail(request):
            outcome = self.fake_export(request)
            if request.variant == 'v1':
                return ExportOutcome(7, request.destination, 'simulated failure')
            return outcome
        with patch.object(campaign, 'export_single_run', side_effect=fail) as exporter:
            code, stdout, stderr = self.invoke()
        self.assertEqual(code, 1)
        self.assertIn('simulated failure', stderr)
        self.assertEqual(exporter.call_count, 2)
        folder = self.out / 'mknapcb1/inst_00'
        config = json.loads((folder / 'campaign_config.json').read_text())
        self.assertEqual((config['status'], config['completed_runs']), ('failed', 1))
        self.assertTrue((folder / '.runs/v1/seed_01/manifest.json').exists())
        with (folder / 'runs.csv').open() as fh:
            self.assertEqual([r['status'] for r in csv.DictReader(fh)], ['completed', 'failed'])

    def test_real_small_solver_full_winner_exports(self):
        code, stdout, stderr = self.invoke('--seeds', '1', '--population-size', '2',
                                         '--iterations', '2', '--max-evaluations', '26')
        self.assertEqual((code, stderr), (0, ''))
        folder = self.out / 'mknapcb1/inst_00'
        for variant in range(5):
            winner = folder / f'v{variant}/best_run'
            for artifact in EXPECTED_ARTIFACTS:
                self.assertTrue((winner / artifact).is_file(), artifact)
            manifest = json.loads((winner / 'manifest.json').read_text())
            self.assertTrue(manifest['output_completeness']['complete'])
            self.assertEqual(manifest['effective_config']['variant'], f'v{variant}')
            if variant == 1:
                self.assertEqual(manifest['effective_config']['scout_limit'], 100)
        self.assertTrue((folder / 'convergence_comparison.pdf').is_file())


if __name__ == '__main__':
    unittest.main()
