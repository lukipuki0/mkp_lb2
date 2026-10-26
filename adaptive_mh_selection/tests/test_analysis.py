"""Iteration-format summaries; historical evaluation campaigns stay isolated."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

TEMP = Path(__file__).resolve().parent / '.tmp'


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        TEMP.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=TEMP)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_native_iteration_summary_retains_overshoot(self):
        from adaptive_mh_selection.analysis import analyze_campaign
        (self.root / 'protocol.json').write_text(json.dumps({'schema_version': 2, 'budget_unit': 'native_iterations'}))
        for i, actual in enumerate((1003, 1012), 1):
            folder = self.root / 'adaptive_dtw' / f'run_{i:02d}'
            folder.mkdir(parents=True)
            (folder / 'summary.json').write_text(json.dumps({
                'status': 'success', 'requested_iterations': 1000, 'actual_iterations': actual,
                'iteration_overshoot': actual - 1000, 'best_fitness': 100 + i,
                'selector': 'adaptive', 'use_ddtw': False, 'seed': 41 + i,
            }))
        report = analyze_campaign(self.root)
        self.assertEqual(report['successful_runs'], 2)
        self.assertEqual(report['runs'][0]['actual_iterations'], 1003)
        self.assertEqual(report['runs'][1]['iteration_overshoot'], 12)
        self.assertFalse((self.root / 'analysis').exists())  # Read-only summary, no new plot families.

    def test_evaluation_format_is_rejected_without_writes(self):
        from adaptive_mh_selection.analysis import analyze_campaign
        (self.root / 'protocol.json').write_text(json.dumps({'schema_version': 1}))
        with self.assertRaisesRegex(ValueError, 'retired evaluation'):
            analyze_campaign(self.root)
        self.assertEqual(list(self.root.iterdir()), [self.root / 'protocol.json'])

    def test_grouped_schema_reads_every_metric_not_the_winner_only(self):
        from adaptive_mh_selection.analysis import analyze_campaign
        (self.root / 'protocol.json').write_text(json.dumps({
            'schema_version': 3, 'budget_unit': 'native_iterations', 'kind': 'grouped_mkp_campaign',
        }))
        folders = [
            self.root / 'items_100/mknapcb1/inst_00/adaptive_dtw',
            self.root / 'items_250/mknapcb2/inst_01/uniform_dtw',
        ]
        for folder in folders:
            folder.mkdir(parents=True)
        rows = []
        for run in range(1, 32):
            rows.append({
                'run': run, 'seed': 41 + run, 'status': 'failed' if run == 2 else 'success',
                'error': 'RuntimeError: boom' if run == 2 else '',
                'final_fitness': '' if run == 2 else (0 if run == 3 else 10 + run),
                'gap_pct': '', 'requested_iterations': 1000,
                'actual_iterations': '' if run == 2 else 1000 + run,
                'iteration_overshoot': '' if run == 2 else run,
                'n_blocks': '' if run == 2 else 3, 'transitions': '' if run == 2 else 2,
                'elapsed_seconds': '' if run == 2 else 0.1,
            })
        fields = list(rows[0])
        with (folders[0] / 'runs_resultados.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        with (folders[1] / 'runs_resultados.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerow({**rows[0], 'run': 1, 'seed': 42, 'status': 'success', 'final_fitness': 8,
                             'actual_iterations': 1004, 'iteration_overshoot': 4, 'n_blocks': 1,
                             'transitions': 0, 'elapsed_seconds': 0.2, 'error': ''})
        winner = folders[0] / 'best_run_31'
        winner.mkdir()
        (winner / 'summary.json').write_text(json.dumps({'status': 'success', 'best_fitness': 99999, 'seed': 72}))
        before = sorted(path.relative_to(self.root).as_posix() for path in self.root.rglob('*'))
        report = analyze_campaign(self.root)
        after = sorted(path.relative_to(self.root).as_posix() for path in self.root.rglob('*'))
        self.assertEqual(before, after)
        self.assertEqual(report['schema_version'], 3)
        self.assertEqual(report['kind'], 'grouped_mkp_campaign')
        self.assertEqual(len(report['metrics']), 32)
        self.assertEqual(report['distinct_cases'], 2)
        self.assertEqual(report['successful_runs'], 31)
        self.assertEqual(report['failed_runs'], 1)
        self.assertIsNone(report['metrics'][1]['fitness'])
        self.assertNotEqual(report['metrics'][1]['fitness'], 0)
        self.assertIsInstance(report['metrics'][0]['seed'], int)
        self.assertEqual(report['metrics'][0]['seed'], 42)
        self.assertEqual(report['metrics'][2]['fitness'], 0.0)
        self.assertIsInstance(report['metrics'][0]['iterations'], int)
        self.assertEqual(report['metrics'][0]['iterations'], 1001)
        self.assertEqual(report['metrics'][0]['elapsed_seconds'], 0.1)
        self.assertEqual(report['metrics'][0]['n_blocks'], 3)
        self.assertEqual(report['metrics'][0]['transitions'], 2)
        self.assertIsNone(report['metrics'][1]['elapsed_seconds'])
        self.assertIsNone(report['metrics'][1]['n_blocks'])
        self.assertEqual(report['metrics'][-1]['transitions'], 0)
        self.assertEqual(report['metrics'][0]['case'], 'items_100/mknapcb1/inst_00')
        self.assertEqual(report['metrics'][0]['condition'], 'adaptive_dtw')
        self.assertEqual(report['metrics'][-1]['case'], 'items_250/mknapcb2/inst_01')
        self.assertEqual(report['metrics'][-1]['condition'], 'uniform_dtw')
        self.assertNotIn(99999, [metric['fitness'] for metric in report['metrics']])
        self.assertFalse((self.root / 'analysis').exists())

    def test_grouped_schema_without_metrics_is_rejected_without_writes(self):
        from adaptive_mh_selection.analysis import analyze_campaign
        (self.root / 'protocol.json').write_text(json.dumps({
            'schema_version': 3, 'budget_unit': 'native_iterations', 'kind': 'grouped_mkp_campaign',
        }))
        with self.assertRaisesRegex(ValueError, 'no run summaries'):
            analyze_campaign(self.root)
        self.assertEqual(list(self.root.iterdir()), [self.root / 'protocol.json'])

    def _write_metrics(self, path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def _metric_row(self, seed, fitness):
        return {
            'run': 1, 'seed': seed, 'status': 'success', 'error': '',
            'final_fitness': fitness, 'gap_pct': '', 'requested_iterations': 1000,
            'actual_iterations': 1000, 'iteration_overshoot': 0, 'n_blocks': 1,
            'transitions': 0, 'elapsed_seconds': 0.1,
        }

    def test_grouped_reader_uses_protocol_paths_and_keeps_condition_without_directory(self):
        from adaptive_mh_selection.analysis import analyze_campaign
        campaign = self.root / 'flat'
        campaign.mkdir()
        (campaign / 'protocol.json').write_text(json.dumps({
            'schema_version': 3, 'budget_unit': 'native_iterations', 'kind': 'grouped_mkp_campaign',
            'cases': [{
                'family': 1, 'index': 0, 'n': 100, 'relative_output_path': '',
                'conditions': [{'condition': 'adaptive_ddtw', 'relative_output_path': ''}],
            }, {
                'family': 2, 'index': 1, 'n': 250, 'relative_output_path': 'items_250/mknapcb2_inst_01',
                'conditions': [
                    {'condition': 'adaptive_dtw', 'relative_output_path': 'items_250/mknapcb2_inst_01/adaptive_dtw'},
                    {'condition': 'uniform_dtw', 'relative_output_path': 'items_250/mknapcb2_inst_01/uniform_dtw'},
                ],
            }],
        }), encoding='utf-8')
        self._write_metrics(campaign / 'runs_resultados.csv', [self._metric_row(42, 7)])
        self._write_metrics(campaign / 'items_250/mknapcb2_inst_01/adaptive_dtw/runs_resultados.csv',
                            [self._metric_row(43, 8)])
        self._write_metrics(campaign / 'items_250/mknapcb2_inst_01/uniform_dtw/runs_resultados.csv',
                            [self._metric_row(44, 9)])
        decoy = campaign / 'items_100/mknapcb1/inst_00/adaptive_ddtw/runs_resultados.csv'
        self._write_metrics(decoy, [self._metric_row(99, 99999)])
        before = sorted(path.relative_to(campaign).as_posix() for path in campaign.rglob('*'))
        report = analyze_campaign(campaign)
        after = sorted(path.relative_to(campaign).as_posix() for path in campaign.rglob('*'))
        self.assertEqual(before, after)
        self.assertEqual(
            [(metric['case'], metric['condition'], metric['fitness'], metric['seed']) for metric in report['metrics']],
            [
                ('items_100/mknapcb1_inst_00', 'adaptive_ddtw', 7.0, 42),
                ('items_250/mknapcb2_inst_01', 'adaptive_dtw', 8.0, 43),
                ('items_250/mknapcb2_inst_01', 'uniform_dtw', 9.0, 44),
            ],
        )
        self.assertNotIn(99999, [metric['fitness'] for metric in report['metrics']])
        self.assertEqual(report['distinct_cases'], 2)
        self.assertEqual(report['case_conditions'], 3)

    def test_protocol_paths_that_escape_are_not_read(self):
        from adaptive_mh_selection.analysis import analyze_campaign
        campaign = self.root / 'campaign'
        campaign.mkdir()
        secret = self.root / 'secret'
        secret.mkdir()
        self._write_metrics(secret / 'runs_resultados.csv', [self._metric_row(42, 5)])
        (campaign / 'protocol.json').write_text(json.dumps({
            'schema_version': 3, 'budget_unit': 'native_iterations', 'kind': 'grouped_mkp_campaign',
            'cases': [{'relative_output_path': '../secret', 'conditions': [
                {'condition': 'adaptive_ddtw', 'relative_output_path': '../secret'},
            ]}],
        }), encoding='utf-8')
        before = sorted(path.relative_to(self.root).as_posix() for path in self.root.rglob('*'))
        with self.assertRaisesRegex(ValueError, 'no run summaries'):
            analyze_campaign(campaign)
        after = sorted(path.relative_to(self.root).as_posix() for path in self.root.rglob('*'))
        self.assertEqual(before, after)
