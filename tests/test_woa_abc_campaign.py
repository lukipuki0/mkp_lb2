"""Bounded campaign wiring checks: mocked exports, never launch the real campaign."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from woa_abc import campaign, benchmark
from woa_abc.profiles import POLICIES

ROOT = Path(__file__).resolve().parents[1]
INSTANCES = ROOT / 'instancias'


def stub_export(request):
    cfg = benchmark.build_hybrid_config(request)
    destination = benchmark.allocate_explicit_destination(request.destination)
    benchmark._write_json(destination / 'manifest.json', {
        'status': 'completed', 'counts': {'n_profile_changes': 0, 'n_recoveries': 2},
        'timing': {'solve_seconds': .5}})
    benchmark._write_json(destination / 'runs/run_01.json', {
        'best_value': 100., 'gap': 1., 'evaluations': 20, 'cycles_completed': 3,
        'scout_limit': 100 if cfg.variant == 'v1' else 3000,
        'evaluations_by_phase': {'scouts': 2},
        'metadata': {'config': {'reaction': cfg.reaction, 'variant': cfg.variant, 'policy': cfg.policy}}})
    return benchmark.ExportOutcome(0, destination, 'ok')


class CampaignTests(unittest.TestCase):
    def test_variant_campaign_plan_and_all_exports(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls = []
            def export(request):
                calls.append(request)
                return stub_export(request)
            output = io.StringIO()
            with patch.object(campaign, 'export_single_run', export), contextlib.redirect_stdout(output):
                result = campaign.run_campaign(INSTANCES, Path(tmp), timestamp='plan')
            self.assertEqual(result.code, 0, result.message)
            self.assertEqual(len(calls), 135)
            self.assertEqual(output.getvalue().count('start variant='), 135)
            self.assertEqual(output.getvalue().count('completed variant='), 135)
            self.assertEqual([(c.instance.name, c.instance_index, c.variant) for c in calls],
                [(f, i, v) for f in campaign.FILES for i in campaign.INDICES for v in campaign.VARIANTS])
            self.assertEqual({c.reaction for c in calls}, {'variant'})
            self.assertEqual({c.policy for c in calls}, {None})
            self.assertEqual({c.dtw_kirito for c in calls}, {False})
            first = calls[0]
            self.assertEqual((first.seed, first.population_size, first.iterations, first.max_evaluations),
                             (43, 30, 1000, 180030))
            self.assertIsNone(first.scout_limit)
            summary = json.loads((result.output_dir / 'campaign_summary.json').read_text())
            experiment = json.loads((result.output_dir / 'experiment.json').read_text())
            self.assertEqual((summary['status'], summary['runs_completed'], summary['runs_failed']),
                             ('completed', 135, 0))
            self.assertEqual(summary['groups'], {'100': 45, '250': 45, '500': 45})
            self.assertFalse(summary['superiority_claim'])
            self.assertEqual(experiment['settings']['reaction'], 'variant')
            self.assertEqual(experiment['selections'][0]['reference'], 24381.)
            self.assertEqual(experiment['selections'][0]['reference_source'], 'mkcbres.txt')
            import csv
            with (result.output_dir / 'campaign_results.csv').open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 135)
            self.assertEqual(rows[0]['recoveries'], '2')
            self.assertEqual(rows[0]['policy'], '')
            for selection in campaign._selections(INSTANCES):
                target = campaign.destination_for(result.output_dir, selection)
                self.assertEqual(target.parts[-4:], (str(selection.n), selection.path.stem,
                                 f'inst_{selection.index:02d}', selection.variant))
            with patch.object(campaign, 'export_single_run', side_effect=AssertionError('rerun')):
                with self.assertRaisesRegex(campaign.CampaignError, 'refusing to overwrite'):
                    campaign.run_campaign(INSTANCES, Path(tmp), timestamp='plan')

    def test_separate_policy_campaign_keeps_policy_and_kirito(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls = []
            def export(request):
                calls.append(request)
                return stub_export(request)
            with patch.object(campaign, 'export_single_run', export), contextlib.redirect_stdout(io.StringIO()):
                result = campaign.run_campaign(INSTANCES, Path(tmp), reaction='policy',
                    policy='binary_simple', dtw_kirito=True, timestamp='policy')
            self.assertEqual(result.code, 0)
            self.assertEqual(len(calls), 27)
            self.assertEqual({c.variant for c in calls}, {None})
            self.assertEqual({c.policy for c in calls}, {'binary_simple'})
            self.assertEqual({c.dtw_kirito for c in calls}, {True})
            self.assertEqual(len(campaign._selections(INSTANCES, reaction='policy')), 108)
            self.assertEqual({s.policy for s in campaign._selections(INSTANCES, reaction='policy')}, set(POLICIES))

    def test_single_instance_policy_campaign(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(campaign, 'export_single_run', side_effect=stub_export) as export, contextlib.redirect_stdout(io.StringIO()):
                result = campaign.run_campaign(INSTANCES, Path(tmp), reaction='policy',
                    files=['mknapcb1.txt'], indices=[2], seed=7, timestamp='subset')
            self.assertEqual(result.code, 0, result.message)
            calls = [call.args[0] for call in export.call_args_list]
            self.assertEqual([(r.instance.name, r.instance_index, r.policy) for r in calls],
                             [('mknapcb1.txt', 2, p) for p in POLICIES])
            for request in calls:
                self.assertEqual(request.destination.relative_to(result.output_dir).parts,
                                 ('100', 'mknapcb1', 'inst_02', request.policy))
                self.assertTrue((request.destination / 'manifest.json').is_file())
            experiment = json.loads((result.output_dir / 'experiment.json').read_text())
            self.assertEqual(experiment['runs_planned'], 4)
            self.assertEqual(experiment['runs_completed'], 4)
            self.assertEqual(experiment['groups'], {'100': 4, '250': 0, '500': 0})
            self.assertEqual(experiment['settings']['files'], ['mknapcb1.txt'])
            self.assertEqual(experiment['settings']['indices'], [2])

    def test_cli_subset_uses_files_order_and_sorted_indices(self):
        with tempfile.TemporaryDirectory() as tmp:
            stdout, stderr = io.StringIO(), io.StringIO()
            with patch.object(campaign, 'export_single_run', side_effect=stub_export) as export, contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                code = campaign.main(['--instances-dir', str(INSTANCES), '--output-root', tmp,
                    '--reaction', 'policy', '--policy', 'binary_simple',
                    '--files', 'mknapcb7.txt', 'mknapcb2.txt', 'mknapcb1.txt', '--indices', '2', '0'])
            self.assertEqual(code, 0, stderr.getvalue())
            self.assertEqual(stderr.getvalue(), '')
            calls = [call.args[0] for call in export.call_args_list]
            self.assertEqual([(r.instance.name, r.instance_index) for r in calls],
                [(f, i) for f in ('mknapcb1.txt', 'mknapcb2.txt', 'mknapcb7.txt') for i in (0, 2)])
            experiment = json.loads((calls[0].destination.parents[3] / 'experiment.json').read_text())
            self.assertEqual(experiment['settings']['files'], ['mknapcb1.txt', 'mknapcb2.txt', 'mknapcb7.txt'])
            self.assertEqual(experiment['settings']['indices'], [0, 2])

    def test_cli_invalid_subset_refused_before_allocation(self):
        cases = [
            (['--files', 'unknown.txt'], 'files must be basenames from mknapcb1.txt through mknapcb9.txt'),
            (['--files', 'dir/mknapcb1.txt'], 'files must be basenames from mknapcb1.txt through mknapcb9.txt'),
            (['--files', 'mknapcb1.txt', 'mknapcb1.txt'], 'files must not contain duplicates'),
            (['--indices', '-1'], 'indices must be nonnegative integers'),
            (['--indices', '0', '0'], 'indices must not contain duplicates'),
            (['--indices', '999999'], 'instance index must be in 0..29'),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            for options, message in cases:
                with self.subTest(options=options):
                    stdout, stderr = io.StringIO(), io.StringIO()
                    with patch.object(campaign, 'export_single_run') as export, contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                        code = campaign.main(['--instances-dir', str(INSTANCES), '--output-root', tmp] + options)
                    self.assertEqual(code, 1)
                    self.assertEqual(stdout.getvalue(), '')
                    self.assertEqual(stderr.getvalue(), f'error: {message}\n')
                    export.assert_not_called()
                    self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_api_noninteger_indices_refused_before_allocation(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'out'
            for index in (True, 1.5, '2'):
                with self.subTest(index=index), patch.object(campaign, 'export_single_run') as export:
                    with self.assertRaisesRegex(campaign.CampaignError, 'indices must be nonnegative integers'):
                        campaign.run_campaign(INSTANCES, out, indices=[index])
                    export.assert_not_called()
                    self.assertFalse(out.exists())

    def test_failed_export_stops_without_retry_and_retains_raw(self):
        with tempfile.TemporaryDirectory() as tmp:
            calls = []
            def export(request):
                calls.append(request)
                outcome = stub_export(request)
                if len(calls) == 3:
                    return benchmark.ExportOutcome(1, outcome.output_dir, 'plot failed')
                return outcome
            with patch.object(campaign, 'export_single_run', export), contextlib.redirect_stdout(io.StringIO()):
                result = campaign.run_campaign(INSTANCES, Path(tmp), timestamp='failure')
            self.assertEqual(result.code, 1)
            self.assertEqual([r.variant for r in calls], ['v0', 'v1', 'v2'])
            summary = json.loads((result.output_dir / 'campaign_summary.json').read_text())
            self.assertEqual((summary['status'], summary['runs_completed'], summary['runs_failed']), ('failed', 2, 1))
            self.assertEqual(summary['failed_selection']['variant'], 'v2')
            self.assertTrue((calls[-1].destination / 'runs/run_01.json').is_file())

    def test_preflight_rejects_mixed_modes_urls_invalid_seed_and_wrong_size_without_allocation(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'out'
            for values in (dict(policy='binary_simple'), dict(dtw_kirito=True), dict(seed=-1)):
                with self.subTest(values=values), patch.object(campaign, 'export_single_run', side_effect=AssertionError('solve')):
                    with self.assertRaises(ValueError):
                        campaign.run_campaign(INSTANCES, out, **values)
                    self.assertFalse(out.exists())
            with self.assertRaisesRegex(campaign.CampaignError, 'local path'):
                campaign.run_campaign('https://example.test/instances', out)
            with self.assertRaisesRegex(campaign.CampaignError, 'missing instance file'):
                campaign.run_campaign(Path(tmp), out)
            with patch.object(campaign.hybrid, '_load_local_instance', return_value=(None, {'n': 250})):
                with self.assertRaisesRegex(campaign.CampaignError, 'expected 100'):
                    campaign.run_campaign(INSTANCES, out)
            self.assertFalse(out.exists())

    def test_defaults_help_and_readme(self):
        args = campaign.build_parser().parse_args(['--instances-dir', 'instancias'])
        self.assertEqual(args.reaction, 'variant')
        self.assertIsNone(args.policy)
        self.assertEqual(args.max_evaluations, 180030)
        text = campaign.build_parser().format_help()
        for term in ('135', '45', '--reaction', '--policy', '--dtw-kirito', 'no superiority'):
            self.assertIn(term, text)
        readme = (ROOT / 'woa_abc/README.md').read_text(encoding='utf-8')
        self.assertIn('python -m woa_abc.campaign --instances-dir instancias', readme)
        self.assertIn('135', readme)


if __name__ == '__main__':
    unittest.main()
