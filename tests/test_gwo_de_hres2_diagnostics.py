"""S9 diagnostics through saved records and both public entry points."""
import contextlib
import copy
import io
import json
import weakref
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from gwo_de_hres2 import campaign, run
from gwo_de_hres2.control import run_search
from tests.test_gwo_de_hres2 import FakeFunction, monitor_cfg

ARTIFACTS = ('result.json', 'fitness_convergence.png', 'parameters.png',
             'temporal_diagnostics.png', 'temporal_diagnostics.pdf',
             'temporal_series.npz', 'temporal_series.csv', 'temporal_metadata.json')


class DiagnosticsTests(unittest.TestCase):
    def record(self, policy='binary_simple', signal=True):
        return run_search(FakeFunction(), monitor_cfg(signal), policy, 4, 22, 1)

    def test_export_releases_renderers_between_artifacts_and_on_return(self):
        from matplotlib.backends.backend_agg import RendererAgg
        from gwo_de_hres2.diagnostics import export_run, plot_outcomes
        rec = self.record()
        refs, live_before = [], []
        original = RendererAgg.__init__

        def tracked(renderer, *args, **kwargs):
            live_before.append(sum(ref() is not None for ref in refs))
            refs.append(weakref.ref(renderer))
            original(renderer, *args, **kwargs)

        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(RendererAgg, '__init__', tracked):
            export_run(rec, Path(tmp, 'winner'))
            self.assertFalse(any(ref() is not None for ref in refs))
            plot_outcomes([rec], Path(tmp, 'convergence_comparison.png'), 'comparison')
            self.assertFalse(any(ref() is not None for ref in refs))
            self.assertEqual(max(live_before), 0)
            for path in (Path(tmp, 'winner', 'temporal_diagnostics.png'),
                         Path(tmp, 'convergence_comparison.pdf')):
                self.assertGreater(path.stat().st_size, 0)

    def test_campaign_postprocessing_failure_preserves_all_winners(self):
        argv = ['--signals', 'dtw', '--policies', 'binary_simple', 'binary_complex',
                '--seeds', '1', '--pop-size', '4', '--iterations', '3']
        stages = ('export_run', 'plot_convergence', 'group_statistics', '_write_csv')
        for stage in stages:
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp, 'campaign')
                out, err = io.StringIO(), io.StringIO()
                with mock.patch.object(campaign, 'load_function', FakeFunction), \
                        mock.patch.object(campaign, stage, side_effect=MemoryError('bounded allocation failure')), \
                        contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                    code = campaign.main(argv + ['--out', str(root)])
                self.assertEqual(code, 1)
                self.assertNotIn('Completed ', out.getvalue())
                self.assertIn('MemoryError: bounded allocation failure', err.getvalue())
                state = json.loads((root / 'campaign_config.json').read_text())
                self.assertEqual((state['status'], state['completed_runs']), ('failed', 2))
                self.assertIn('finished', state)
                self.assertIn('MemoryError: bounded allocation failure', state['error'])
                self.assertEqual(len((root / 'runs.jsonl').read_text().splitlines()), 2)
                for policy in ('binary_simple', 'binary_complex'):
                    best = json.loads((root / 'dtw' / policy / 'best_run.json').read_text())
                    self.assertEqual(len(best['best_series']), 3)

    def test_renderer_failure_preserves_results_and_closes_new_figures_without_retry(self):
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_agg import RendererAgg
        existing = plt.figure()
        try:
            with tempfile.TemporaryDirectory() as tmp, \
                    mock.patch.object(campaign, 'load_function', FakeFunction), \
                    mock.patch.object(RendererAgg, '__init__', side_effect=MemoryError('simulated RendererAgg')) as renderer, \
                    contextlib.redirect_stdout(io.StringIO()) as out, \
                    contextlib.redirect_stderr(io.StringIO()) as err:
                code = campaign.main(['--signals', 'dtw', '--policies', 'binary_simple', 'binary_complex',
                                      '--seeds', '1', '--pop-size', '4', '--iterations', '3', '--out', tmp])
                self.assertEqual(code, 1)
                self.assertEqual(renderer.call_count, 1)
                self.assertEqual(plt.get_fignums(), [existing.number])
                self.assertNotIn('Completed ', out.getvalue())
                self.assertIn('MemoryError: simulated RendererAgg', err.getvalue())
                root = Path(tmp)
                state = json.loads((root / 'campaign_config.json').read_text())
                self.assertEqual((state['status'], state['completed_runs']), ('failed', 2))
                self.assertEqual(state['error'], 'MemoryError: simulated RendererAgg')
                self.assertEqual(len((root / 'runs.jsonl').read_text().splitlines()), 2)
                self.assertEqual(len(list(root.glob('dtw/*/best_run.json'))), 2)
                self.assertTrue((root / 'dtw' / 'binary_simple' / 'best_run' / 'result.json').exists())
        finally:
            plt.close(existing)

    def test_campaign_interruption_marks_failed(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(campaign, 'load_function', FakeFunction), \
                mock.patch.object(campaign, 'export_run', side_effect=KeyboardInterrupt), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as err:
            code = campaign.main(['--signals', 'dtw', '--policies', 'binary_simple',
                                  '--seeds', '1', '--pop-size', '4', '--iterations', '3', '--out', tmp])
            self.assertEqual(code, 1)
            self.assertIn('KeyboardInterrupt', err.getvalue())
            state = json.loads(Path(tmp, 'campaign_config.json').read_text())
            self.assertEqual((state['status'], state['completed_runs']), ('failed', 1))
            self.assertTrue(Path(tmp, 'dtw', 'binary_simple', 'best_run.json').exists())

    def test_actual_observations_and_controller_provenance(self):
        from gwo_de_dtw.controller import ProfileController
        actual = []
        original = ProfileController.observe

        def spy(controller, value):
            out = original(controller, value)
            actual.append(copy.deepcopy(out))
            return out

        with mock.patch.object(ProfileController, 'observe', spy):
            rec = self.record()
        self.assertEqual(rec['monitor'], actual)
        self.assertIn('dtw_kirito', rec['controller_config'])
        self.assertEqual(set(rec['controller_config']['exploit']), {'name', 'a', 'F', 'CR', 'G3'})

    def test_verified_replay_for_all_policies_and_signals_rejects_tampering(self):
        from gwo_de_hres2.diagnostics import temporal_series
        for policy in campaign.CAMPAIGN_POLICIES:
            for signal in (False, True):
                rec = self.record(policy, signal)
                data = temporal_series(rec)
                np.testing.assert_array_equal(data['f_star'], -np.array(rec['best_series']))
                for field in ('monitor', 'profile_series', 'parameter_series', 'intensity_series', 'changes'):
                    broken = copy.deepcopy(rec)
                    if field == 'monitor':
                        broken[field][-1]['changed'] = not broken[field][-1]['changed']
                    elif field == 'parameter_series':
                        broken[field]['a'][0] += 1
                    elif field == 'changes':
                        broken[field].append(999)
                    elif field == 'intensity_series':
                        broken[field][0] = 0.123
                    else:
                        broken[field][0] = 'incorrect'
                    with self.assertRaises(ValueError, msg=(policy, signal, field)):
                        temporal_series(broken)

    def test_rejected_export_leaves_existing_files_unchanged(self):
        from gwo_de_hres2.diagnostics import export_run
        rec = self.record()
        rec['controller_config']['exploit']['a'] += 1
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'keep.txt').write_text('unchanged')
            with self.assertRaisesRegex(ValueError, 'Profile timing mismatch'):
                export_run(rec, root)
            self.assertEqual([p.name for p in root.iterdir()], ['keep.txt'])
            self.assertEqual((root / 'keep.txt').read_text(), 'unchanged')

    def test_campaign_exports_best_layout_and_keeps_compact_records(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(campaign, 'load_function', FakeFunction), \
                contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp, 'campaign')
            self.assertEqual(campaign.main(['--signals', 'dtw', '--policies', 'binary_complex',
                                           '--seeds', '1', '--pop-size', '4', '--iterations', '3',
                                           '--out', str(root)]), 0)
            parent = root / 'dtw' / 'binary_complex'
            self.assertEqual(json.loads((parent / 'best_run.json').read_text()),
                             json.loads((parent / 'best_run' / 'result.json').read_text()))
            for name in ARTIFACTS:
                artifact = parent / 'best_run' / name
                self.assertGreater(artifact.stat().st_size, 0)
                if name.endswith('.png'):
                    self.assertTrue(artifact.read_bytes().startswith(b'\x89PNG\r\n\x1a\n'))
                if name.endswith('.pdf'):
                    self.assertTrue(artifact.read_bytes().startswith(b'%PDF'))
            self.assertFalse((parent / 'best_run' / 'dtw_delta.png').exists())
            self.assertTrue((root / 'dtw' / 'convergence_comparison.pdf').exists())
            compact = json.loads((root / 'runs.jsonl').read_text())
            self.assertNotIn('monitor', compact)
            self.assertNotIn('controller_config', compact)

    def test_ready_export_and_one_seed_outputs(self):
        from gwo_de_hres2.diagnostics import export_run
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            rec = self.record('continuous_complex')
            export_run(rec, root / 'ready')
            self.assertTrue((root / 'ready' / 'dtw_delta.png').exists())
            metadata = json.loads((root / 'ready' / 'temporal_metadata.json').read_text())
            self.assertEqual(metadata['monitor_input'], '-objective')
            self.assertTrue(metadata['verified_controller_replay'])
            with np.load(root / 'ready' / 'temporal_series.npz') as data:
                np.testing.assert_array_equal(data['f_star'], -np.array(rec['best_series']))
            func = FakeFunction()
            func.config = {}
            with mock.patch.object(run, 'load_function', return_value=func):
                self.assertEqual(run.main(['--policies', 'binary_complex', '--pop-size', '4',
                                           '--iterations', '2', '--output-dir', str(root / 'single')]), 0)
            for name in ARTIFACTS:
                self.assertTrue((root / 'single' / 'binary_complex' / name).exists(), name)
            self.assertTrue((root / 'single' / 'convergence_comparison.pdf').exists())


if __name__ == '__main__':
    unittest.main()
