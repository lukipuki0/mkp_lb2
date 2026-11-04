"""S9 diagnostics through saved records and both public entry points."""
import contextlib
import copy
import io
import json
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
