"""Temporal diagnostics through the public controller and plot interfaces."""
import json
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest

import numpy as np
from dtw_stagnation import StagnationConfig
from gwo_de_dtw.controller import ProfileController
from gwo_de_dtw.plots import save_all_plots
from gwo_de_dtw.plots.temporal import build_series, plot_temporal


def record(values, use_ddtw=False, **kwargs):
    cfg = StagnationConfig(window=4, min_slope=2, plateau_max=2,
                           patience=1, use_ddtw=use_ddtw, **kwargs)
    controller = ProfileController(cfg)
    profiles, outputs, params = [], [], {k: [] for k in ('a', 'F', 'CR')}
    for value in values:
        profiles.append(controller.profile.name)
        for key in params:
            params[key].append(getattr(controller.profile, key))
        outputs.append(controller.observe(value))
    return dict(best_series=values, profile_series=profiles, monitor=outputs,
                parameter_series=params, monitor_config=asdict(cfg),
                use_ddtw=use_ddtw, seed=1, window=4)


class TemporalTests(unittest.TestCase):
    def test_distances_and_fresh_windows_for_both_signals(self):
        for signal in (False, True):
            res = record([100.] * 12 + list(range(101, 125)), signal)
            data = build_series(res)
            self.assertTrue(any(o['changed'] for o in res['monitor']))
            for i, out in enumerate(res['monitor']):
                self.assertEqual(data['ready'][i], out['ready'])
                self.assertEqual(data['s'][i], int(out['profile'] == 'explore'))
                self.assertEqual(data['mode_used'][i], int(res['profile_series'][i] == 'explore'))
                if out['ready']:
                    self.assertEqual(data['D_R'][i], out['D1_vs_ramp'])
                    self.assertEqual(data['D_C'][i], out['D2_vs_const'])
                else:
                    self.assertTrue(np.isnan(data['X'][i]).all())
                if out['changed'] and i + 1 < len(res['monitor']):
                    self.assertFalse(data['ready'][i + 1])

    def test_auto_slope_and_tolerance_use_monitor_window(self):
        res = record([100., 100.1, 100.2, 100.3, 103., 104.], improvement_tol=.01)
        res['monitor_config']['min_slope'] = 0
        cfg = StagnationConfig(**res['monitor_config'])
        c = ProfileController(cfg)
        res['monitor'] = [c.observe(x) for x in res['best_series']]
        data = build_series(res)
        np.testing.assert_array_equal(data['X'][3], [100.] * 4)
        self.assertAlmostEqual(data['R'][3, -1] - 100, .0075)

    def test_mismatched_record_is_rejected_before_writing(self):
        res = record([100.] * 8)
        res['monitor'][3]['delta'] += 1
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, 'generation 4'):
                plot_temporal(res, tmp)
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_exports_short_flat_and_rising_runs(self):
        for values in ([100., 100.], [100.] * 16, list(range(100, 120))):
            with tempfile.TemporaryDirectory() as tmp:
                res = record(values)
                path = plot_temporal(res, tmp)
                self.assertEqual(Path(path).read_bytes()[:8], b'\x89PNG\r\n\x1a\n')
                self.assertTrue((Path(tmp) / 'temporal_diagnostics.pdf').read_bytes().startswith(b'%PDF'))
                with np.load(Path(tmp) / 'temporal_series.npz') as data:
                    self.assertEqual(data['R'].shape, (len(values), 4))
                    self.assertNotIn('G3', data.files)
                rows = (Path(tmp) / 'temporal_series.csv').read_text().splitlines()
                self.assertEqual(len(rows), len(values) + 1)
                meta = json.loads((Path(tmp) / 'temporal_metadata.json').read_text())
                self.assertEqual(meta['monitor_config'], res['monitor_config'])

    def test_connected_diagnostics_and_reference_window_colors(self):
        from unittest.mock import patch
        from matplotlib.figure import Figure
        res = record([100.] * 12 + list(range(101, 125)))
        figures = []
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(Figure, 'savefig', lambda fig, *a, **k: figures.append(fig)):
                plot_temporal(res, tmp)
            fig = figures[0]
            for ax in fig.axes[1:3]:
                self.assertEqual([line.get_color() for line in ax.lines],
                                 ['#1a1a1a', '#0072B2', '#D55E00'])
                self.assertEqual(ax.lines[2].get_linestyle(), '--')
            data = build_series(res)
            self.assertTrue(np.isnan(data['D_R']).any())
            for ax, keys in ((fig.axes[3], ('D_R', 'theta_R', 'D_C', 'theta_C')),
                             (fig.axes[4], ('delta', 'theta_delta'))):
                for line, key in zip(ax.lines, keys):
                    mask = np.isfinite(data[key]) & data['adaptive_threshold'].astype(bool)
                    np.testing.assert_array_equal(line.get_xdata(), data['t'][mask])
                    np.testing.assert_array_equal(line.get_ydata(), data[key][mask])

    def test_rising_window_matches_ramp_rise_instead_of_initial_jump(self):
        from gwo_de_dtw.plots.temporal import select_windows
        data = dict(ready=np.ones(4, dtype=bool),
                    X=np.array([[0., 1000.], [0., 8.], [0., 11.], [0., 0.]]),
                    R=np.array([[0., 10.]] * 4), mode_used=np.array([0, 0, 0, 1]))
        self.assertEqual(select_windows(data), (2, 3))
        data['mode_used'][:] = 1
        self.assertEqual(select_windows(data), (2, 3))

    def test_automatic_plot_registration(self):
        from gwo_de_dtw.plots import PLOTS
        self.assertIn(plot_temporal, PLOTS)


if __name__ == '__main__':
    unittest.main()
