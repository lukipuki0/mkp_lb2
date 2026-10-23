"""Temporal panels must describe recorded MH switching, not reference BDE states."""
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from dtw_stagnation import StagnationConfig, StagnationMonitor
from hybrid_mkp.orchestrator import PipelineResult, SwitchLog

TEMP = Path(__file__).resolve().parent / '.tmp'


def recorded(use_ddtw=False):
    config = StagnationConfig(window=3, min_slope=.5, plateau_max=1,
                              patience=1, adapt_thresholds=False, use_ddtw=use_ddtw)
    monitor = StagnationMonitor(config)
    values = [10., 11., 12., 12., 12., 12.]
    statuses = [monitor.update(value) for value in values]
    deltas = [row.get('delta', float('nan')) for row in statuses]
    switches = [SwitchLog('PSO', 'poblacional', 5., 0., .1, 3, [.5], []),
                SwitchLog('GA', 'poblacional', 12., .1, .2, 6,
                          [d for d in deltas if np.isfinite(d)], statuses)]
    result = PipelineResult(12., [1, 0], [5.] * 3 + values, [5.] * 3 + values,
                            [float('nan'), float('nan'), .5] + deltas,
                            dtw_info_global=statuses, log_switches=switches)
    metadata = {'seed': 42, 'selector': 'adaptive', 'use_ddtw': use_ddtw,
                'settings': {'monitor': asdict(config)},
                'effective_monitor': {'PSO': asdict(config), 'GA': asdict(config)}}
    return result, metadata


class TemporalTests(unittest.TestCase):
    def test_missing_histories_are_gaps_and_windows_never_cross_mh_boundaries(self):
        from adaptive_mh_selection.temporal_diagnostics import build_temporal_data
        result, metadata = recorded()
        data = build_temporal_data(result, metadata)
        np.testing.assert_equal(data['D1_vs_ramp'][:3], [np.nan] * 3)
        np.testing.assert_equal(data['fire'][:3], [np.nan] * 3)
        np.testing.assert_equal(data['delta'][:3], [np.nan, np.nan, .5])
        self.assertEqual(data['active_mh'], ['PSO'] * 3 + ['GA'] * 6)
        self.assertEqual(data['t'].tolist(), list(range(1, 10)))
        self.assertEqual({w['kind'] for w in data['windows']}, {'progress', 'plateau'})
        for window in data['windows']:
            self.assertEqual(window['solver'], 'GA')
            self.assertGreaterEqual(window['start_index'], 3)
            self.assertLess(window['end_index'], 9)
            self.assertTrue(window['verified'])
            self.assertEqual(len(window['observed']), 3)
        self.assertEqual(data['verification_failures'], 0)
        self.assertNotIn('s', data)
        self.assertNotIn('F', data)
        self.assertNotIn('CR', data)

    def test_c_waits_for_full_window_and_adaptive_thresholds_in_each_block(self):
        from adaptive_mh_selection.temporal_diagnostics import build_temporal_data, plot_temporal_diagnostics
        import matplotlib.figure
        config = StagnationConfig(window=3, min_slope=.5, adapt_thresholds=True)
        blocks, statuses, fitness = [], [], []
        for name, values in [('PSO', list(range(10, 22))), ('GA', [21.] * 12)]:
            monitor = StagnationMonitor(config)
            rows = [monitor.update(value) for value in values]
            blocks.append(SwitchLog(name, 'poblacional', 21., 0., .1, 12, [], rows))
            statuses.extend(rows)
            fitness.extend(values)
        result = PipelineResult(21., [1, 0], fitness, fitness,
                                [row.get('delta', np.nan) for row in statuses],
                                dtw_info_global=statuses, log_switches=blocks)
        metadata = {'seed': 42, 'effective_monitor': {name: asdict(config) for name in ('PSO', 'GA')}}
        before = build_temporal_data(result, metadata)
        original = matplotlib.figure.Figure.savefig

        def inspect(fig, path, *args, **kwargs):
            if Path(path).suffix == '.png':
                c = fig.axes[3]
                for label, key in [('D1 vs ramp', 'D1_vs_ramp'), ('D2 vs constant', 'D2_vs_const'),
                                   ('theta_r', 'theta_r'), ('theta_c', 'theta_c')]:
                    line = next(line for line in c.lines if line.get_label() == label)
                    np.testing.assert_equal(line.get_xdata(), [12, 24])
                    np.testing.assert_equal(line.get_ydata(), before[key][[11, 23]])
                d = fig.axes[-1]
                delta = next(line for line in d.lines if line.get_label() == 'Recorded delta')
                np.testing.assert_equal(delta.get_ydata(), before['delta'][np.isfinite(before['delta'])])
            return original(fig, path, *args, **kwargs)

        TEMP.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=TEMP) as tmp, patch.object(matplotlib.figure.Figure, 'savefig', new=inspect):
            plot_temporal_diagnostics(result, Path(tmp), metadata)
            rows = (Path(tmp) / 'temporal_diagnostics.csv').read_text(encoding='utf-8').splitlines()
            self.assertEqual(len(rows), 25)
        after = build_temporal_data(result, metadata)
        for key in ('fitness', 'D1_vs_ramp', 'D2_vs_const', 'theta_r', 'theta_c', 'delta'):
            np.testing.assert_equal(before[key], after[key])

    def test_progress_window_has_gain_closest_to_its_verified_ramp(self):
        from adaptive_mh_selection.temporal_diagnostics import build_temporal_data
        result, metadata = recorded()
        before = list(result.historial_global)
        data = build_temporal_data(result, metadata)
        progress = next(window for window in data['windows'] if window['kind'] == 'progress')
        self.assertEqual(progress['gain'], 1.)
        self.assertEqual(progress['ramp'][-1], 1.)
        self.assertEqual(progress['end_index'], 6)
        self.assertTrue(progress['verified'])
        self.assertEqual(data['verification_failures'], 0)
        self.assertEqual(result.historial_global, before)

    def test_recorded_ddtw_configuration_verifies_with_the_actual_metric(self):
        from adaptive_mh_selection.temporal_diagnostics import build_temporal_data
        result, metadata = recorded(use_ddtw=True)
        data = build_temporal_data(result, metadata)
        self.assertEqual(data['verification_failures'], 0)
        self.assertEqual({w['kind'] for w in data['windows']}, {'progress', 'plateau'})
        for window in data['windows']:
            index = window['end_index']
            self.assertAlmostEqual(window['D1'], data['D1_vs_ramp'][index])
            self.assertAlmostEqual(window['D2'], data['D2_vs_const'][index])

    def test_unverified_or_unconfigured_windows_are_not_invented(self):
        from adaptive_mh_selection.temporal_diagnostics import build_temporal_data
        result, metadata = recorded()
        self.assertEqual(build_temporal_data(result, None)['windows'], [])
        for status in result.log_switches[1].dtw_info_hist:
            if status.get('ready'):
                status['D1_vs_ramp'] += 100
        data = build_temporal_data(result, metadata)
        self.assertEqual(data['windows'], [])
        self.assertEqual(data['verification_failures'], 4)

    def test_redraw_preserves_saved_results_and_does_not_execute_solvers(self):
        from contextlib import redirect_stdout
        import io
        from adaptive_mh_selection.exports import write_json
        from adaptive_mh_selection.temporal_diagnostics import main
        result, metadata = recorded()
        metadata.update(schema_version=2, budget_unit='native_iterations', actual_iterations=9)
        TEMP.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=TEMP) as tmp:
            folder = Path(tmp)
            write_json(folder / 'native_result.json', result)
            write_json(folder / 'summary.json', metadata)
            source = {p.name: p.read_bytes() for p in folder.glob('*.json')}
            (folder / 'instantaneous_convergence.png').write_bytes(b'retired')
            (folder / 'keep.txt').write_text('keep this unrelated file')
            with patch('hybrid_mkp.orchestrator._ejecutar_mh') as solver, redirect_stdout(io.StringIO()) as output:
                self.assertEqual(main(['--run', str(folder)]), 0)
            solver.assert_not_called()
            self.assertIn('from stored data', output.getvalue())
            self.assertFalse((folder / 'instantaneous_convergence.png').exists())
            self.assertEqual((folder / 'keep.txt').read_text(), 'keep this unrelated file')
            for name, data in source.items():
                self.assertEqual((folder / name).read_bytes(), data)
            self.assertTrue((folder / 'fig_temporal_diagnostics.pdf').is_file())

    def test_c_and_d_connect_observed_samples_and_c_keeps_full_range(self):
        from adaptive_mh_selection.temporal_diagnostics import build_temporal_data, plot_temporal_diagnostics
        result, metadata = recorded()
        data = build_temporal_data(result, metadata)
        before = {key: data[key].copy() for key in ('D1_vs_ramp', 'D2_vs_const', 'delta', 'theta_delta')}
        import matplotlib.figure
        original = matplotlib.figure.Figure.savefig
        checked = []

        def inspect(fig, path, *args, **kwargs):
            if Path(path).name == 'fig_temporal_diagnostics.png':
                checked.append(path)
                c = fig.axes[3]
                d = next(ax for ax in fig.axes if any(line.get_label() == 'Recorded delta' for line in ax.lines))
                self.assertEqual(c.get_yscale(), 'linear')
                self.assertIn('linear', c.get_ylabel())
                for axis, label, key in ((c, 'D1 vs ramp', 'D1_vs_ramp'),
                                         (c, 'D2 vs constant', 'D2_vs_const'),
                                         (d, 'Recorded delta', 'delta'),
                                         (d, 'Recorded theta_delta', 'theta_delta')):
                    line = next(line for line in axis.lines if line.get_label() == label)
                    finite = np.isfinite(data[key])
                    np.testing.assert_equal(line.get_xdata(), data['t'][finite])
                    np.testing.assert_equal(line.get_ydata(), data[key][finite])
                    self.assertTrue(np.all(np.isfinite(line.get_ydata())))
                self.assertLessEqual(c.get_ylim()[0], 0)
                self.assertGreaterEqual(c.get_ylim()[1], np.nanmax(data['D1_vs_ramp']))
                fig.canvas.draw()
                renderer = fig.canvas.get_renderer()
                main = fig._suptitle.get_window_extent(renderer)
                panel = fig.axes[0]._left_title.get_window_extent(renderer)
                self.assertGreaterEqual(main.y0 - panel.y1, 12)
                for upper in fig.axes:
                    if not upper.get_xlabel():
                        continue
                    below = [lower for lower in fig.axes
                             if lower.get_position().y1 <= upper.get_position().y0 + 1e-3
                             and lower.get_position().x0 < upper.get_position().x1 - 1e-3
                             and lower.get_position().x1 > upper.get_position().x0 + 1e-3]
                    if not below:
                        continue
                    lower = max(below, key=lambda ax: ax.get_position().y1)
                    xlabel = upper.xaxis.label.get_window_extent(renderer)
                    title = lower._left_title.get_window_extent(renderer)
                    self.assertFalse(xlabel.overlaps(title))
            return original(fig, path, *args, **kwargs)

        TEMP.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=TEMP) as tmp, patch.object(matplotlib.figure.Figure, 'savefig', new=inspect):
            plot_temporal_diagnostics(result, Path(tmp), metadata)
            report = json.loads((Path(tmp) / 'temporal_diagnostics.json').read_text())
            self.assertEqual(report['rendering']['panel_c_scale'], 'linear')
            self.assertEqual(report['rendering']['connections'], 'recorded_samples_only')
        after = build_temporal_data(result, metadata)
        for key, values in before.items():
            np.testing.assert_equal(after[key], values)
        self.assertEqual(len(checked), 1)

    def test_exactly_four_vertical_panels_and_pdf_png_from_recorded_series(self):
        from adaptive_mh_selection.temporal_diagnostics import build_temporal_data, plot_temporal_diagnostics
        result, metadata = recorded()
        TEMP.mkdir(parents=True, exist_ok=True)
        import matplotlib.figure
        original = matplotlib.figure.Figure.savefig
        inspected = []

        def check(fig, path, *args, **kwargs):
            if Path(path).name == 'fig_temporal_diagnostics.png':
                inspected.append(path)
                self.assertEqual(len(fig.axes), 5)
                axes = fig.axes
                self.assertGreater(axes[0].get_position().y0, axes[-1].get_position().y0)
                self.assertIn('active MH', axes[0].get_title(loc='left'))
                self.assertIn('Native Iteration', axes[-1].get_xlabel())
                self.assertTrue(any(ax.get_ylabel() == 'Delta DTW' for ax in axes))
                d1 = next(line for ax in axes for line in ax.lines if line.get_label() == 'D1 vs ramp')
                data = build_temporal_data(result, metadata)
                np.testing.assert_equal(d1.get_ydata(), data['D1_vs_ramp'][np.isfinite(data['D1_vs_ramp'])])
                self.assertFalse(any('F=' in text.get_text() or 'CR=' in text.get_text()
                                     for ax in axes for text in ax.texts))
            return original(fig, path, *args, **kwargs)

        with tempfile.TemporaryDirectory(dir=TEMP) as tmp, patch.object(matplotlib.figure.Figure, 'savefig', new=check):
            paths = plot_temporal_diagnostics(result, Path(tmp), metadata)
            self.assertEqual({Path(p).name for p in paths}, {'fig_temporal_diagnostics.png', 'fig_temporal_diagnostics.pdf'})
            for path in paths:
                self.assertGreater(Path(path).stat().st_size, 1000)
            report = json.loads((Path(tmp) / 'temporal_diagnostics.json').read_text())
            self.assertEqual(report['iterations'], 9)
            self.assertEqual(report['missing_detail_iterations'], 3)
            self.assertEqual(len(report['windows']), 2)
        self.assertEqual(len(inspected), 1)

    def test_reference_style_layout_keeps_recorded_values(self):
        """Keep recorded series and reference windows, without a lower MH strip."""
        from matplotlib.colors import to_rgba
        import matplotlib
        import matplotlib.figure
        from hybrid_mkp.orchestrator import COLORES_MH
        from adaptive_mh_selection.temporal_diagnostics import build_temporal_data, plot_temporal_diagnostics

        result, metadata = recorded()
        data = build_temporal_data(result, metadata)
        before = {key: np.array(data[key], copy=True) for key in
                  ('fitness', 'D1_vs_ramp', 'D2_vs_const', 'delta', 'theta_r', 'theta_c', 'theta_delta')}
        family_before = list(matplotlib.rcParams['font.family'])
        original = matplotlib.figure.Figure.savefig
        inspected = []

        def rgb(color):
            return tuple(round(channel, 4) for channel in to_rgba(color)[:3])

        def inspect(fig, path, *args, **kwargs):
            if Path(path).name != 'fig_temporal_diagnostics.png':
                return original(fig, path, *args, **kwargs)
            inspected.append(path)
            width, height = fig.get_size_inches()
            self.assertAlmostEqual(width, 7.4, places=2)
            self.assertAlmostEqual(height, 8.6, places=2)
            self.assertEqual(len(fig.axes), 5)
            self.assertFalse(any(ax.get_ylabel() == 'MH' for ax in fig.axes))
            rising = next(ax for ax in fig.axes if 'Rising' in ax.get_title(loc='left'))
            flat = next(ax for ax in fig.axes if 'Flat' in ax.get_title(loc='left'))
            self.assertAlmostEqual(rising.get_position().y0, flat.get_position().y0, places=2)
            self.assertLess(rising.get_position().x0, flat.get_position().x0)
            progress = next(window for window in data['windows'] if window['kind'] == 'progress')
            plateau = next(window for window in data['windows'] if window['kind'] == 'plateau')
            expected = np.arange(progress['start_index'] + 1, progress['end_index'] + 2)
            observed = next(line for line in rising.lines if rgb(line.get_color()) == rgb('#1a1a1a'))
            ramp = next(line for line in rising.lines if rgb(line.get_color()) == rgb('#0072B2'))
            constant = next(line for line in rising.lines if rgb(line.get_color()) == rgb('#D55E00'))
            np.testing.assert_equal(observed.get_xdata(), expected)
            np.testing.assert_allclose(observed.get_ydata(), progress['observed'])
            np.testing.assert_allclose(ramp.get_ydata(), progress['ramp'])
            np.testing.assert_allclose(constant.get_ydata(), progress['constant'])
            flat_x = next(line for line in flat.lines if rgb(line.get_color()) == rgb('#1a1a1a')).get_xdata()
            np.testing.assert_equal(flat_x, np.arange(plateau['start_index'] + 1, plateau['end_index'] + 2))
            bottom = fig.axes[-1]
            self.assertEqual(bottom.get_ylabel(), 'Delta DTW')
            self.assertIn('Native Iteration', bottom.get_xlabel())
            delta_line = next(line for line in bottom.lines if line.get_label() == 'Recorded delta')
            np.testing.assert_equal(delta_line.get_ydata(), data['delta'][np.isfinite(data['delta'])])
            labels = [tick.get_text() for tick in bottom.get_yticklabels()]
            self.assertLess(bottom.get_position().y1, rising.get_position().y0)
            banned = ' '.join(labels + [ax.get_title(loc='left') for ax in fig.axes] +
                              [text.get_text() for ax in fig.axes for text in list(ax.texts) + list(ax.get_yticklabels())])
            for token in ('F=', 'CR=', 'Exploration', 'Exploitation', 'explore', 'exploit'):
                self.assertNotIn(token, banned)
            a = fig.axes[0]
            painted = {rgb(patch.get_facecolor()) for patch in a.patches if patch.get_facecolor()[3] > 0}
            self.assertIn(rgb(COLORES_MH['PSO']), painted)
            self.assertIn(rgb(COLORES_MH['GA']), painted)
            self.assertNotIn(rgb('#F4A261'), painted)
            self.assertNotIn(rgb('#8ECAE6'), painted)
            marked = {rgb(patch.get_edgecolor()) for ax in fig.axes if ax.get_ylabel() == 'Distance (linear)'
                      or 'Delta DTW' in ax.get_ylabel() or ax is a
                      for patch in ax.patches if patch.get_edgecolor()[3] > 0}
            self.assertIn(rgb('#0072B2'), marked)
            self.assertIn(rgb('#D55E00'), marked)
            distance = next(ax for ax in fig.axes if ax.get_ylabel() == 'Distance (linear)')
            self.assertEqual(distance.get_yscale(), 'linear')
            finite = np.isfinite(data['D1_vs_ramp'])
            d1 = next(line for line in distance.lines if line.get_label() == 'D1 vs ramp')
            np.testing.assert_equal(d1.get_ydata(), data['D1_vs_ramp'][finite])
            self.assertGreaterEqual(distance.get_ylim()[1], np.nanmax(data['D1_vs_ramp']))
            self.assertLessEqual(distance.get_ylim()[0], 0)
            disclosure = ' '.join(text.get_text() for text in fig.texts).lower()
            self.assertIn('linear', disclosure)
            self.assertIn('not measurement', disclosure)
            self.assertIn('serif', a._left_title.get_fontfamily())
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            self.assertFalse(fig.texts[-1].get_window_extent(renderer).overlaps(
                bottom.xaxis.label.get_window_extent(renderer)))
            self.assertGreaterEqual(fig._suptitle.get_window_extent(renderer).y0
                                    - a._left_title.get_window_extent(renderer).y1, 12)
            return original(fig, path, *args, **kwargs)

        TEMP.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=TEMP) as tmp, patch.object(matplotlib.figure.Figure, 'savefig', new=inspect):
            plot_temporal_diagnostics(result, Path(tmp), metadata)
            report = json.loads((Path(tmp) / 'temporal_diagnostics.json').read_text(encoding='utf-8'))
            self.assertEqual(report['rendering']['panel_c_scale'], 'linear')
            self.assertIsNone(report['rendering']['panel_c_linthresh'])
            self.assertEqual(report['rendering']['connections'], 'recorded_samples_only')
            self.assertEqual(report['rendering']['d_strip'], 'none')
            self.assertAlmostEqual(report['windows'][0]['D1'], data['windows'][0]['D1'])
            rows = (Path(tmp) / 'temporal_diagnostics.csv').read_text(encoding='utf-8').splitlines()
            self.assertEqual(len(rows), 10)
            self.assertIn('GA', rows[4])
            self.assertIn(f"{before['fitness'][3]:.1f}", rows[4])
        after = build_temporal_data(result, metadata)
        for key, values in before.items():
            np.testing.assert_equal(after[key], values)
        self.assertEqual(list(matplotlib.rcParams['font.family']), family_before)
        self.assertEqual(len(inspected), 1)
        with tempfile.TemporaryDirectory(dir=TEMP) as empty_dir:
            notes = []

            def capture(fig, path, *args, **kwargs):
                if Path(path).name == 'fig_temporal_diagnostics.png':
                    notes.extend(text.get_text() for ax in fig.axes for text in ax.texts)
                return original(fig, path, *args, **kwargs)

            with patch.object(matplotlib.figure.Figure, 'savefig', new=capture):
                paths = plot_temporal_diagnostics(result, Path(empty_dir), None)
            self.assertTrue(any('No verified window' in note for note in notes))
            self.assertEqual(len(paths), 2)
            self.assertTrue(all(Path(path).stat().st_size > 1000 for path in paths))
        bare, bare_meta = recorded()
        for switch in bare.log_switches:
            switch.dtw_info_hist = []
            switch.dtw_deltas = []
        bare.dtw_deltas_global = [float('nan')] * len(bare.historial_global)
        with tempfile.TemporaryDirectory(dir=TEMP) as bare_dir:
            bare_paths = plot_temporal_diagnostics(bare, Path(bare_dir), bare_meta)
            self.assertEqual(len(bare_paths), 2)


if __name__ == '__main__':
    unittest.main()
