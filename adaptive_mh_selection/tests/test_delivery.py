"""Public iteration CLI and actual native figure/export contracts."""
import contextlib
import csv
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from hybrid_mkp.orchestrator import PipelineResult, SwitchLog, COLORES_MH

ROOT = Path(__file__).resolve().parents[2]
TEMP = ROOT / 'adaptive_mh_selection/tests/.tmp'


def temporary():
    TEMP.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(dir=TEMP)


def invoke(*args):
    return subprocess.run([sys.executable, '-B', '-m', 'adaptive_mh_selection.run', *map(str, args)],
                          cwd=ROOT, capture_output=True, text=True, timeout=90)


def native_result():
    return PipelineResult(12., [1, 0], [10., 10., 12., 12.], [9., 10., 11., 12.],
                          [float('nan'), .2, float('nan'), .3],
                          dtw_info_global=[{'ready': False}, {'ready': True, 'delta': .3}],
                          log_switches=[SwitchLog('PSO', 'poblacional', 10., 0., .1, 2, [.2], []),
                                        SwitchLog('SA', 'trayectoria', 12., .1, .2, 2, [.3],
                                                  [{'ready': False}, {'ready': True, 'delta': .3}])])


class DeliveryTests(unittest.TestCase):
    def test_cli_help_defaults_dryrun_and_retired_flags(self):
        help_result = invoke('--help')
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn('--iterations', help_result.stdout)
        self.assertNotIn('--max-evaluations', help_result.stdout)
        with temporary() as tmp:
            output = Path(tmp) / 'campaign'
            result = invoke('--dry-run', '--output', output)
            self.assertEqual(result.returncode, 0, result.stderr)
            protocol = json.loads(result.stdout)
            self.assertEqual(protocol['index'], 0)
            self.assertTrue(protocol['instance_file'].replace('\\', '/').endswith('instancias/mknapcb1.txt'))
            self.assertEqual(protocol['instance_dimensions'], {'n': 100, 'm': 5})
            from adaptive_mh_selection.config import RunConfig, default_monitor
            study, source = RunConfig(), default_monitor()
            config, = protocol['runs']
            self.assertEqual((config['seed'], config['selector'], config['max_iters'], config['max_epoch_iters']),
                             (study.seed, study.selector, study.max_iters, None))
            self.assertEqual((config['alpha'], config['epsilon'], config['pop_injection_mode']),
                             (study.alpha, study.epsilon, study.pop_injection_mode))
            self.assertEqual(config['use_ddtw'], study.use_ddtw)
            self.assertTrue(config['monitor']['use_ddtw'])
            for name in ('window', 'band', 'min_slope', 'plateau_max', 'patience',
                         'adapt_thresholds', 'p_low', 'p_high', 'improvement_tol'):
                self.assertEqual(config['monitor'][name], getattr(source, name), name)
            self.assertFalse(output.exists())
            alias = invoke('--dry-run', '--max-iters', 4, '--runs', 2, '--seed', 7,
                           '--selectors', 'uniform', '--modes', 'ddtw')
            self.assertEqual(alias.returncode, 0, alias.stderr)
            configs = json.loads(alias.stdout)['runs']
            self.assertEqual([c['seed'] for c in configs], [7, 8])
            self.assertTrue(all(c['max_iters'] == 4 and c['monitor']['use_ddtw'] for c in configs))
            self.assertTrue(all(c['monitor']['band'] == source.band and c['monitor']['window'] == source.window
                                for c in configs))
            overridden = invoke('--dry-run', '--modes', 'dtw', '--window', '12', '--min-slope', '0.25',
                                '--plateau-max', '9', '--patience', '4', '--output', output / 'dtw')
            self.assertEqual(overridden.returncode, 0, overridden.stderr)
            chosen = json.loads(overridden.stdout)['runs'][0]
            self.assertFalse(chosen['use_ddtw'])
            self.assertFalse(chosen['monitor']['use_ddtw'])
            self.assertEqual(chosen['monitor']['window'], 12)
            self.assertEqual(chosen['monitor']['min_slope'], 0.25)
            self.assertEqual(chosen['monitor']['plateau_max'], 9)
            self.assertEqual(chosen['monitor']['patience'], 4)
            self.assertEqual(chosen['monitor']['band'], source.band)
            self.assertEqual((chosen['monitor']['p_low'], chosen['monitor']['p_high']), (source.p_low, source.p_high))
            self.assertEqual(chosen['monitor']['adapt_thresholds'], source.adapt_thresholds)
            self.assertEqual(chosen['monitor']['improvement_tol'], source.improvement_tol)
            self.assertEqual(chosen['pop_injection_mode'], 'mixed')
            self.assertFalse((output / 'dtw').exists())
            for flag, value in [('--max-evaluations', 10), ('--max-block-evaluations', 2),
                                ('--evaluation-reference', 100), ('--iterations', 0), ('--index', -1)]:
                rejected = invoke(flag, value, '--output', output)
                self.assertEqual(rejected.returncode, 2, rejected.stdout + rejected.stderr)
                self.assertFalse(output.exists())

    def test_remaining_native_renderers_and_temporal_diagnostics(self):
        import hybrid_mkp.plots as native_plots
        from adaptive_mh_selection.plotting import plot_run
        functions = ['grafico_convergencia', 'grafico_dtw_delta', 'grafico_switches',
                     'grafico_solo_instantaneo']
        result = native_result()
        with temporary() as tmp, contextlib.ExitStack() as stack:
            calls = {name: stack.enter_context(patch.object(native_plots, name, wraps=getattr(native_plots, name)))
                     for name in functions}
            # Inspect global offsets generated by the *native* delta renderer.
            from matplotlib.axes import Axes
            original = Axes.plot
            lines = []
            def capture(ax, *args, **kwargs):
                if kwargs.get('linewidth') == 2.5 and kwargs.get('alpha') == .85:
                    lines.append(list(args[0]))
                return original(ax, *args, **kwargs)
            obsolete = Path(tmp) / 'instantaneous_convergence.png'
            obsolete.write_bytes(b'obsolete plot')
            with patch.object(Axes, 'plot', new=capture), patch.object(native_plots, 'grafico_instantaneo') as retired:
                paths = plot_run(result, Path(tmp))
            retired.assert_not_called()
            self.assertFalse(obsolete.exists())
            expected = {'fitness_convergence.png', 'dtw_delta.png', 'switches_gantt.png',
                        'instantaneous_fitness_only.png', 'fig_temporal_diagnostics.png',
                        'fig_temporal_diagnostics.pdf'}
            self.assertEqual({Path(p).name for p in paths}, expected)
            self.assertEqual(lines, [[1], [1, 3]])
            for name, call in calls.items():
                call.assert_called_once()
                self.assertIs(call.call_args.kwargs['log_switches'], result.log_switches)
                self.assertIs(call.call_args.kwargs['colores_mh'], COLORES_MH)
                for key in ('historial_global', 'historial_inst_global', 'dtw_deltas_global'):
                    if key in call.call_args.kwargs:
                        self.assertIs(call.call_args.kwargs[key], getattr(result, key))
            for path in paths:
                self.assertGreater(Path(path).stat().st_size, 1000)

    def test_native_missing_deltas_skip_without_fabrication(self):
        from adaptive_mh_selection.plotting import plot_run
        result = native_result()
        result.dtw_deltas_global = [float('nan')] * 4
        with temporary() as tmp:
            paths = plot_run(result, Path(tmp))
            self.assertEqual(len(paths), 5)
            self.assertFalse((Path(tmp) / 'dtw_delta.png').exists())
            self.assertFalse((Path(tmp) / 'instantaneous_convergence.png').exists())

    def test_native_export_missing_info_does_not_shift_later_blocks(self):
        from adaptive_mh_selection.exports import write_histories, write_json
        with temporary() as tmp:
            root = Path(tmp)
            write_histories(native_result(), root)
            with (root / 'historial_dtw.csv').open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([int(r['iteracion']) for r in rows], [1, 2, 3, 4])
            self.assertEqual([r['mh'] for r in rows], ['PSO', 'PSO', 'SA', 'SA'])
            self.assertEqual(rows[0]['dtw_ready'], '')
            self.assertEqual(rows[2]['dtw_ready'], 'False')
            self.assertEqual(rows[1]['dtw_delta'], '0.2')
            self.assertEqual(rows[3]['dtw_delta'], '0.3')
            self.assertIn('no disponible', rows[0]['estado_dtw'])
            self.assertTrue((root / 'historial_dtw_detalle.txt').is_file())
            write_json(root / 'native.json', {'nan': float('nan'), 'inf': float('inf'), 'missing': None})
            self.assertEqual(json.loads((root / 'native.json').read_text()), dict.fromkeys(['nan', 'inf', 'missing']))

    def test_tiny_batch_uses_native_files_and_rejects_existing_output(self):
        with temporary() as tmp:
            root = Path(tmp)
            fixture = root / 'tiny.txt'
            fixture.write_text('1\n8 2 0\n10 9 8 7 6 5 4 3\n3 4 2 3 1 2 1 2\n2 1 3 2 4 1 2 1\n10 9\n', encoding='utf-8')
            output = root / 'campaign'
            result = invoke('--instance', fixture, '--iterations', 4, '--max-epoch-iters', 2,
                            '--runs', 2, '--seed', 42, '--modes', 'dtw', '--output', output, '--selection-trace')
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            condition = output / 'adaptive_dtw'
            self.assertTrue((condition / 'runs_resultados.csv').is_file())
            self.assertTrue((condition / 'resumen_pipeline.txt').is_file())
            self.assertTrue((condition / 'boxplot_runs.png').is_file())
            summaries = sorted(condition.glob('run_*/summary.json'))
            self.assertEqual(len(summaries), 2)
            for path in summaries:
                data = json.loads(path.read_text())
                self.assertEqual((data['requested_iterations'], data['actual_iterations'], data['iteration_overshoot']), (4, 4, 0))
                self.assertEqual(data['n_blocks'], data['transitions'] + 1)
                self.assertNotIn('evaluations', data)
                self.assertTrue((path.parent / 'selection_trace.csv').is_file())
                self.assertEqual(len(data['best_solution']), 8)
            best, = condition.glob('best_run_*')
            for name in ('resumen_pipeline_best_run.txt', 'historial_dtw.csv', 'historial_dtw_detalle.txt',
                         'fitness_convergence.png', 'switches_gantt.png',
                         'instantaneous_fitness_only.png', 'fig_temporal_diagnostics.png',
                         'fig_temporal_diagnostics.pdf'):
                self.assertTrue((best / name).is_file(), name)
            self.assertFalse((best / 'instantaneous_convergence.png').exists())
            # The configured window is larger than these capped blocks, so no delta plot is ready.
            self.assertFalse((best / 'dtw_delta.png').exists())
            before = {str(p.relative_to(output)): p.read_bytes() for p in output.rglob('*') if p.is_file()}
            rejected = invoke('--instance', fixture, '--output', output)
            self.assertEqual(rejected.returncode, 2)
            self.assertIn('already exists', rejected.stderr)
            self.assertEqual(before, {str(p.relative_to(output)): p.read_bytes() for p in output.rglob('*') if p.is_file()})


if __name__ == '__main__':
    unittest.main()
