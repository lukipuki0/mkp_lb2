"""S11 public-interface checks replacing the removed legacy adaptive engines."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from dtw_stagnation import StagnationConfig
from mkp_core.problem import MKPInstance
from woa_abc import solver
from woa_abc import benchmark, campaign, run
from woa_abc.plots import plot_record
from woa_abc.plots.temporal import build_series
from woa_abc.plots.comparison import convergence_comparison_figure
from woa_abc.variants import VARIANTS

ROOT = Path(__file__).resolve().parents[1]
INSTANCE = ROOT / 'instancias/mknapcb1.txt'


def flat_run(variant='v0', **kwargs):
    inst = MKPInstance(n=3, m=1, valor_optimo=6., p=np.array([1., 2., 3.]),
                       r=np.ones((1, 3)), b=np.array([10.]))
    values = dict(reaction='variant', variant=variant, population_size=2,
                  iterations=12, scout_limit=100,
                  stag_cfg=StagnationConfig(window=4, band=1, min_slope=2.,
                      plateau_max=2, patience=2, adapt_thresholds=False))
    values.update(kwargs)
    return solver.solve(inst, solver.HybridConfig(**values), seed=1)


class ModeTests(unittest.TestCase):
    def test_policy_default_and_variant_config_are_separate(self):
        self.assertEqual(solver.HybridConfig().policy, 'binary_complex')
        self.assertIsNone(solver.HybridConfig().variant)
        self.assertEqual(solver.HybridConfig(reaction='variant').variant, 'v0')
        for values in (dict(variant='v0'), dict(reaction='variant', policy='binary_complex'),
                       dict(reaction='variant', dtw_kirito=True)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                solver.HybridConfig(**values)

    def test_solver_cli_selects_variants_and_rejects_mixed_flags_before_solve(self):
        base = ['--instance', str(INSTANCE), '--seed', '1', '--population-size', '2',
                '--iterations', '2']
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = solver.main(base + ['--variant', 'v2'])
        self.assertEqual((code, err.getvalue()), (0, ''))
        cfg = json.loads(out.getvalue())['metadata']['config']
        self.assertEqual((cfg['reaction'], cfg['variant']), ('variant', 'v2'))
        self.assertNotIn('policy', cfg)
        for flags in (['--policy', 'binary_complex'], ['--dtw-kirito']):
            out, err = io.StringIO(), io.StringIO()
            with patch.object(solver, 'solve', side_effect=AssertionError('solve called')):
                with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
                    code = solver.main(base + ['--variant', 'v0'] + flags)
            self.assertEqual(code, 1)
            self.assertEqual(out.getvalue(), '')
            self.assertEqual(err.getvalue(), 'error: ' + solver.CLI_MODE_ERROR + '\n')

    def test_clock_fires_next_cycle_and_replays_every_variant(self):
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                result = flat_run(variant)
                self.assertEqual([d['cycle'] for d in result.diagnostics if d['fire']], [4, 9])
                self.assertEqual([d['cycle'] for d in result.diagnostics if d['recovery_applied']], [4, 9])
                self.assertEqual(result.profile_changes, [])
                self.assertEqual(result.history[5].schedule_index, 0)
                expected = (1., 1.5) if variant == 'v2' else (2., 2.)
                self.assertEqual((result.history[5].a, result.history[5].rho), expected)
                self.assertEqual(result.evaluations, 146)
                self.assertEqual(sum(result.source_trials), 48)
                self.assertEqual(result.best_value, 6.)
                self.assertEqual(result.phases_executed[1:5], ['woa', 'employed', 'onlookers', 'scouts'])
                rec = plot_record(result)
                self.assertEqual((rec['reaction'], rec['variant']), ('variant', variant))
                self.assertIsNone(rec['policy'])
                self.assertEqual(rec['changes'], [4, 9])
                series = build_series(rec)
                self.assertEqual(list(series['fire'].nonzero()[0]), [4, 9])
                self.assertEqual(list(series['a']), [r.a for r in result.history])
                self.assertEqual(list(series['rho']), [r.rho for r in result.history])
                if variant == 'v3':
                    self.assertEqual((result.history[-1].a, result.history[-1].rho), (0., 1.))

    def test_recovery_requires_next_pair_and_v3_r1(self):
        for budget, applied in ((62, False), (63, False), (64, True)):
            result = flat_run(iterations=6, max_evaluations=budget)
            self.assertEqual(result.diagnostics[-1]['recovery_applied'], applied)
            self.assertEqual(result.evaluations, 64 if applied else 62)
            self.assertEqual(len(result.diagnostics), 5)
            self.assertFalse(result.history[-1].completeness)
        last = flat_run(iterations=5)
        self.assertTrue(last.diagnostics[-1]['fire'])
        self.assertFalse(last.diagnostics[-1]['recovery_applied'])
        r1 = flat_run('v3', iterations=6)
        self.assertEqual((r1.history[-1].a, r1.history[-1].rho), (0., 1.))
        off = flat_run(use_stagnation=False)
        self.assertEqual(off.diagnostics, [])
        self.assertEqual([r.rho for r in off.history], [1.] * 12)
        self.assertEqual(off.history[-1].a, 0.)

    def test_exporter_variant_writes_same_required_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            request = benchmark.ExportRequest(instance=INSTANCE, reaction='variant', variant='v3',
                population_size=2, iterations=3, output_root=Path(tmp))
            with contextlib.redirect_stdout(io.StringIO()):
                outcome = benchmark.export_single_run(request)
            self.assertEqual(outcome.code, 0, outcome.message)
            for name in benchmark.EXPECTED_ARTIFACTS:
                self.assertTrue((outcome.output_dir / name).is_file(), name)
            manifest = json.loads((outcome.output_dir / 'manifest.json').read_text())
            self.assertEqual(manifest['effective_config']['variant'], 'v3')
            self.assertTrue(manifest['output_completeness']['complete'])
            self.assertEqual(manifest['input']['reference'], 24381.)

    def test_variant_group_and_comparison(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'group'
            args = ['--instance', str(INSTANCE), '--reaction', 'variant', '--pop', '2',
                    '--iterations', '6', '--window', '4', '--out', str(out)]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(run.main(args), 0)
            records = []
            for variant in VARIANTS:
                folder = out / variant
                for name in ('fitness_convergence.png', 'parameters.png', 'dtw_delta.png',
                             'temporal_diagnostics.png', 'temporal_series.npz', 'result.json'):
                    self.assertTrue((folder / name).is_file(), name)
                records.append(plot_record(json.loads((folder / 'result.json').read_text())))
            from matplotlib import pyplot as plt
            fig = convergence_comparison_figure(records)
            try:
                labels = [t.get_text() for t in fig.axes[0].get_legend().get_texts()]
                self.assertEqual(labels[:5], list(VARIANTS))
                self.assertIn('five variants', fig.axes[0].get_title())
                self.assertEqual(len({line.get_color() for line in fig.axes[0].lines[:5]}), 5)
            finally:
                plt.close(fig)
            self.assertTrue((out / 'convergence_comparison.pdf').is_file())

    def test_variant_convergence_and_delta_use_base_recovery_colors(self):
        from matplotlib import pyplot as plt
        from woa_abc.plots import plot_convergence, plot_dtw_delta
        record = plot_record(flat_run())
        for draw in (plot_convergence, plot_dtw_delta):
            with tempfile.TemporaryDirectory() as tmp:
                kept = []
                with patch.object(plt, 'close', side_effect=kept.append), contextlib.redirect_stdout(io.StringIO()):
                    draw(record, tmp)
                try:
                    legend = kept[0].axes[0].get_legend()
                    from matplotlib.colors import to_rgba
                    self.assertEqual([tuple(h.get_facecolor()) for h in legend.legend_handles[:2]],
                                     [to_rgba('#1f77b4'), to_rgba('#ff7f0e')])
                finally:
                    for fig in kept:
                        plt.close(fig)

    def test_campaign_variant_plan_has_135_selections(self):
        selections = campaign._selections(INSTANCE.parent, reaction='variant')
        self.assertEqual(len(selections), 135)
        self.assertEqual({s.variant for s in selections}, set(VARIANTS))
        self.assertEqual([sum(s.n == n for s in selections) for n in (100, 250, 500)], [45]*3)
        request = benchmark.ExportRequest(instance=INSTANCE, reaction='variant', variant='v0')
        self.assertIsNone(benchmark.build_hybrid_config(request).policy)


if __name__ == '__main__':
    unittest.main()
