"""HRES2-only adapters: minimized outcomes, verified maximizing monitor replay.

No historical observations are synthesized. Controller replay verifies the
profile/parameters used before each observation as well as the decision after it.
The reference temporal builder additionally verifies filtered windows/distances.
"""
import gc
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from dtw_stagnation import StagnationConfig
from gwo_de_dtw.controller import ProfileController
from gwo_de_dtw.hybrid import Profile
from gwo_de_dtw.plots.dtw_delta import plot_dtw_delta
from gwo_de_dtw.plots.parameters import plot_parameters
from gwo_de_dtw.plots.temporal import build_series, select_windows, SCALARS

OBJECTIVE_LABEL = 'Best objective (LCOE if feasible, penalty otherwise; minimize)'


def _same(actual, expected):
    if actual is None or expected is None:
        return actual is expected
    if isinstance(actual, str) or isinstance(expected, str):
        return actual == expected
    return bool(np.isclose(actual, expected, rtol=0, atol=1e-8, equal_nan=True))


def temporal_series(rec):
    """Reject incomplete or inconsistent provenance before any export is written."""
    settings = dict(rec['controller_config'])
    settings['exploit'] = Profile(**settings['exploit'])
    settings['explore'] = Profile(**settings['explore'])
    controller = ProfileController(StagnationConfig(**rec['monitor_config']),
                                   policy=rec['policy'], **settings)
    count = len(rec['best_series'])
    arrays = [rec['monitor'], rec['profile_series'], rec['intensity_series'],
              *[rec['parameter_series'][key] for key in ('a', 'F', 'CR')]]
    if any(len(values) != count for values in arrays):
        raise ValueError('Incomplete temporal history')
    changes = []
    for i, objective in enumerate(rec['best_series']):
        used = controller.profile
        pairs = [(used.name, rec['profile_series'][i]),
                 (controller.intensity, rec['intensity_series'][i])]
        pairs += [(getattr(used, key), rec['parameter_series'][key][i])
                  for key in ('a', 'F', 'CR')]
        if not all(_same(a, b) for a, b in pairs):
            raise ValueError(f'Profile timing mismatch at generation {i + 1}')
        observed = controller.observe(-objective)
        saved = rec['monitor'][i]
        if observed.keys() != saved.keys() or not all(
                _same(value, saved[key]) for key, value in observed.items()):
            raise ValueError(f'Controller observation mismatch at generation {i + 1}')
        if observed['changed']:
            changes.append(i)
    if changes != rec['changes']:
        raise ValueError('Controller changes mismatch')
    adapted = dict(rec, best_series=[-v for v in rec['best_series']],
                   dtw_kirito=settings['dtw_kirito'])
    return build_series(adapted)


def plot_outcomes(records, path, title):
    """Release comparison renderers before returning, including on failure."""
    _closed_plot(_plot_outcomes, records, path, title)


def export_run(rec, output_dir):
    """Release each winner's figures without retaining native pixel buffers."""
    _closed_plot(_export_run, rec, output_dir)


def _plot_outcomes(records, path, title):
    """Raw minimized objectives; no invented optimum or feasibility assumption."""
    # Matplotlib figures contain reference cycles; release closed renderers
    # between exports so long campaigns do not retain their pixel buffers.
    gc.collect()
    fig, ax = plt.subplots(figsize=(10, 6))
    try:
        for rec in records:
            ax.plot(range(1, len(rec['best_series']) + 1), rec['best_series'],
                    label=f"{rec['policy']} (seed {rec['seed']})")
        ax.set(xlabel='Generation', ylabel=OBJECTIVE_LABEL, title=title)
        ax.grid(alpha=.3)
        ax.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        if Path(path).name == 'convergence_comparison.png':
            fig.savefig(Path(path).with_suffix('.pdf'))
    finally:
        plt.close(fig)


def _closed_plot(plot, *args):
    """Reference plotters may raise before close; own only their new figures."""
    existing = set(plt.get_fignums())
    try:
        # Layout and save must use the same DPI. Otherwise tight_layout creates
        # a 100-DPI renderer followed by a second 150-DPI pixel buffer at save.
        with plt.rc_context({'figure.dpi': 150}):
            return plot(*args)
    finally:
        for number in set(plt.get_fignums()) - existing:
            fig = plt.figure(number)
            fig.clear()
            plt.close(fig)
        gc.collect()


def _export_run(rec, output_dir):
    """Write the reference individual artifact set from a newly captured record."""
    data = temporal_series(rec)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'result.json').write_text(json.dumps(rec, indent=2), encoding='utf-8')
    _closed_plot(_plot_outcomes, [rec], output / 'fitness_convergence.png', 'HRES2 objective convergence')
    _closed_plot(plot_parameters, rec, output)
    _closed_plot(plot_dtw_delta, rec, output)
    np.savez_compressed(output / 'temporal_series.npz', **data)
    np.savetxt(output / 'temporal_series.csv', np.column_stack([data[k] for k in SCALARS]),
               delimiter=',', header=','.join(SCALARS), comments='')
    metadata = {key: rec[key] for key in ('policy', 'signal', 'seed', 'monitor_config',
                                         'controller_config', 'objective', 'evaluations')}
    metadata.update(monitor_input='-objective', verified_controller_replay=True,
                    notes={'f_star': 'Negated best objective (maximize), not physical LCOE.',
                           's': 'Decision after t, applied at t+1: 0 exploit, 1 explore.',
                           'mode_used': 'Profile used at t: 0 exploit, 1 explore.',
                           'parameters': 'a/F/CR and intensity used at t, before observation.',
                           'X_R_C': 'Verified filtered monitor window and baselines; NaN when unready.'})
    (output / 'temporal_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    gc.collect()
    fig = plt.figure(figsize=(12, 14), layout='constrained')
    grid = fig.add_gridspec(5, 1)
    first = fig.add_subplot(grid[0])
    windows = grid[1].subgridspec(1, 2)
    window_axes = [fig.add_subplot(windows[0]), fig.add_subplot(windows[1])]
    axes = [first] + [fig.add_subplot(grid[i], sharex=first) for i in (2, 3, 4)]
    try:
        t = data['t']
        axes[0].step(t, data['f_star'], where='mid')
        axes[0].set_ylabel('-objective (maximize)')
        width = rec['monitor_config']['window']
        for ax, idx, title in zip(window_axes, select_windows(data), ('Rising window', 'Flat window')):
            ax.set(title=title, xlabel='Generation', ylabel='Change from window start')
            if idx is None:
                ax.text(.5, .5, 'No eligible window', transform=ax.transAxes, ha='center')
                continue
            xs = np.arange(idx - width + 2, idx + 2)
            for key, label in (('X', 'Observed -objective'), ('R', 'Ramp'), ('C', 'Constant')):
                ax.plot(xs, data[key][idx] - data[key][idx, 0], label=label)
            ax.legend()
        for key in ('D_R', 'D_C', 'theta_R', 'theta_C'):
            axes[1].plot(t, data[key], label=key)
        axes[1].set_ylabel(rec['signal'].upper() + ' distance')
        for key in ('delta', 'theta_delta'):
            axes[2].plot(t, data[key], label=key)
        axes[2].set_ylabel('D_R - D_C')
        axes[3].step(t, data['mode_used'], where='mid', label='Profile used at t')
        axes[3].step(t, data['s'], where='post', label='Decision after t (used at t+1)')
        axes[3].plot(t, data['intensity'], label='Intensity used at t')
        axes[3].set(xlabel='Generation', yticks=[0, 1], yticklabels=['Exploit', 'Explore'])
        for ax in axes:
            ax.grid(alpha=.3)
        for ax in axes[1:]:
            ax.legend()
        fig.suptitle(f"HRES2 verified temporal replay | {rec['signal'].upper()} | seed {rec['seed']}")
        fig.savefig(output / 'temporal_diagnostics.png', dpi=150)
        fig.savefig(output / 'temporal_diagnostics.pdf')
    finally:
        plt.close(fig)
