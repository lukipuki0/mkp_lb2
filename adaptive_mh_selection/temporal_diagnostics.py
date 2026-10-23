"""Four recorded-data diagnostic rows, drawn in the compact reference layout."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from dtw_stagnation import ramp_baseline, constant_baseline, dtw_distance, ddtw_distance
from hybrid_mkp.orchestrator import COLORES_MH, PipelineResult, SwitchLog
from .exports import write_csv, write_json

METRICS = ('D1_vs_ramp', 'D2_vs_const', 'theta_r', 'theta_c', 'delta', 'theta_delta', 'ready', 'fire')


def _number(value):
    return float(value) if value is not None and np.isfinite(value) else float('nan')


def _plot_observed(axis, t, values, **style):
    """Connect existing samples for display; do not impute or extrapolate data."""
    observed = np.isfinite(values)
    return axis.plot(t[observed], values[observed], **style)[0]


def build_temporal_data(result: PipelineResult, metadata: dict | None = None) -> dict:
    """Align recorded statuses by native blocks and verify two reference windows.

    Only delta may fall back to the separately recorded native delta history.
    No missing distances, thresholds, ready/fire flags or controller states are
    inferred. Reference windows are derived from recorded settings and accepted
    only if recomputed D1/D2 match the actual recorded distances.
    """
    count = len(result.historial_global)
    if count == 0 or sum(s.n_iters for s in result.log_switches) != count:
        raise ValueError('Nonempty native histories and aligned switch lengths are required')
    data = {key: np.full(count, np.nan) for key in METRICS}
    data.update(t=np.arange(1, count + 1), fitness=np.asarray(result.historial_global, dtype=float),
                active_mh=[], pool=[], blocks=[], windows=[], notes=[], verification_failures=0,
                panel_c_visible=np.zeros(count, dtype=bool))
    for i, value in enumerate(result.dtw_deltas_global[:count]):
        data['delta'][i] = _number(value)
    candidates, offset, known = [], 0, 0
    metadata = metadata or {}
    fallback = metadata.get('settings', {}).get('monitor')
    for block_index, switch in enumerate(result.log_switches):
        start, end = offset, offset + switch.n_iters
        data['active_mh'].extend([switch.mh_nombre] * switch.n_iters)
        data['pool'].extend([switch.tipo] * switch.n_iters)
        data['blocks'].append(dict(start_index=start, end_index=end - 1,
                                   solver=switch.mh_nombre, pool=switch.tipo))
        config = metadata.get('effective_monitor', {}).get(switch.mh_nombre, fallback)
        statuses = switch.dtw_info_hist or []
        ready_samples = 0
        for local, status in enumerate(statuses[:switch.n_iters]):
            if not status:
                continue
            known += 1
            index = start + local
            for key in METRICS:
                if status.get(key) is not None:
                    data[key][index] = _number(status[key])
            if status.get('ready') and config:
                ready_samples += 1
                # Native n counts monitor updates, including unrecorded detail rows.
                n, window = status.get('n'), config.get('window')
                samples = n - window + 1 if n is not None and window is not None else ready_samples
                data['panel_c_visible'][index] = not config.get('adapt_thresholds', True) or samples >= 10
            required = {'window', 'band', 'min_slope', 'use_ddtw'}
            if not config or not required.issubset(config) or not status.get('ready'):
                continue
            window = int(config['window'])
            first = index - window + 1
            if window < 2 or first < start:
                continue
            observed = data['fitness'][first:index + 1].copy()
            if not np.all(np.isfinite(observed)):
                continue
            slope = float(config.get('min_slope', 0.))
            if slope == 0.:
                slope = .01 * max(1., abs(observed[-1] - observed[0])) / window
            band = int(config.get('band', 0)) or max(1, int(.1 * window))
            ramp = ramp_baseline(observed[0], slope, window)
            constant = constant_baseline(observed[0], window)
            distance = ddtw_distance if config.get('use_ddtw', False) else dtw_distance
            d1, d2 = distance(observed, ramp, window=band), distance(observed, constant, window=band)
            stored = (data['D1_vs_ramp'][index], data['D2_vs_const'][index])
            if not np.all(np.isfinite(stored)) or not np.allclose([d1, d2], stored, rtol=1e-9, atol=1e-6):
                data['verification_failures'] += 1
                continue
            candidates.append(dict(block_index=block_index, solver=switch.mh_nombre,
                                   start_index=first, end_index=index, verified=True,
                                   observed=observed - observed[0], ramp=ramp - observed[0],
                                   constant=constant - observed[0], D1=d1, D2=d2,
                                   gain=float(observed[-1] - observed[0]),
                                   flat=bool(np.all(observed == observed[0]))))
        offset = end
    progress = [c for c in candidates if c['gain'] > 0]
    if progress:
        # Choose a comparable observed/reference rise, not the largest jump.
        closest = min(progress, key=lambda c: (
            abs(c['gain'] - c['ramp'][-1]) / max(1., abs(c['ramp'][-1])),
            c['end_index']))
        data['windows'].append({**closest, 'kind': 'progress'})
    else:
        data['notes'].append('No verified progress window is available')
    flat = [c for c in candidates if c['flat']]
    runs = []
    for candidate in flat:
        if not runs or candidate['block_index'] != runs[-1][-1]['block_index'] or candidate['end_index'] != runs[-1][-1]['end_index'] + 1:
            runs.append([])
        runs[-1].append(candidate)
    if runs:
        longest = max(runs, key=len)
        data['windows'].append({**longest[(len(longest) - 1) // 2], 'kind': 'plateau'})
    else:
        data['notes'].append('No verified flat window is available')
    data['missing_detail_iterations'] = count - known
    if not fallback and not metadata.get('effective_monitor'):
        data['notes'].append('Monitor settings unavailable: references are not reconstructed')
    if data['verification_failures']:
        data['notes'].append('Unverified reference windows were excluded; recorded signals are not modified')
    return data


_PAPER_RC = {
    'font.family': 'serif',
    'font.serif': ['Nimbus Roman', 'Times New Roman', 'DejaVu Serif', 'STIX'],
    'mathtext.fontset': 'stix',
    'axes.labelsize': 8.5,
    'axes.titlesize': 9,
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5,
    'legend.fontsize': 6.5,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
}
_WINDOW_EDGE = {'progress': '#0072B2', 'plateau': '#D55E00'}


def _shade_blocks(axis, blocks, alpha):
    """Color recorded MH blocks. These are solver identities, not controller modes."""
    for block in blocks:
        axis.axvspan(block['start_index'] + .5, block['end_index'] + 1.5,
                     color=COLORES_MH.get(block['solver'], 'gray'), alpha=alpha,
                     linewidth=0, zorder=0)


def _mark_windows(axis, windows):
    for window in windows:
        axis.axvspan(window['start_index'] + 1, window['end_index'] + 1, facecolor='none',
                     edgecolor=_WINDOW_EDGE[window['kind']], linewidth=1.05, zorder=4)


def _window_heading(prefix, window):
    if not window:
        return f'{prefix}, unavailable'
    return rf"{prefix}, $t={window['end_index'] + 1}$"


def _draw_window(axis, window, title):
    """One verified window on its native iteration axis. Missing windows stay blank."""
    from matplotlib.ticker import MaxNLocator
    axis.set_title(title, loc='left', fontsize=8.5)
    axis.set_xlabel('Iteration')
    if not window:
        axis.text(.5, .5, 'No verified window', transform=axis.transAxes,
                  ha='center', va='center', fontsize=7)
        return ()
    iterations = np.arange(window['start_index'] + 1, window['end_index'] + 2)
    ramp, = axis.plot(iterations, window['ramp'], color='#0072B2', lw=1.05, zorder=2)
    observed, = axis.step(iterations, window['observed'], where='pre', color='#1a1a1a', lw=1.25, zorder=3)
    constant, = axis.plot(iterations, window['constant'], color='#D55E00', lw=1.15, ls='--', zorder=4)
    axis.set_xlim(float(iterations[0]), float(iterations[-1]))
    axis.xaxis.set_major_locator(MaxNLocator(4, integer=True))
    return observed, ramp, constant


def _compose_temporal_figure(data, metadata, mode):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    t = data['t']
    windows = {window['kind']: window for window in data['windows']}
    fig = plt.figure(figsize=(7.4, 8.6))
    grid = fig.add_gridspec(4, 1, height_ratios=[1.12, 1.02, 1.02, 1.62], hspace=.62)
    trajectory = fig.add_subplot(grid[0, 0])
    split = grid[1, 0].subgridspec(1, 2, wspace=.22)
    rising = fig.add_subplot(split[0, 0])
    flat = fig.add_subplot(split[0, 1], sharey=rising)
    distance = fig.add_subplot(grid[2, 0], sharex=trajectory)
    delta = fig.add_subplot(grid[3, 0], sharex=trajectory)

    _shade_blocks(trajectory, data['blocks'], .28)
    trajectory.step(t, data['fitness'], where='post', color='#1a1a1a', lw=1.05, label=r'$f_t^*$', zorder=3)
    for axis in (trajectory, distance, delta):
        _mark_windows(axis, data['windows'])
    trajectory.set_xlim(1, float(t[-1]) if len(t) > 1 else 2.)
    trajectory.set_ylabel('Best fitness')
    trajectory.set_title('(a)  Trajectory and active MH', loc='left')
    trajectory.legend(loc='lower right', frameon=False)
    trajectory.xaxis.set_major_locator(MaxNLocator(7, integer=True))

    progress_lines = _draw_window(rising, windows.get('progress'),
                                  _window_heading('(b)  Rising window', windows.get('progress')))
    plateau_lines = _draw_window(flat, windows.get('plateau'),
                                 _window_heading('Flat window', windows.get('plateau')))
    rising.set_ylabel('Fitness change')
    flat.tick_params(labelleft=False)
    # Shared scale leaves the flat panel's upper corner clear, so the key does not cover the rising trace.
    if plateau_lines:
        flat.legend(plateau_lines, [r'$X_t$', r'$R_t$', r'$C_t$'],
                    loc='upper left', frameon=False, handlelength=1.6)
    elif progress_lines:
        rising.legend(progress_lines, [r'$X_t$', r'$R_t$', r'$C_t$'],
                      loc='center right', frameon=False, handlelength=1.6)

    visible_c = data['panel_c_visible']
    _plot_observed(distance, t[visible_c], data['D1_vs_ramp'][visible_c], color='#0072B2', lw=.95, label='D1 vs ramp')
    _plot_observed(distance, t[visible_c], data['D2_vs_const'][visible_c], color='#D55E00', lw=.95, label='D2 vs constant')
    _plot_observed(distance, t[visible_c], data['theta_r'][visible_c], color='#0072B2', lw=.8, ls='--', label='theta_r')
    _plot_observed(distance, t[visible_c], data['theta_c'][visible_c], color='#D55E00', lw=.8, ls='--', label='theta_c')
    # Keep all post-warm-up magnitudes on the requested linear scale.
    distance.set_yscale('linear')
    distance.yaxis.set_major_formatter(FuncFormatter(
        lambda value, _: f'{value:,.0f}' if abs(value) >= 1 or value == 0 else f'{value:g}'))
    distance.set_ylabel('Distance (linear)')
    distance.set_title('(c)  Distances and thresholds after warm-up', loc='left')
    distance.tick_params(labelbottom=False)
    distance.legend(loc='upper right', frameon=False, ncol=2)

    _plot_observed(delta, t, data['delta'], color='#009E73', lw=.95, label='Recorded delta')
    _plot_observed(delta, t, data['theta_delta'], color='#009E73', lw=.8, ls='--', label='Recorded theta_delta')
    fires = data['fire'] == 1
    visible = fires & np.isfinite(data['delta'])
    if np.any(visible):
        delta.scatter(t[visible], data['delta'][visible], marker='v', s=18, color='#C62828',
                      zorder=5, label='Recorded DTW fire')
    finite = data['delta'][np.isfinite(data['delta'])]
    if finite.size:
        low, high = float(np.min(finite)), float(np.max(finite))
        pad = max(1., .12 * (high - low if high > low else 1.))
        delta.set_ylim(low - pad, high + pad)
    delta.set_ylabel('Delta DTW')
    delta.set_title('(d)  Delta, threshold and recorded triggers', loc='left')
    delta.set_xlabel('Native Iteration $t$')
    delta.legend(loc='lower right', frameon=False, ncol=2)

    seed = metadata.get('seed', 'not supplied')
    fig.suptitle(f'MKP temporal diagnostics — {mode}, seed {seed}', fontsize=8, y=.995)
    fig.subplots_adjust(left=.16, right=.985, top=.942, bottom=.118)
    missing = data['missing_detail_iterations']
    note = (' '.join(data['notes']) + ' ') if data['notes'] else ''
    fig.text(.16, .008,
             f'Details unavailable at {missing}/{len(t)} iterations. {note}'
             'Panels (c) and (d) join recorded samples only; connecting segments are visual guides, not measurements.\n'
             'Panel (c) is linear and omits block warm-up plus the first 9 ready samples when thresholds adapt.\n'
             'All later magnitudes are retained; CSV preserves the complete recorded series.',
             fontsize=6.1, va='bottom', ha='left')
    return fig


def plot_temporal_diagnostics(result: PipelineResult, output: Path, metadata: dict | None = None) -> list[str]:
    """Draw the compact reference layout. No BDE F/CR or exploration/exploitation state."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    data = build_temporal_data(result, metadata)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    metadata = metadata or {}
    mode = ('DDTW' if metadata.get('use_ddtw') is True else
            'DTW' if metadata.get('use_ddtw') is False else 'DTW/DDTW (mode unavailable)')
    paths = [output / 'fig_temporal_diagnostics.png', output / 'fig_temporal_diagnostics.pdf']
    figure = None
    try:
        with plt.rc_context(_PAPER_RC):
            figure = _compose_temporal_figure(data, metadata, mode)
            figure.savefig(paths[0], dpi=300)
            figure.savefig(paths[1])
    finally:
        if figure is not None:
            plt.close(figure)
    t = data['t']
    write_json(output / 'temporal_diagnostics.json', {
        'iterations': len(t), 'seed': metadata.get('seed'), 'use_ddtw': metadata.get('use_ddtw'),
        'missing_detail_iterations': data['missing_detail_iterations'],
        'verification_failures': data['verification_failures'],
        'recorded_fire_count': int(np.sum(data['fire'] == 1)),
        'mh_transitions': max(0, len(data['blocks']) - 1),
        'windows': data['windows'], 'notes': data['notes'],
        'selection_rule': 'Closest relative window gain to its verified ramp; middle of longest consecutive verified flat-window segment',
        'reference_note': 'Derived using recorded monitor settings and verified against recorded D1/D2; not new optimization data',
        'postprocessor_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'best_fitness': float(result.mejor_valor_global),
        'data_source': 'Recorded native histories; optimization is not executed by this postprocessor',
        'rendering': {'panel_c_scale': 'linear', 'panel_c_linthresh': None,
                      'panel_c_start': 'full_window_and_adaptive_thresholds_per_block',
                      'panel_c_hidden_ready_samples': int(np.sum(np.isfinite(data['D1_vs_ramp']) & ~data['panel_c_visible'])),
                      'panel_c_visible_iterations': int(np.sum(data['panel_c_visible'])),
                      'connections': 'recorded_samples_only',
                      'layout': 'reference_7.4x8.6_b_split',
                      'd_strip': 'none',
                      'note': 'Straight visual connections across missing iterations are not measurements; CSV retains missing values'},
    })
    fields = ['iteration', 'fitness', 'active_mh', 'pool', *METRICS]
    write_csv(output / 'temporal_diagnostics.csv', [
        dict(iteration=int(t[i]), fitness=float(data['fitness'][i]), active_mh=data['active_mh'][i],
             pool=data['pool'][i], **{key: data[key][i] for key in METRICS}) for i in range(len(t))
    ], fields)
    return [str(path) for path in paths]


def load_recorded_run(folder: Path):
    folder = Path(folder)
    metadata = json.loads((folder / 'summary.json').read_text(encoding='utf-8'))
    if metadata.get('schema_version') != 2 or metadata.get('budget_unit') != 'native_iterations':
        raise ValueError('A recorded native-iteration run is required; evaluation campaigns are not converted')
    raw = json.loads((folder / 'native_result.json').read_text(encoding='utf-8'))
    for key in ('historial_global', 'historial_inst_global', 'dtw_deltas_global'):
        raw[key] = [_number(value) for value in raw[key]]
    raw['log_switches'] = [SwitchLog(**value) for value in raw['log_switches']]
    result = PipelineResult(**raw)
    if len(result.historial_global) != metadata['actual_iterations']:
        raise ValueError('Saved iteration count disagrees with the native history')
    return result, metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True, help='Saved run/best_run directory; no solvers are executed')
    args = parser.parse_args(argv)
    try:
        result, metadata = load_recorded_run(args.run)
        from .plotting import plot_run
        paths = plot_run(result, args.run, metadata=metadata)
    except (ValueError, KeyError, OSError, TypeError) as error:
        parser.error(str(error))
    print(f'Redrawn {len(paths)} figure files from stored data: {args.run}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
