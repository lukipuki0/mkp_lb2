"""Synchronized four-panel diagnostics, adapted from fig_temporal_diagnostics (2).

Replay only the monitor, never the optimizer or RNG. Verify the recorded
observations before exporting; reset after the decision, not before it.
"""
import json
from pathlib import Path

import numpy as np
from dtw_stagnation import (StagnationConfig, StagnationMonitor,
                            constant_baseline, ramp_baseline,
                            dtw_distance, ddtw_distance)
from ._common import plt, profile_color, profile_segments


SCALARS = ('t', 'f_star', 'D_R', 'D_C', 'delta', 'theta_R', 'theta_C',
           'theta_delta', 's', 'mode_used', 'a', 'F', 'CR', 'ready',
           'adaptive_threshold', 'fire', 'no_improve_len', 'trigger_streak')


def build_series(res: dict) -> dict:
    """Reconstruct the exact filtered windows using the saved monitor config."""
    cfg = StagnationConfig(**res['monitor_config'])
    monitor = StagnationMonitor(cfg)
    count, width = len(res['best_series']), cfg.window
    data = {key: np.full(count, np.nan) for key in SCALARS}
    data.update({key: np.full((count, width), np.nan) for key in ('X', 'R', 'C')})
    data['t'] = np.arange(1, count + 1)
    data['f_star'] = np.asarray(res['best_series'], dtype=float)
    for key in ('a', 'F', 'CR'):
        data[key] = np.asarray(res['parameter_series'][key], dtype=float)
    for i, fitness in enumerate(res['best_series']):
        out = monitor.update(fitness)
        saved = res['monitor'][i]
        for key, value in out.items():
            if key not in saved or not np.isclose(value, saved[key], rtol=0, atol=1e-8):
                raise ValueError(f'Monitor mismatch at generation {i + 1}: {key}')
        data['s'][i] = int(saved['profile'] == 'explore')
        data['mode_used'][i] = int(res['profile_series'][i] == 'explore')
        for key in ('ready', 'fire', 'no_improve_len', 'trigger_streak'):
            data[key][i] = out.get(key, 0)
        data['adaptive_threshold'][i] = bool(cfg.adapt_thresholds and len(monitor.d1_hist) >= 10)
        if out['ready']:
            x = np.asarray(monitor.best_so_far[-width:])
            slope = cfg.min_slope or (.01 * max(1., abs(x[-1] - x[0])) / width)
            r = ramp_baseline(x[0], slope, width)
            c = constant_baseline(x[0], width)
            distance = ddtw_distance if cfg.use_ddtw else dtw_distance
            for actual, expected in ((distance(x, r, cfg.band), out['D1_vs_ramp']),
                                     (distance(x, c, cfg.band), out['D2_vs_const'])):
                if not np.isclose(actual, expected, rtol=0, atol=1e-8):
                    raise ValueError(f'Distance mismatch at generation {i + 1}')
            data['X'][i], data['R'][i], data['C'][i] = x, r, c
            for target, source in (('D_R', 'D1_vs_ramp'), ('D_C', 'D2_vs_const'),
                                   ('delta', 'delta'), ('theta_R', 'theta_r'),
                                   ('theta_C', 'theta_c'), ('theta_delta', 'theta_delta')):
                data[target][i] = out[source]
        if saved['changed']:
            monitor = StagnationMonitor(cfg)
    return data


def select_windows(data):
    """Rising window closest to ramp rise; midpoint of longest flat explore run.

    Missing categories remain missing: do not label a rising window as flat.
    """
    ready = data['ready'].astype(bool)
    rise = data['X'][:, -1] - data['X'][:, 0]
    rising = np.flatnonzero(ready & (rise > 0) & (data['mode_used'] == 0))
    if not rising.size:
        rising = np.flatnonzero(ready & (rise > 0))
    flat = ready & (rise == 0) & (data['mode_used'] == 1)
    if not flat.any():
        flat = ready & (rise == 0)
    best, start = [], None
    for i, value in enumerate(list(flat) + [False]):
        if value and start is None:
            start = i
        elif not value and start is not None:
            if i - start > len(best):
                best = list(range(start, i))
            start = None
    ramp_rise = data['R'][:, -1] - data['R'][:, 0]
    progress = (int(rising[np.argmin(np.abs(rise[rising] - ramp_rise[rising]))])
                if rising.size else None)
    return (progress,
            best[len(best) // 2] if best else None)


def plot_temporal(res: dict, output_dir) -> str:
    """Write PNG/PDF plus NPZ windows, scalar CSV and metadata for one run."""
    if not res['best_series']:
        return ''
    if 'monitor_config' not in res:
        print('  [plot] temporal_diagnostics.png -> skipped: missing monitor_config')
        return ''  # Old JSONs cannot safely infer the original configuration.
    data = build_series(res)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    meta = {key: res[key] for key in ('instance', 'seed', 'binarization', 'monitor_config',
                                    'best_fitness', 'gap_pct', 'evaluations') if key in res}
    meta['notes'] = {'s': 'Decision after t, used at t+1; 0 exploit, 1 explore.',
                     'mode_used': 'Profile that generated iteration t; used for shading.',
                     'X': 'Monitor-filtered window; NaN while warming up after each reset.',
                     'parameters': 'a, F, CR used at t; G3 is not plotted.',
                     'window_selection': 'Rising exploit window with total rise closest to its ramp (fallback any rising); midpoint of longest flat explore run (fallback any flat). Missing categories omitted.'}
    np.savez_compressed(output / 'temporal_series.npz', **data)
    np.savetxt(output / 'temporal_series.csv', np.column_stack([data[k] for k in SCALARS]),
               delimiter=',', header=','.join(SCALARS), comments='')
    (output / 'temporal_metadata.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
    with plt.rc_context({'font.size': 9}):
        fig = plt.figure(figsize=(12, 13), layout='constrained')
        grid = fig.add_gridspec(4, 1, height_ratios=[1, 1, 1, 1.4])
        ax_a = fig.add_subplot(grid[0])
        windows = grid[1].subgridspec(1, 2)
        ax_bp, ax_bc = fig.add_subplot(windows[0]), fig.add_subplot(windows[1])
        ax_c = fig.add_subplot(grid[2], sharex=ax_a)
        bottom = grid[3].subgridspec(2, 1)
        ax_d, ax_s = fig.add_subplot(bottom[0], sharex=ax_a), fig.add_subplot(bottom[1], sharex=ax_a)
        t = data['t']
        for ax in (ax_a, ax_s):
            for name, start, end in profile_segments(res['profile_series']):
                ax.axvspan(start + .5, end + .5, color=profile_color(name), alpha=.18)
        ax_a.step(t, data['f_star'], where='mid', color='black')
        ax_a.set(title='(a) Best fitness and profile used', ylabel='Best fitness')
        selected = select_windows(data)
        width = res['monitor_config']['window']
        for ax, idx, title, color in zip((ax_bp, ax_bc), selected,
                                        ('Rising window', 'Flat window'), ('tab:blue', 'tab:orange')):
            ax.set(title='(b) ' + title, xlabel='Generation', ylabel='Change from window start')
            if idx is None:
                ax.text(.5, .5, 'No eligible window', transform=ax.transAxes, ha='center')
                continue
            xs = np.arange(idx - width + 2, idx + 2)
            ax_a.axvspan(xs[0], xs[-1], facecolor='none', edgecolor=color, linewidth=1.5)
            for key, label, style, color in (('X', 'Observed', '-', '#1a1a1a'),
                                              ('R', 'Ramp', '-', '#0072B2'),
                                              ('C', 'Constant', '--', '#D55E00')):
                y = data[key][idx] - data[key][idx, 0]
                ax.plot(xs, y, style, color=color, label=label,
                        drawstyle='steps-pre' if key == 'X' else 'default')
            ax.legend()
        for key, color, style in (('D_R', 'tab:blue', '-'), ('theta_R', 'tab:blue', '--'),
                                   ('D_C', 'tab:orange', '-'), ('theta_C', 'tab:orange', '--')):
            valid = np.isfinite(data[key]) & data['adaptive_threshold'].astype(bool)
            ax_c.plot(t[valid], data[key][valid], color=color, linestyle=style, label=key)
        signal = 'DDTW' if res['use_ddtw'] else 'DTW'
        ax_c.set(title='(c) Distances and thresholds (adaptive samples only)', ylabel=signal + ' distance')
        ax_c.legend(ncol=4)
        # Join available samples visually; exported missing samples stay NaN.
        for key, label, style in (('delta', 'Delta', '-'), ('theta_delta', 'Delta threshold', '--')):
            valid = np.isfinite(data[key]) & data['adaptive_threshold'].astype(bool)
            ax_d.plot(t[valid], data[key][valid], style, label=label, color='tab:green')
        ax_d.set(title='(d) Relative indicator and controller decision', ylabel='D_R - D_C')
        ax_d.legend()
        ax_s.step(t, data['s'], where='post', color='black', label='Decision s(t), applied at t+1')
        ax_s.set(yticks=[0, 1], yticklabels=['Exploit', 'Explore'], ylim=(-.2, 1.4), xlabel='Generation (1..T)')
        ax_s.legend(loc='upper right')
        # Describe parameters actually used, without plotting the fixed G3.
        descriptions = []
        for name in dict.fromkeys(res['profile_series']):
            i = res['profile_series'].index(name)
            descriptions.append(name + ': ' + ', '.join(f'{key}={data[key][i]:g}' for key in ('a', 'F', 'CR')))
        ax_s.set_title('; '.join(descriptions), fontsize=9)
        for ax in (ax_a, ax_c, ax_d, ax_s):
            ax.set_xlim(.5, len(t) + .5)
        fig.suptitle(f'GWO-DE temporal diagnostics | {signal} | seed {res["seed"]}')
        path = output / 'temporal_diagnostics.png'
        fig.savefig(path, dpi=150)
        fig.savefig(output / 'temporal_diagnostics.pdf')
        plt.close(fig)
    print(f"  [plot] temporal_diagnostics.png -> '{path}'")
    return str(path)
