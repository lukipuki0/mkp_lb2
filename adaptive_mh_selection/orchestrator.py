"""Run-local policy over the actual hybrid_mkp pipeline; no solver loop here."""

from contextlib import contextmanager
from dataclasses import asdict
import math
import random
import time

import numpy as np
from hybrid_mkp import orchestrator as native
from .config import RunConfig, positive_integer
from .reward import compute_reward
from .selection import AdaptiveSelector, AdaptiveExploratorySelector, UniformSelector


@contextmanager
def isolated_rng(seed):
    """Seed once for the entire native run and restore both caller RNGs.

    Sequential use only. Adaptive choices have independent Random streams;
    uniform choice consumes the same global stream as the unwrapped pipeline.
    """
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        random.seed(seed)
        np.random.seed(seed)
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


class SelectionPolicy:
    """Implement native choose/observe, updating only the completed block's pool."""

    def __init__(self, config: RunConfig):
        cls = {'adaptive': AdaptiveSelector, 'uniform': UniformSelector,
               'adaptive_exploratory': AdaptiveExploratorySelector}[config.selector]
        self.selectors = {
            tipo: cls(tuple(names), random.Random(f'{config.seed}:{tipo}'),
                      alpha=config.alpha, epsilon=config.epsilon)
            for tipo, names in (('poblacional', config.population_pool),
                                ('trayectoria', config.trajectory_pool))
        }
        self.first_block_fitness = None
        self.records = []
        self.iterations = 0
        self.pending = None

    def choose(self, pool_names, tipo):
        selector = self.selectors[tipo]
        expected = (native.POOL_POBLACIONAL if tipo == 'poblacional'
                    else native.POOL_TRAYECTORIA)
        if tuple(pool_names) != tuple(expected) or self.pending is not None:
            raise ValueError('Policy requires native pools and one completed block per choice')
        mh, probabilities = selector.choose()
        self.pending = (mh, tipo, probabilities, selector.scores.copy())
        return mh

    def observe(self, mh, tipo, fitness_before, epoch_result, n_iters, fitness_after):
        if self.pending is None or self.pending[:2] != (mh, tipo):
            raise ValueError('Observation must match the pending choice')
        positive_integer(n_iters, 'n_iters')
        if n_iters != len(epoch_result.historial):
            raise ValueError('Iterations must equal len(native historial)')
        if not math.isfinite(fitness_after):
            raise ValueError('Completed block fitness must be finite')
        warmup = self.first_block_fitness is None
        if warmup:
            if fitness_before != -math.inf:
                raise ValueError('First native block must inherit -inf')
            first_fitness = float(fitness_after)
            gain, raw, reward = None, 0., 0.
        else:
            first_fitness = self.first_block_fitness
            gain, raw, reward = compute_reward(fitness_before, fitness_after, first_fitness, n_iters)
        selector = self.selectors[tipo]
        selector.update(mh, reward)
        _, _, probabilities, scores_before = self.pending
        info = getattr(epoch_result, 'dtw_info_hist', None) or []
        self.records.append({
            'block_index': len(self.records), 'pool': tipo, 'solver': mh,
            'iteration_start': self.iterations, 'iteration_end': self.iterations + n_iters,
            'n_iters': n_iters, 'fitness_before': None if warmup else float(fitness_before),
            'fitness_after': float(fitness_after), 'first_block_fitness': first_fitness,
            'warmup': warmup, 'gain': gain, 'raw_reward': raw, 'reward': reward,
            'probabilities_before': probabilities, 'scores_before': scores_before,
            'scores_after': selector.scores.copy(),
            'dtw_info_iterations': len(info), 'dtw_info_complete': len(info) == n_iters,
        })
        self.first_block_fitness = first_fitness
        self.iterations += n_iters
        self.pending = None


def run_pipeline(instance, config: RunConfig, *, verbose=False) -> dict:
    """Return native_result (PipelineResult), selection_records and summary.

    Native histories/NaNs are untouched. Only summary and selection_records are
    JSON-native. There is no added initial solution, evaluation proxy or per-block
    reseeding. max_epoch_iters=None retains the original epoch-boundary overshoot.
    """
    policy = SelectionPolicy(config)
    started = time.perf_counter()
    with isolated_rng(config.seed):
        result = native.ejecutar_pipeline(
            inst=instance, max_iters=config.max_iters, stag_cfg=config.monitor,
            pop_injection_mode=config.pop_injection_mode, verbose=verbose,
            max_epoch_iters=config.max_epoch_iters, policy=policy,
        )
    actual = len(result.historial_global)
    gap = result.gap_pct
    settings = asdict(config)
    summary = {
        'schema_version': 2, 'budget_unit': 'native_iterations', 'status': 'success',
        'seed': config.seed, 'selector': config.selector, 'use_ddtw': config.use_ddtw,
        'requested_iterations': config.max_iters, 'actual_iterations': actual,
        'iteration_overshoot': max(0, actual - config.max_iters),
        'best_fitness': float(result.mejor_valor_global) if math.isfinite(result.mejor_valor_global) else None,
        'best_solution': [int(x) for x in result.mejor_solucion_global],
        'first_block_fitness': policy.first_block_fitness,
        'gap_pct': float(gap) if gap is not None and math.isfinite(gap) else None,
        'n_blocks': len(result.log_switches), 'transitions': max(0, len(result.log_switches) - 1),
        'dtw_info_iterations': sum(r['dtw_info_iterations'] for r in policy.records),
        'dtw_info_complete': all(r['dtw_info_complete'] for r in policy.records),
        'elapsed_seconds': time.perf_counter() - started,
        'settings': settings,
        'effective_monitor': {
            name: {**asdict(config.monitor),
                   'patience': max(config.monitor.patience, 14) if name in ('GA', 'ILS', 'WOA', 'GWO')
                   else config.monitor.patience}
            for name in native.POOL_POBLACIONAL + native.POOL_TRAYECTORIA
        },
        'rng_protocol': 'python+numpy seeded once; uniform shares random.choice; adaptive Random(seed:pool); sequential only',
    }
    return {'native_result': result, 'selection_records': policy.records, 'summary': summary}
