# Binary WOA–ABC

Shared-population solver for the multidimensional knapsack problem. One population is initialized once. Every complete cycle then runs whale optimization, employed bees, onlookers, and eligible scouts. A stagnation detector can restart only the `a` and `rho` clock. It does not abort the run, reset the population, or choose another algorithm.

The code is the `woa_abc` package. Run the solver with `python -m woa_abc`. Save one functional run with `python -m woa_abc.benchmark`. Results default to `woa_abc/resultados/`, beside this package, not the process working directory.

This document describes the implementation. It does not claim that the hybrid improves on any published method.

## Population

Each source keeps four pieces of state:

- a continuous position
- a repaired binary solution
- a cached fitness
- an ABC failure counter

Initialization follows the WOA sign-bit rule: positions are drawn uniformly from `[-1, 1]`, positive coordinates become 1, and each bit row is repaired once. Scouts later replace an exhausted source with a position in `[-v_max, v_max]` and independent random bits. The archived best survives that replacement.

LB2 builds two flip candidates from the existing L1 and L2 transfers. Both repaired candidates are evaluated. A tie keeps the L2 candidate. WOA always stores the new position and accepts a binary fitness tie. ABC accepts only a strict improvement and updates position, bits, and fitness together. A failure counter is cleared only when fitness actually improves.

The search uses a local NumPy generator. It does not draw from the global Python or NumPy RNGs.

## Cycle

A complete cycle has a fixed order:

1. WOA, one proposal per source
2. ABC employed bees, one proposal per source
3. ABC onlookers, one fitness-roulette proposal per source
4. scouts for sources whose failure counter has reached the limit

The onlooker probability is raw fitness divided by the fitness sum. The default scout limit is population size times the number of items. The default population is 30 and the default horizon `T` is 300. `v_max` is 6, the spiral constant `b` is 1, and `G1 = G2 = G3 = 0.5`.

The WOA leader coordinates are copied at the start of the WOA phase. Later updates to that whale's row do not move the target used by the other whales. The peer branch and the spiral are both available. New positions are clipped to `[-v_max, v_max]`.

## Clocks

`T` is the configured cycle count and stays fixed after every recovery.

Before the first fire, `s` is the ordinary cycle index, `a = 2 - 2s/(T-1)`, and `rho = 1`. ABC samples its scalar step from `U[-rho, rho]`, so the baseline interval is `[-1, 1]`.

After a fire, the next executed cycle starts at `s = 0`, which gives `a = 2` and `rho = 2`. Later recovered cycles use the same original horizon:

- `a = max(0, 2 - 2s/(T-1))`
- `rho = max(1, 2 - s/(T-1))`

The slope is therefore the same before and after a restart. A fire on the last cycle is recorded, but recovery is not marked applied because no following cycle runs.

`a2` and the spiral draw follow the ordinary cycle index. They are not restarted. Only `a` and `rho` use the recovery clock.

## Detector

Detection uses the existing `StagnationMonitor` and `StagnationConfig`. This solver does not change their window, plateau, thresholds, or patience. `stag_cfg = None` means those existing defaults. It does not turn detection off. Set `use_stagnation=False`, or pass `--no-stagnation`, for the fixed baseline.

The monitor receives the archived best once per complete cycle. It does not receive a partial cycle, and the solver does not evaluate that best again for the detector. On fire the solver copies the returned ready flag, distances, thresholds, and counters, then calls `monitor.reset()`. The next fire can occur only after warm-up, the detector conditions, and patience have accrued again. Population, failure counters, and the archive are kept.

Each diagnostic stores the cycle, schedule index, cumulative evaluations, `a`, `rho`, best, detector status, and whether recovery was applied to a later cycle. Two `solve` calls copy the monitor settings and do not share monitor state. Passing a config object does not rewrite it.

Invalid monitor options are rejected at the config and CLI boundary: window at least 2, band at least 0, plateau at least 0, patience at least 1, and finite nonnegative slope and tolerance. Percentiles must satisfy `0 <= p_low <= p_high <= 100`.

## Budgets

Every repair is one objective evaluation. Initialization and each scout cost one. An LB2 proposal costs two and is not started unless two evaluations remain, so a terminal budget may leave one evaluation unused. If a phase cannot finish, later phases in that cycle do not run. An incomplete cycle is recorded and does not update the detector.

The run stops at the cycle horizon or at the evaluation ceiling, whichever comes first. Phase counters for initialization, WOA, employed bees, onlookers, and scouts sum to the total evaluation count.

## API

```python
from woa_abc import HybridConfig, solve
from dtw_stagnation import StagnationConfig
from mkp_core.problem import MKPInstance

config = HybridConfig(
    population_size=4,
    iterations=6,
    max_evaluations=200,
    stag_cfg=StagnationConfig(window=4, band=1, min_slope=2.0, plateau_max=2, patience=2),
)
result = solve(instance, config, seed=1)
```

`instance` must be an `MKPInstance` with nonnegative profits, weights, and capacities, and positive weight on every item. A zero reference means the optimum is unknown and `gap` is `None`. A positive reference produces the percentage gap `(reference - best) / reference * 100`.

`result.metadata` repeats the seed, the resolved budget and scout limit, the LB2 constants, and the effective detector settings. `result.history` has one record per entered cycle, including a partial final cycle.

## CLI

The command reads one local OR-Library file. It rejects URLs and other non-files before importing the parser, and it does not download data.

```text
python -m woa_abc --instance instancias/mknapcb1.txt --instance-index 0 --seed 1 --population-size 4 --iterations 6 --max-evaluations 200
```

`--instance-index` is zero-based. `--scout-limit` is optional. `--ddtw` selects derivative DTW. `--window`, `--band`, `--min-slope`, `--plateau-max`, and `--patience` override the corresponding monitor defaults. `--no-stagnation` keeps `rho` at 1. JSON is printed to stdout and includes the search result, diagnostics, effective config, and selected instance metadata (`path`, `index`, `n`, `m`, `reference`).

## Checks

The focused check for this solver is:

```text
python -m unittest discover -s tests -p "test_binary_woa_abc.py" -v
```

The older full suite is not the check for this work. `tests/test_woa_abc_variants.py` imports removed legacy modules and is not the check for this package.

## Single-run results

One local command writes a fresh results folder for one functional check. It does not compare algorithms, and it does not download data.

```text
PYTHONDONTWRITEBYTECODE=1 python -m woa_abc.benchmark --instance instancias/mknapcb1.txt --instance-index 0 --seed 43 --population-size 30 --iterations 1000
```

Defaults are seed 43, population 30, 1000 hybrid cycles, and index 0. The detector settings are the DDTW settings copied from `hybrid_mkp/batch_benchmark.py`: window 40, band 0 resolved to the auto band 4, min slope 0.1, plateau 15, patience 8, adaptive percentiles 30/70, and improvement tolerance 0. A stored reference of 0 stays unknown.

The folder is `woa_abc/resultados/run_ddtw_<timestamp>/<file stem>/inst_<index>/`. The timestamp directory must not already exist. The completed seed-43 run is `woa_abc/resultados/run_ddtw_20261019_071137/`. It was relocated here and was not rerun.

| File | What it is |
|------|------------|
| `manifest.json` | Effective solver config, input path/hash/index, timing, actual counts, output completeness |
| `runs/run_01.json` | Full solver result, written before plots |
| `runs_resultados.csv` | One row. `n_switches` is 0 |
| `resumen_pipeline.txt` | Single-run note. No superiority claim |
| `boxplot_runs.png` | Labelled n = 1. Not inferential evidence |
| `best_run_01/` | History, detail, convergence, delta, and `parameter_recovery.png` |

A hybrid cycle is not a legacy single-metaheuristic iteration. `n_recoveries` counts `a`/`rho` restarts, not algorithm switches. If plotting fails, the raw JSON stays and the process exits nonzero without a second solve.
