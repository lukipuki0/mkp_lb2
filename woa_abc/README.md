# Binary WOA–ABC

Shared-population solver for the multidimensional knapsack problem. One population is initialized once. Every complete cycle then runs whale optimization, employed bees, onlookers, and eligible scouts, in that order. Two separate reaction modes are available: the four GWO–DE policies (default), or the original recovery-clock variants v0–v4. They never combine. It does not abort the run, reset the population, or choose another algorithm.

The code is the `woa_abc` package. Run the solver with `python -m woa_abc`. Save one functional run with `python -m woa_abc.benchmark`. Run the four policies side by side with `python -m woa_abc.run`, or five variants with `--reaction variant`. Run the 135-selection variant campaign with `python -m woa_abc.campaign`. Results default to `woa_abc/resultados/`, beside this package, not the process working directory.

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

The onlooker probability is raw fitness divided by the fitness sum. The default scout limit is population size times the number of items. The default population is 30 and the default horizon `T` is 300. `v_max` is 6, the spiral constant `b` is 1, and `G1 = G2 = 0.5`. `G3` comes from the active profile.

The WOA leader coordinates are copied at the start of the WOA phase. Later updates to that whale's row do not move the target used by the other whales. The peer branch and the spiral are both available. New positions are clipped to `[-v_max, v_max]`.

`a2` and the spiral draw follow the ordinary cycle index: `a2 = -1 - s/(T-1)`. Profiles do not change them.

## Policy mode (default)

`HybridConfig(reaction="policy")` selects policies; no `variant` is accepted. Omitted `policy` resolves to `binary_complex`.

### Profiles

A profile fixes three values for a whole cycle:

| Profile | WOA `a` | ABC `rho` | LB2 `G3` |
|---------|---------|-----------|----------|
| exploit | 0.5 | 1.0 | 0.5 |
| explore | 2.0 | 2.0 | 0.5 |

`a` is constant within a profile. There is no linear decay. ABC samples its scalar step from `U[-rho, rho]`. With `a = 0.5`, `|A| < 1`, so the WOA peer branch only appears in explore.

These values are provisional. They live in one place, `woa_abc/profiles.py`.

## Stagnation reaction

The reaction mirrors `gwo_de_dtw/controller.py` and `gwo_de_dtw/policies.py`. Choose it with `policy` (CLI `--policy`). The run starts in exploit.

| Policy | Rule on a ready observation |
|--------|-----------------------------|
| `binary_simple` | exploit → explore when `D2_vs_const <= theta_c` |
| `binary_complex` (default) | exploit → explore when the monitor fires |
| `continuous_simple` | intensity = `1 - clip(D2_vs_const / (theta_c * 2.0 + eps), 0, 1)` |
| `continuous_complex` | intensity = `sigmoid(5.0 * (delta / (theta_delta + eps) - 0.5))` |

Binary policies:

- In explore, return to exploit when `no_improve_len < plateau_max`. Otherwise stay in explore.
- After every profile change the monitor is replaced by a fresh one. No decision is taken until `window` new observations made with the new profile exist.
- `dtw_kirito=True` (CLI `--dtw-kirito`, default off) keeps one monitor for the whole run. Every ready observation sets explore if the policy trigger holds, exploit otherwise. It has no effect on continuous policies.

Continuous policies:

- The monitor is never replaced.
- Intensity is 0 before the first ready observation.
- The active profile is the linear blend of `a`, `rho`, and `G3`: exploit at 0, explore at 1. It is named explore when intensity is above 0.5.

A change decided after cycle `g` applies from cycle `g + 1`. Population, failure counters, and the archive are never reset. `use_stagnation=False`, or `--no-stagnation`, keeps the exploit profile for the whole run.

## Detector

Detection uses the existing `StagnationMonitor` and `StagnationConfig`. This solver does not change their window, plateau, thresholds, or patience. `stag_cfg = None` means those existing defaults. It does not turn detection off.

The monitor receives the archived best once per complete cycle. It does not receive a partial cycle, and the solver does not evaluate that best again for the detector.

Each `history` record stores the profile used by that cycle: `profile`, `a`, `rho`, `G3`, and `intensity` (None for binary policies). Each diagnostic stores the cycle, cumulative evaluations, best, the profile used, the detector status (ready flag, distances, thresholds, counters), `changed`, `next_profile`, and `next_intensity`. `profile_changes` lists the cycles after which the profile changed.

Two `solve` calls copy the monitor settings and do not share monitor state. Passing a config object does not rewrite it.

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
    policy="binary_complex",
    dtw_kirito=False,
)
result = solve(instance, config, seed=1)
```

`instance` must be an `MKPInstance` with nonnegative profits, weights, and capacities, and positive weight on every item. A zero reference means the optimum is unknown and `gap` is `None`. A positive reference produces the percentage gap `(reference - best) / reference * 100`.

## References

The headers of `instancias/mknapcb*.txt` store 0. Every command that loads a file (`python -m woa_abc`, `woa_abc.benchmark`, `woa_abc.campaign`, `woa_abc.run`) then takes the best known value of that same index from `mkcbres.txt` in the folder of the instance file (OR-Library, Chu & Beasley 1997, first table "Best Feasible Solution Value"). The lookup is `woa_abc/best_known.py`, copied from `gwo_de_dtw/problem.py` on `feat/gwo-de-dtw-hybrid`: `mknapcbK.txt` index `i` is the entry `<m>.<n>-<ii>` of `FAMILY_PREFIX`, for example `mknapcb1.txt` index 0 is `5.100-00` = 24381. A positive header value is kept. Without `mkcbres.txt`, or without that entry, the reference is 0 and `gap` is `None`. It is never copied from another index.

| `reference_source` | Meaning |
|--------------------|---------|
| `header` | Positive value stored in the instance file |
| `mkcbres.txt` | Header was 0; best known value from `mkcbres.txt` |
| `unknown` | Header was 0 and no best known entry; reference 0, no gap |

The solver JSON (`metadata.instance`), the single-run manifest (`input`), `runs_resultados.csv`, `resumen_pipeline.txt`, and the campaign files carry `reference` and `reference_source`.

`result.metadata` repeats the seed, the resolved budget and scout limit, the LB2 constants, the policy, `dtw_kirito`, both profiles, and the effective detector settings. `result.history` has one record per entered cycle, including a partial final cycle.

## Variant mode (separate)

API: `HybridConfig(reaction="variant", variant="v0")`. CLI: `--variant v0` (also v1–v4) selects this mode. No `policy` or `dtw_kirito` is accepted in variant mode, even an explicitly supplied default policy. Policy mode has no variant. This restores the search semantics of commit `8cc99f5`, not a variant of the profile policies.

`T` stays fixed. Before any applied fire, `s` is the ordinary cycle index, `a = max(0, 2 - 2s/(T-1))`, and `rho = 1`. On fire the monitor is reset, not the population, failure counters or archive. Recovery is applied only when a next configured cycle exists and at least two evaluations remain for its first LB2 pair. A last-cycle fire is recorded but not applied. Partial cycles never update the monitor. `a2 = -1 - cycle/(T-1)` never restarts. G3 stays 0.5.

| Variant | Only change |
|---------|-------------|
| `v0` | Applied fire → next cycle `s=0`, `a=2`, `rho=2`; then `a=max(0,2-2s/(T-1))`, `rho=max(1,2-s/(T-1))` |
| `v1` | Same clock; omitted scout limit is 100. An explicit limit still wins |
| `v2` | After applied fire: `a=max(0,1-2s/(T-1))`, `rho=max(1,1.5-s/(T-1))`; pre-fire matches v0 |
| `v3` | After applied fire recompute remaining cycles R, including the next cycle; for R>1 use `a=max(0,2-2s/(R-1))`, `rho=max(1,2-s/(R-1))`. R=1 uses a=0, rho=1. Another applied fire recomputes R |
| `v4` | Same clock; employed bees and onlookers perturb one random coordinate with the scalar phi; full-vector LB2/repair and strict ABC acceptance remain unchanged |

With detection off the ordinary clock still decays a, with rho=1 throughout. JSON always names `reaction` and, in variant mode, `variant` including v0; it contains no policy. Variant history records `schedule_index`, `segment` (base/recovery), a, rho, G3, best, evaluations and completeness. Diagnostics record every fire and `recovery_applied`; `profile_changes` stays empty. Exported tables distinguish `n_recoveries` from policy profile changes and algorithm switches.

## CLI

The command reads one local OR-Library file. It rejects URLs and other non-files before importing the parser, and it does not download data.

```text
python -m woa_abc --instance instancias/mknapcb1.txt --instance-index 0 --seed 1 --population-size 4 --iterations 6 --max-evaluations 200
```

`--instance-index` is zero-based. `--scout-limit` is optional. Omitting `--variant` selects policy mode. `--variant v0` through `--variant v4` instead selects variant mode and cannot be combined with `--policy` or `--dtw-kirito`. `--policy` is `binary_complex` when omitted; the choices are `binary_simple`, `binary_complex`, `continuous_simple`, and `continuous_complex`. `--dtw-kirito` selects the rolling binary control. `--ddtw` selects derivative DTW. `--window`, `--band`, `--min-slope`, `--plateau-max`, and `--patience` override the corresponding monitor defaults. `--no-stagnation` keeps the exploit profile. JSON is printed to stdout and includes the search result, profile changes, diagnostics, effective config, and selected instance metadata (`path`, `index`, `n`, `m`, `reference`, `reference_source`).

## Checks

The focused check for this package is:

```text
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -p "test_*woa_abc*.py" -v
```

All matching tests cover the current package, including mode separation, original variant clocks, replay, and isolated package-import boundaries.

## Single-run results

One local command writes a fresh results folder for one functional check. It does not compare algorithms, and it does not download data.

```text
PYTHONDONTWRITEBYTECODE=1 python -m woa_abc.benchmark --instance instancias/mknapcb1.txt --instance-index 0 --seed 43 --population-size 30 --iterations 1000
```

Defaults are seed 43, population 30, 1000 hybrid cycles, index 0, and policy `binary_complex` (no variant). Pass `--policy` and `--dtw-kirito` as in the solver CLI. The detector settings are the DDTW settings copied from `hybrid_mkp/batch_benchmark.py`: window 40, band 0 resolved to the auto band 4, min slope 0.1, plateau 15, patience 8, adaptive percentiles 30/70, and improvement tolerance 0. A header reference of 0 is resolved from `mkcbres.txt` (see References).

One checked run per variant uses the same instance, seed, population, horizon, and evaluation cap. Run them separately. This command still writes one folder and does not grow a multi-run option.

```text
PYTHONDONTWRITEBYTECODE=1 python -m woa_abc.benchmark --instance instancias/mknapcb1.txt --instance-index 0 --seed 43 --population-size 30 --iterations 1000 --max-evaluations 180030 --variant v1
```

Repeat with `--variant v0`, `v2`, `v3`, and `v4` for the other recovery-clock variants. Omit `--variant` for default policy mode. Do not pass `--scout-limit` when the check is the `v1` default of 100.

The folder is `woa_abc/resultados/run_ddtw_<timestamp>/<file stem>/inst_<index>/`. The variant and policy are not part of that path. Read `manifest.json` field `effective_config` and `runs/run_01.json` fields `metadata.config.variant`, `metadata.config.policy`, `profile_changes`, `scout_limit`, `evaluations`, `evaluations_by_phase`, `metadata.config.max_evaluations`, `best_value`, `best_solution`, `gap`, and `termination_reason`. Variant mode always records `variant`, including `v0`. The timestamp directory must not already exist. The completed seed-43 run `woa_abc/resultados/run_ddtw_20261019_071137/` was relocated here and was not rerun. It predates this stagnation reaction.

| File | What it is |
|------|------------|
| `manifest.json` | Effective solver config, input path/hash/index/reference/reference_source, timing, actual counts, output completeness |
| `runs/run_01.json` | Full solver result, written before plots |
| `runs_resultados.csv` | One row. `n_switches` is 0. `n_profile_changes` and `policy` describe the reaction. `reference` and `reference_source` give the gap's reference |
| `resumen_pipeline.txt` | Single-run note with the known optimum and its source. No superiority claim |
| `best_run_01/` | History (with profile per cycle), detail, and the GWO–DE plot set below. `dtw_delta.png` is a required artifact |

## Plots

The plots are the GWO–DE plots. `woa_abc/plots/` is a copy of `gwo_de_dtw/plots/` from branch `feat/gwo-de-dtw-hybrid` (`__init__.py`, `_common.py`, `convergence.py`, `dtw_delta.py`, `parameters.py`, `temporal.py`, `comparison.py`). That package lives on another branch, so it is copied, not imported. Figure sizes, fonts, colors, change lines, legends, and the segment-by-profile logic are unchanged.

| File | What it shows |
|------|---------------|
| `fitness_convergence.png` | Best fitness per cycle, one colored segment per profile turn, a dashed line where each turn starts, and the optimum line when the reference is positive |
| `dtw_delta.png` | Monitor delta per cycle, segmented and colored by profile like the convergence plot. Always written: without a ready window it is an empty diagnostic that says "No DTW delta: no complete window" (run shorter than the window) or "No DTW delta: detection off" |
| `parameters.png` | Two panels: `a (WOA)` and `rho (ABC)` per cycle over a background shaded by profile. `G3` is recorded but not plotted |
| `temporal_diagnostics.png` / `.pdf` | Four panels: best fitness and profile used, a rising and a flat window against their ramp and constant baselines, distances and thresholds, and delta with the controller decision |
| `temporal_series.npz` / `.csv`, `temporal_metadata.json` | The series behind the temporal panels. The monitor is replayed from the saved config and checked against every recorded observation before anything is written |
| `convergence_comparison.png` / `.pdf` | Group run only: four policy lines or five variant lines (never mixed), with the common positive optimum reference |

Policy-mode drawing retains the GWO–DE style. Variant mode uses base/recovery segment labels and the same blue/orange colors, with actual a/rho schedules; temporal replay resets on every fire (including unapplied terminal fires), not on policy changes. The temporal decision means recovery active rather than exploit/explore. Repeated recoveries remain visible in the schedules and exported changes. `parameters.png` has panels `a` and `rho` instead of `a`, `F`, and `CR`. The parameter and temporal titles say WOA–ABC. Axes that say "Generation" in GWO–DE say "Cycle". The convergence and delta plots keep the GWO–DE titles and the "Accumulated Iterations" axis, because GWO–DE uses those texts unchanged. `PROFILE_COLORS` lives in `woa_abc/profiles.py`.

GWO–DE has no boxplot, so `boxplot_runs.png` is no longer written. The old parameter-profile plot is replaced by `parameters.png`.

`woa_abc.plots.plot_record` turns a WOA–ABC result, or its JSON, into the run record the plots read:

- Series have one entry per complete cycle. A partial final cycle, stopped by the evaluation cap, is left out because it never updated the monitor. `best_fitness` is still the archived best.
- The monitor entry of cycle `g` is the diagnostic of cycle `g`. Its `profile` is the decision taken after `g`, used from `g + 1`.
- With the detector on, a complete cycle without its diagnostic is an error.
- With the detector off there is no monitor output. The temporal outputs are skipped. Convergence, parameters, and the empty delta diagnostic are still written.

## Four-policy group run

This command mirrors `gwo_de_dtw/run.py`. It runs `binary_simple`, `binary_complex`, `continuous_simple`, and `continuous_complex` one after the other, with the same settings and seed.

```text
PYTHONDONTWRITEBYTECODE=1 python -m woa_abc.run --instance instancias/mknapcb1.txt --index 0 --seed 1 --pop 30 --iterations 1000 --out woa_abc/resultados/mknapcb1_0_s1.json
```

The group folder holds one `<policy>/` folder per policy and `convergence_comparison.png`/`.pdf`. Each policy folder has `result.json`, the full WOA–ABC result, plus the plot files above.

Destination rules are the GWO–DE rules:

- `--out X.json` writes to the folder `X/`.
- Any other `--out` path is the folder itself.
- Without `--out`, a new folder `<instance>_<index>_s<seed>_<timestamp>` is created in `woa_abc/resultados/`.
- A destination that is not a folder, or is not empty, is refused before anything runs.

The monitor defaults are the single-run settings above. `--signal`, `--window`, `--band`, `--plateau-max`, `--patience`, and `--dtw-kirito` override them. `--index` and `--pop` also accept `--instance-index` and `--population-size`. `--max-evaluations` and `--scout-limit` work as in the solver CLI. Group mode is chosen by `--reaction policy|variant`; it has no individual `--variant` or `--policy` flag. `--dtw-kirito` is rejected in variant mode.

## Five-variant group run

```text
PYTHONDONTWRITEBYTECODE=1 python -m woa_abc.run --reaction variant --instance instancias/mknapcb1.txt --index 0 --seed 1 --pop 30 --iterations 1000 --max-evaluations 180030 --out woa_abc/resultados/variants_s1
```

This runs v0, v1, v2, v3, v4 separately with the same seed/settings. Each `<variant>/` contains `result.json` and exactly the same plot suite as a policy folder (including ALWAYS `dtw_delta.png`, even before window readiness), plus `convergence_comparison.png/.pdf` at group level. No policy is run or combined. Both modes preserve the optimum reference.

## Campaign of 135 runs

One command runs indices 0, 1, and 2 of `mknapcb1.txt` through `mknapcb9.txt` for separate variants `v0`–`v4`, without policies. That is 135 sequential runs (27 instances × 5 variants), one run per selection, in three groups of 45. It does not download data, repeat a selection 31 times, and it claims no superiority from n=1.

```text
python -m woa_abc.campaign --instances-dir instancias
```

## Quick path

```text
python -B -m woa_abc.campaign --instances-dir instancias
```

Defaults are seed 43, population 30, 1000 hybrid cycles, an explicit evaluation cap of 180030, and reaction `variant`. Mixed policy flags are rejected before allocation. For a separate policy campaign use `--reaction policy` (all four, 108 runs), or also `--policy binary_complex` (one policy, 27 runs); `--dtw-kirito` is policy-only. Output is a new `woa_abc/resultados/campaign_<timestamp>/` directory. Old result folders are left in place.

## Details

| Topic | Decision |
|-------|----------|
| Files | `mknapcb1.txt` … `mknapcb9.txt`, indices 0, 1, and 2 only |
| Variants | `v0`–`v4`, one export each; reaction and variant are recorded even for v0 |
| Policy | Separate `--reaction policy` campaign only; no variants |
| Groups | Loaded `n`, checked against the file: 1, 4, 7 → 100; 2, 5, 8 → 250; 3, 6, 9 → 500 |
| Budget | The cap is the explicit 180030 default. A variant may finish before 1000 cycles |
| Reference | That index's header value, else its `mkcbres.txt` entry, else 0 (unknown). Recorded with `reference_source`. Do not copy index 0 onto index 1 or 2 |
| Collision | Refuse an existing campaign directory or destination. Do not overwrite it |
| Failure | Stop, keep the raw artifacts, do not rerun, and do not mark the campaign completed. No superiority claim is added |

Each variant folder holds the single-run artifacts directly: `manifest.json`, `runs/run_01.json`, `runs_resultados.csv`, `resumen_pipeline.txt`, and `best_run_01/` with the GWO–DE plots. The campaign also writes `experiment.json`, `campaign_results.csv`, and `campaign_summary.json` at the top, plus `group_summary.csv` inside `100/`, `250/`, and `500/`. The `profile_changes` column counts policy profile changes; `recoveries` counts applied variant restarts. For a same-instance comparison PNG/PDF use the group commands above.

## Checklist

- [ ] 135 start lines and 135 completion lines, or a failed stop with the raw JSON still present
- [ ] `campaign_summary.json` says `"superiority_claim": false` and `"status": "completed"` only after 135 variant completions
- [ ] Each run's manifest shows the resolved `reference` and its `reference_source`

## Next step

Inspect `campaign_summary.json` before comparing variants. One run is a functional record, not a ranking.

A hybrid cycle is not a legacy single-metaheuristic iteration. `n_profile_changes` counts exploit/explore changes, not algorithm switches. If plotting fails, the raw JSON stays and the process exits nonzero without a second solve.
