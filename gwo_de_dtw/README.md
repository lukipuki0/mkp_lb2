# GWO–DE hybrid with DTW/DDTW profile control

Binary MKP hybrid: one shared population where GWO and DE compete for every individual, and the best-so-far fitness curve (DTW or DDTW) switches the joint parameter profile.
Design contract: `odd/tasks/gwo-de-dtw-hybrid.md` (repository root); background in `planes/GWO_DE_DTW_DDTW.md`.

## Configuration

All defaults live in `config.py`; `hybrid.py`, `controller.py` and `run.py` read them from there.

| Setting | Default |
|---|---|
| instance | `instancias/mknapcb1.txt`, index 0 |
| population `N` | 30 |
| iterations (generations) | 1000 |
| seed | 1 (one run for now) |
| binarization | `lb2` (`--binarization lb2\|sigmoid`) |
| continuous space | `[-100, 100]` with lb2, `[-6, 6]` with sigmoid |
| LB2 `G1`, `G2` | 0.5, 0.5 (fixed) |
| LB2 `G3` exploit / explore | 0.1 / 0.1, i.e. fixed G3 = 0.1 (`--g3-exploit`, `--g3-explore`) |
| monitor `W` / band | 40 / 0 (10% of `W` = 4) |
| monitor `min_slope` / `plateau_max` / `patience` | 2.0 / 15 / 8 |
| signal | DDTW |
| adaptive thresholds `p_low` / `p_high` | 40 / 60 |
| `improvement_tol` | 0.0 |

## Sources

- Monitor: imported from `dtw_stagnation.py` at the repository root (the same module `hybrid_mkp` uses); run from the repository root so it is importable.
- `problem.py`: instance parsing, density and repair ported from `hybrid_mkp/mkp_core` (see below).

## How one generation works

1. Copy alpha, beta and delta (the three best individuals).
2. For every individual `i`, build two real candidates:
   - GWO (`gwo.py`): average of the three leader-guided positions, using `a`.
   - DE/rand/1/bin (`de.py`): `x_r1 + F (x_r2 - x_r3)` with binomial crossover `CR`.

   `hybrid.py` only combines both operators over the shared population.
3. Each candidate is binarized: with LB2 over the current bits of `i` (two evaluations each), or with sigmoid (one evaluation each); see below.
4. The better candidate replaces `i` when its fitness is `>=` the current one; real and binary vectors are written together.

Initial population: real vectors uniform in `[-100, 100]` (lb2) or `[-6, 6]` (sigmoid). With lb2 the initial bits are the positive coordinates, repaired; with sigmoid they are sampled from `sigmoid(x_real)`, repaired. One evaluation each.

## Repair

Same as `hybrid_mkp/mkp_core/repair.py`. Item density is `min_i(p_j / r_ij)`; infinite ratios (zero weight) become 10 × the largest finite ratio.

1. Expulsion: from lowest to highest density, drop selected items until the solution is feasible.
2. Insertion: from highest to lowest density, add every item that still fits; items that do not fit are skipped.

The known optimum for the gap is the third number of the instance header. In `instancias/mknapcb*.txt` it is 0, so the best known value of each of the 270 instances is read from `instancias/mkcbres.txt`, the unmodified OR-Library file (https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/mkcbres.txt, "Best Feasible Solution Value" table of Chu & Beasley 1997). Names map as mknapcb1..9 = 5.100, 5.250, 5.500, 10.100, 10.250, 10.500, 30.100, 30.250, 30.500, so `mknapcb1.txt` index 1 is `5.100-01`. These are best known values, not certified optima. If `mkcbres.txt` is not next to the instance file, the gap is `n/a`.

## LB2 binarization

Port of `lb2/binarization.py` in `binarization.py`. The real vector `v` does not produce bits from scratch; it gives the probability of flipping each current bit:

```text
L1 = clip(-G1 * v / (v_max - G2) + G3, 0, 1)
L2 = clip( G1 * v / (v_max - G2) + G3, 0, 1)
```

One candidate flips with `L1`, another with `L2`; both are repaired with `problem.reparar` and the better one is kept (L2 on ties). Defaults `G1 = G2 = 0.5`, `v_max = 100`. `G3` is the active profile's base flip probability (default 0.1 in both profiles, so G3 is fixed at 0.1). Exploit 0.1 / explore 0.3 was tried and produced long exploration episodes without improvement in the tested seeds (Log L32). `G2 >= v_max` is rejected because the slope would change sign. Flips use the run's seeded generator.

## Sigmoid binarization

Selected with `--binarization sigmoid` (`config.BINARIZATION` keeps `lb2` as default). Each bit is sampled with `P(bit = 1) = 1 / (1 + exp(-v))` using the run's seeded generator and then repaired with `problem.reparar`: one evaluation per candidate. The continuous space is `[-6, 6]` (`config.SIGMOID_V_MAX`) for initialization and every clip. Sigmoid does not use `G1`, `G2` or `G3`. `--g1`, `--g2`, `--g3-exploit`, `--g3-explore` and the `G2 < v_max` check apply only to lb2. `G3` is still recorded in `parameter_series` for every generation, including sigmoid runs.

| Profile | `a` (GWO) | `F` (DE) | `CR` (DE) | `G3` (LB2) |
|---|---|---|---|---|
| exploit (start) | 0.5 | 0.5 | 0.9 | 0.1 |
| explore | 2.0 | 0.9 | 0.3 | 0.1 |

## Stop and evaluations

A run stops after `--iterations` generations (default 1000). Evaluations are counted and reported, not used to stop: `N` for initialization plus, per generation, `4N` with lb2 (2 operators × 2 LB2 candidates) or `2N` with sigmoid (2 operators × 1 candidate). With `N=30` and 1000 iterations, a run uses 120030 evaluations with lb2 and 60030 with sigmoid.

## Profile control

The monitor (`dtw_stagnation.StagnationMonitor`, DDTW by default) looks at the last `W` best-so-far values.

- exploit + `fire` → explore.
- explore → exploit when a ready observation has `no_improve_len < plateau_max`; otherwise stay in explore.
- A change decided after generation `g` applies from `g+1`.
- Fresh window: after a change the monitor is replaced (window, plateau counter, streak, thresholds); no decision until `W` new generations. The return happens on that first ready observation when it shows recent improvement.
- Global best and population are kept.

## Usage

From the repository root:

```bash
python -m gwo_de_dtw.run --instance instancias/mknapcb1.txt --index 0 \
    --binarization lb2 --signal ddtw --window 40 --pop 30 --iterations 1000 --seed 1 \
    --out gwo_de_dtw/results/mknapcb1_0_ddtw_s1.json
```

Every option defaults to `config.py`, so `python -m gwo_de_dtw.run` alone runs the same configuration.

Output:

```text
instance=mknapcb1.txt[0] binarization=lb2 signal=ddtw W=40 pop=30 iterations=1000 seed=1
best=<fitness> known_optimum=<best known> gap=<pct>% evaluations=<used> generations=<g> changes=<count>
```

`--out` stores the full record: series of best fitness, profile per generation, parameters `a`, `F`, `CR` and `G3` per generation, change generations, monitor outputs and GWO/DE win counts per generation. Run `python -m gwo_de_dtw.run --help` for all options. `--g3` is not an option; use `--g3-exploit` and `--g3-explore`.

## Plots

Written automatically when `--out` is given, after the JSON, into `<out without suffix>_plots/` (`results/x.json` → `results/x_plots/`). Without `--out` nothing is written. Four PNGs plus temporal diagnostic data (code in `plots/`, colors per profile in `config.PROFILE_COLORS`):

| File | Content |
|---|---|
| `fitness_convergence.png` | best-so-far fitness, colored by profile; red dashed line at the best known value |
| `dtw_delta.png` | monitor delta (DTW or DDTW) where the window is ready, connected without gaps; skipped when no window was ready |
| `parameters.png` | `a` (GWO), `F` and `CR` (DE) used in every generation (x = generation 1..T), background shaded by profile. `G3` is recorded in the JSON (`parameter_series`) but not plotted. |

`fitness_convergence.png` and `dtw_delta.png` follow `hybrid_mkp/plots` exactly (code copied, not imported): same title, labels, x = accumulated iterations from 0, dashed line at the start of every segment (only with fewer than 50 segments), same `[plot]` line. The only difference: hybrid_mkp's segments are metaheuristic turns, here they are profile turns (consecutive generations in exploit or explore).

### Temporal diagnostics

Every run with `--out` also writes `temporal_diagnostics.png` and `.pdf` in the same plot directory, for both DTW/DDTW and LB2/sigmoid. Adapted from `fig_temporal_diagnostics (2)/temporal_diagnostics.py`, without importing that project or changing the optimizer.

Four synchronized rows show: (a) best fitness and the profile actually used; (b) a rising and a flat window against the ramp and constant references, each normalized to its own start; (c) distances and thresholds; (d) delta and the decision after each generation, applied at the next generation. Parameters a/F/CR are identified; G3 is not plotted. The rising exploitation window whose total increase is closest to its ramp increase is selected (fallback: any rising window; earliest on ties); the flat window is the midpoint of the longest flat exploration run (fallback: any flat run). Missing categories are labeled instead of fabricated.

`temporal_series.npz` stores X/R/C windows and scalar series; `temporal_series.csv` stores scalars; `temporal_metadata.json` stores configuration and interpretation. The run JSON now includes `monitor_config`. Diagnostics replay only the monitor with that saved configuration and verify each output and distance before writing. Fresh-window warm-up remains NaN in the exports, including after switches. Panels (c) and (d) visually connect available samples across those gaps; connecting segments are not measured values. Panels (c) and (d) show distances, delta and thresholds only at samples where adaptive thresholds are active (the tenth ready sample onward, after every fresh-window reset). This replaces the earlier 2W visual cutoff. All samples, including initial fixed thresholds, remain in the exports; the optimizer and controller are unchanged. Panel (b) follows the reference colors: observed black (#1a1a1a), ramp blue (#0072B2), constant dashed orange (#D55E00). Fixed initial thresholds are retained, with an adaptive-threshold flag in the exports. Old JSONs without `monitor_config` are explicitly skipped rather than guessing their settings.

Tests: `python -m unittest tests.test_gwo_de_dtw tests.test_gwo_de_temporal` from the repository root.

## Experimental evidence and delivery scope

These are exploratory runs on mknapcb1[0], not a completed multi-seed benchmark.
All rows below use LB2, N=30, 1000 generations and fixed G3=0.1.

| Signal / window / percentiles | Seed | Explore CR | Best | Gap |
|---|---:|---:|---:|---:|
| DDTW / 40 / 30-70 | 1 | 0.3 | 24243 | 0.57% |
| DTW / 40 / 30-70 | 1 | 0.3 | 24163 | 0.89% |
| DTW / 40 / 30-70 | 1 | 0.7 | 24169 | 0.87% |
| DTW / 40 / 30-70 | 42 | 0.3 | 23964 | 1.71% |
| DTW / 40 / 30-70 | 42 | 0.7 | 23849 | 2.18% |
| DDTW / 200 / 30-70 | 1 | 0.3 | 24106 | 1.13% |
| DDTW / 40 / 40-60 (final defaults) | 1 | 0.3 | 24243 | 0.57% |

Run records and plots are preserved in `results/`, including superseded experiments (counter-only reset, bidirectional patience, variable G3). Those experimental controller modes are not supported by the final code. Historical records predate some metadata additions and must not be interpreted using current defaults. No ranking across configurations is established by one or two seeds.

Latest default record: `results/mknapcb1_0_ddtw_s1_w40_p40_60.json`; its `_plots/` directory contains the current temporal figure. The 31-seed campaign and other-instance comparisons are deferred.

The feature commit includes only `gwo_de_dtw/`, its two test modules, `instancias/mkcbres.txt`, and the feature task document. Root monitor and reference implementations remain unchanged. Python caches and unrelated presentations are excluded. Native review assessment was unavailable because an unrelated untracked directory contains a nested Git repository; functional verification is reported separately, not as review approval.
