# GWO–DE hybrid with DTW/DDTW profile control

Binary MKP hybrid: one shared population where GWO and DE compete for every individual (an opt-in sequential GWO → DE mode exists for experiments), and the best-so-far fitness curve (DTW or DDTW) sets the joint parameter profile through one of four policies. Two fixed baselines, `vanilla_exploitation` and `vanilla_exploration`, keep one profile for the whole run and are included in `run.py` and in the campaign.
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
| policy (`run_hybrid` / `ProfileController`) | `binary_complex`; `run.py` and the campaign run all six, including the two vanilla baselines |
| `SCALE` / `K` / `CENTER` (continuous policies) | 2.0 / 5.0 / 0.5 (no CLI options) |
| binary control `DTW_KIRITO` | `False` = fresh window (`--dtw-kirito` / `--no-dtw-kirito`) |
| `HYBRID_MODE` | `competitive` (`--hybrid-mode competitive\|sequential`; see below) |

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

This is the default `competitive` mode. `step_stats` stores, per generation, `gwo_wins` and `de_wins` (which candidate was better for each individual, they add up to `N`) and `accepted` (individuals replaced, at most `N`).

### Sequential mode (experimental)

`--hybrid-mode sequential` (`hybrid_mode="sequential"` in `run_hybrid` / `GWODEHybrid`) is an opt-in experiment; the default stays `competitive` (`config.HYBRID_MODE`), and `campaign.py` has no option for it. GWO and DE do not compete; each generation runs:

1. Copy alpha, beta and delta (frozen for the whole generation, as in competitive).
2. GWO sweep: for every individual `i`, one GWO candidate, binarized as above over the current bits of `i`; it replaces `i` (real, bits and fitness together) when its fitness is `>=` the current one. A rejected candidate leaves `i` as it was.
3. Global best updated.
4. DE sweep: for every individual `i`, one DE/rand/1/bin candidate built from the population as it is after the GWO sweep, binarized over the current bits of `i`, with the same `>=` replacement. The sweep works in place, as competitive does: DE donors and the base row also include DE candidates already accepted earlier in this sweep.
5. Global best updated; the monitor observes the best-so-far once, after the whole generation.

One profile (`a`, `F`, `CR`, `G3`) is active for the whole generation, both sweeps. Cost is the same as competitive: `N` initialization, `4N` per generation with lb2 and `2N` with sigmoid. `step_stats` uses its own keys: `gwo_accepted` and `de_accepted` (candidates accepted in each sweep, at most `N` each) and `accepted_total = gwo_accepted + de_accepted` (at most `2N`, an individual can be replaced in both sweeps). Every run record stores `hybrid_mode`; the CLI header ends with ` hybrid_mode=sequential` only in that mode. Plots do not read `step_stats` and are the same files for both modes.

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

## Profile control: four policies

The monitor (`dtw_stagnation.StagnationMonitor`, DDTW by default) looks at the last `W` best-so-far values. Four policies (`policies.py::POLICIES`, named after the `DTW_optimization-main` folders) read the same monitor with the same `StagnationConfig` and the same exploit/explore profiles. Before the window is ready nothing is decided and the run stays in exploit. A change decided after generation `g` applies from `g+1`. Global best and population are never reset.

| Policy | exploit → explore | explore → exploit | Monitor after a change |
|---|---|---|---|
| `binary_simple` | `D2_vs_const <= theta_c` | ready observation with `no_improve_len < plateau_max` | replaced (fresh window) |
| `binary_complex` | monitor `fire` (plateau + constant + ramp conditions, `patience` in a row) | same | replaced (fresh window) |
| `continuous_simple` | blended, see below | blended | never replaced |
| `continuous_complex` | blended, see below | blended | never replaced |

Binary policies switch between the two fixed profiles. Fresh window: after a change the monitor is replaced (window, plateau counter, streak, thresholds); no decision until `W` new generations. In explore, the return happens on a ready observation that shows recent improvement; otherwise the run stays in explore. `binary_complex` is the original controller of this package, unchanged, and the library default (`config.POLICY`).

`vanilla_exploitation` and `vanilla_exploration` are fixed baselines, accepted by `run_hybrid` and `ProfileController` but not included in `POLICIES`. The first uses the exploit profile at every generation, the second the explore profile, from generation 0. The monitor is still recorded and never changes the profile, so `changes` stays empty. `run.py` and `campaign.py` both iterate `CAMPAIGN_POLICIES`: the four adaptive policies followed by the two baselines. `POLICIES` remains the four adaptive names.

With `--dtw-kirito` (`config.DTW_KIRITO = True`, `dtw_kirito=True` in `run_hybrid` / `ProfileController`) the binary policies use the Kirito rolling control instead: one monitor for the whole run (never replaced), and on every ready observation the profile is explore if the policy trigger holds (`D2_vs_const <= theta_c` or `fire`) and exploit otherwise, with or without recent improvement. The GWO/DE operators, LB2 and the profiles are the same. Continuous policies are not affected. Default is off (fresh window). Every run record stores `dtw_kirito`; the header line ends with ` dtw_kirito=True` only when it is on. Temporal diagnostics replay `dtw_kirito` runs without resets; older JSONs that store the flag as `dtw_main` are read the same way, and JSONs without either key are replayed as fresh-window runs.

Continuous policies never replace the monitor. On every ready observation they compute an intensity in `[0, 1]` (0 before the first ready observation):

```text
continuous_simple:  intensity = 1 - clip(D2_vs_const / (theta_c * SCALE + 1e-10), 0, 1)
continuous_complex: intensity = 1 / (1 + exp(-K * (delta / (theta_delta + 1e-10) - CENTER)))
```

`SCALE = 2.0`, `K = 5.0`, `CENTER = 0.5` (`config.py`). The active profile is the linear blend `(1 - intensity) * exploit + intensity * explore` of `a`, `F`, `CR` and `G3`, named `explore` when intensity > 0.5 and `exploit` otherwise; `changes` records the generations where that name changes. A negative `theta_delta` is used as is, as in `main` (`DTW_optimization-main`): the ratio `delta / theta_delta` then flips sign, so a negative `delta` gives a high intensity.

The run record adds `policy` and `intensity_series` (intensity of the profile used at each generation; `null` for binary policies); every monitor entry adds `intensity` (decision for the next generation; `null` for binary policies). `parameter_series` stores the blended `a`, `F`, `CR` and `G3` actually used.

## Usage

From the repository root:

```bash
python -m gwo_de_dtw.run --instance instancias/mknapcb1.txt --index 0 \
    --binarization lb2 --signal ddtw --window 40 --pop 30 --iterations 1000 --seed 1 \
    --out gwo_de_dtw/results/mknapcb1_0_ddtw_s1.json
```

Same configuration with the sequential GWO → DE experiment (written to a new folder; existing results are not touched):

```bash
python -m gwo_de_dtw.run --instance instancias/mknapcb1.txt --index 0 \
    --binarization lb2 --signal ddtw --window 40 --pop 30 --iterations 1000 --seed 1 \
    --hybrid-mode sequential --out gwo_de_dtw/results/mknapcb1_0_policies_s1_sequential
```

Same configuration with the Kirito rolling binary control:

```bash
python -m gwo_de_dtw.run --instance instancias/mknapcb1.txt --index 0 \
    --binarization lb2 --signal ddtw --window 40 --pop 30 --iterations 1000 --seed 1 \
    --dtw-kirito --out gwo_de_dtw/results/mknapcb1_0_policies_s1_dtw_kirito
```

Every option defaults to `config.py`, so `python -m gwo_de_dtw.run` alone runs the same configuration. The runner always runs the six policies of `CAMPAIGN_POLICIES`, on the same instance and settings, each with its own hybrid seeded with `--seed`. There is no `--policy` option.

Output:

```text
instance=mknapcb1.txt[0] binarization=lb2 signal=ddtw W=40 pop=30 iterations=1000 seed=1
policy=binary_simple best=<fitness> known_optimum=<best known> gap=<pct>% evaluations=<used> generations=<g> changes=<count>
policy=binary_complex best=...
policy=continuous_simple best=...
policy=continuous_complex best=...
  [plot] convergence_comparison.png -> '<group>/convergence_comparison.png'
output=<group>
```

Each policy's `[plot]` lines follow its `policy=` line. Every invocation writes one group folder with the four runs:

```text
<group>/
  binary_simple/result.json        full run record
  binary_simple/*.png, temporal_*  its plots (see Plots)
  binary_complex/...
  continuous_simple/...
  continuous_complex/...
  convergence_comparison.png       six policies on one axes
  convergence_comparison.pdf
```

- `--out X.json` → group folder `X/` (`results/x.json` → `results/x/binary_simple/result.json`, ...).
- `--out X` with any other suffix or none → `X` itself is the group folder.
- No `--out` → a new folder `<instance>_<index>_s<seed>_<YYYYmmdd-HHMMSS>` in `gwo_de_dtw/results/` (e.g. `gwo_de_dtw/results/mknapcb1_0_s1_20261031-153000/`), relative to the working directory, so run from the repository root. If that name already exists, `_2`, `_3`, ... is appended.
- Old results are never overwritten: a destination that exists and is not empty, or is not a folder, is rejected (`error: output directory is not empty: <group>` / `error: output path is not a directory: <group>`, exit code 2) before the instance is loaded or anything runs. An existing empty folder is used.

Paths are printed as given (the default one relative to the working directory). Each `result.json` stores the full record: series of best fitness, profile per generation, parameters `a`, `F`, `CR` and `G3` per generation, change generations, monitor outputs, `hybrid_mode` and the per-generation `step_stats` (GWO/DE win counts in competitive mode, accepted candidates per sweep in sequential mode). Run `python -m gwo_de_dtw.run --help` for all options. `--g3` is not an option; use `--g3-exploit` and `--g3-explore`.

## Plots

Written automatically after each policy's `result.json`, into the same `<group>/<policy>/` folder. Four PNGs plus temporal diagnostic data (code in `plots/`, colors per profile in `config.PROFILE_COLORS`):

| File | Content |
|---|---|
| `fitness_convergence.png` | best-so-far fitness, colored by profile; red dashed line at the best known value |
| `dtw_delta.png` | monitor delta (DTW or DDTW) where the window is ready, connected without gaps; skipped when no window was ready |
| `parameters.png` | `a` (GWO), `F` and `CR` (DE) used in every generation (x = generation 1..T), background shaded by profile. `G3` is recorded in the JSON (`parameter_series`) but not plotted. |

`fitness_convergence.png` and `dtw_delta.png` follow `hybrid_mkp/plots` exactly (code copied, not imported): same title, labels, x = accumulated iterations from 0, dashed line at the start of every segment (only with fewer than 50 segments), same `[plot]` line. The only difference: hybrid_mkp's segments are metaheuristic turns, here they are profile turns (consecutive generations in exploit or explore).

### Convergence comparison

After the four runs, `<group>/convergence_comparison.png` and `.pdf` show the four `best_series` on one axes: x = generation 1..T, y = best-so-far fitness (maximization), raw values (not normalized). Fixed color per policy (`plots/comparison.py::POLICY_COLORS`): `binary_simple` blue `#0072B2`, `binary_complex` orange `#E69F00`, `continuous_simple` green `#009E73`, `continuous_complex` purple `#CC79A7`; legend labels are the policy names. The red dashed best-known line of `fitness_convergence.png` is drawn only when all four records have the same positive `known_optimum`.

The plot can be rebuilt from saved records without rerunning:

```python
import json
from pathlib import Path
from gwo_de_dtw.plots import plot_convergence_comparison
from gwo_de_dtw.policies import POLICIES

group = Path("gwo_de_dtw/results/<group>")
records = [json.loads((group / p / "result.json").read_text()) for p in POLICIES]
plot_convergence_comparison(records, group)  # returns [png, pdf]
```

`plot_convergence_comparison(records, output_dir)` draws one line per record, in the given order, labeled with its `policy` (a record without `policy` raises `ValueError`); it saves the PNG (dpi 150) and PDF, closes the figure and returns both paths. `convergence_comparison_figure(records)` returns the open figure instead. Both take an optional `seed_labels=False`; with `True` (used by the campaign) each label becomes `<policy> (seed <seed>)` and the title `Best run of each policy - <instance>` (a record without `seed` raises `ValueError`). The default plot of `run.py` is unchanged.

### Temporal diagnostics

Every run also writes `temporal_diagnostics.png` and `.pdf` in the same plot directory, for both DTW/DDTW and LB2/sigmoid. Adapted from `fig_temporal_diagnostics (2)/temporal_diagnostics.py`, without importing that project or changing the optimizer.

Four synchronized rows show: (a) best fitness and the profile actually used; (b) a rising and a flat window against the ramp and constant references, each normalized to its own start; (c) distances and thresholds; (d) delta and the decision after each generation, applied at the next generation. Parameters a/F/CR are identified; G3 is not plotted. The rising exploitation window whose total increase is closest to its ramp increase is selected (fallback: any rising window; earliest on ties); the flat window is the midpoint of the longest flat exploration run (fallback: any flat run). Missing categories are labeled instead of fabricated.

`temporal_series.npz` stores X/R/C windows and scalar series; `temporal_series.csv` stores scalars; `temporal_metadata.json` stores configuration (including `policy`) and interpretation. The `intensity` series (intensity used at `t`, NaN for binary policies) is the last CSV column and an NPZ key; for continuous policies it is also drawn in the decision panel, and the parameter description shows the min-max range of blended values. The replay resets the monitor after each change only for binary policies; continuous policies are replayed without resets. JSONs without `policy` are replayed as `binary_complex`. The run JSON now includes `monitor_config`. Diagnostics replay only the monitor with that saved configuration and verify each output and distance before writing. Fresh-window warm-up remains NaN in the exports, including after switches. Panels (c) and (d) visually connect available samples across those gaps; connecting segments are not measured values. Panels (c) and (d) show distances, delta and thresholds only at samples where adaptive thresholds are active (the tenth ready sample onward, after every fresh-window reset). This replaces the earlier 2W visual cutoff. All samples, including initial fixed thresholds, remain in the exports; the optimizer and controller are unchanged. Panel (b) follows the reference colors: observed black (#1a1a1a), ramp blue (#0072B2), constant dashed orange (#D55E00). Fixed initial thresholds are retained, with an adaptive-threshold flag in the exports. Old JSONs without `monitor_config` are explicitly skipped rather than guessing their settings.

## Campaign (31 seeds, best run per policy)

`campaign.py` runs every selected file × index × seed × policy with the same `run_hybrid` call as `run.py`. Both iterate `CAMPAIGN_POLICIES` (the four adaptive policies plus `vanilla_exploitation` and `vanilla_exploration`). Campaign defaults: `mknapcb1.txt` .. `mknapcb9.txt` from `instancias/`, indices 0 1, seeds 1..31, and every algorithm default of `config.py` (N=30, 1000 generations, DDTW, W=40, LB2, `DTW_KIRITO=False`): 9 × 2 × 31 × 6 = 3348 runs.

Full campaign (long; check it first with `--dry-run`):

```bash
python -m gwo_de_dtw.campaign --dry-run --out gwo_de_dtw/results/campaign_full
python -m gwo_de_dtw.campaign --out gwo_de_dtw/results/campaign_full
```

Small smoke run (8 runs, a few seconds plus plots):

```bash
python -m gwo_de_dtw.campaign --files mknapcb1.txt --indices 0 --seeds 1 2 \
    --pop 8 --iterations 20 --window 10 --out gwo_de_dtw/results/campaign_smoke
```

Options: `--instances-dir` (default `instancias`), `--files` (basenames inside it), `--indices`, `--seeds`, and the algorithm options of `run.py` (`--binarization`, `--signal`, `--window`, `--band`, `--plateau-max`, `--patience`, `--pop`, `--iterations`, `--g1`, `--g2`, `--g3-exploit`, `--g3-explore`, `--dtw-kirito`), plus `--out` and `--dry-run`.

Before the first run or write everything is checked and rejected with exit code 2: file names must be basenames (no folders or `..`) that exist in `--instances-dir`; files, indices and seeds must be unique and nonempty; every index must exist in every file; seeds nonnegative; `--pop >= 4`; `--iterations`, `--window`, `--plateau-max`, `--patience` positive; `--g2` below `v_max` with lb2; the output must not exist, or be an empty folder. Without `--out` a new `gwo_de_dtw/results/campaign_<YYYYmmdd-HHMMSS>` (`_2`, ... if taken) is created. Nothing is overwritten or deleted. `--dry-run` prints the header with the number of runs and stops without running or writing.

Layout of the campaign folder:

```text
<campaign>/
  campaign_config.json        settings, selection, total_runs, completed_runs, status, error
  resumen.txt                 Spanish readable summary, rewritten after each completed instance
  summary.csv                 every run of every instance
  statistics.csv              one row per instance and policy
  mknapcb1/inst_00/
    runs.csv                  every run of this instance (31 seeds x 4 policies = 124 rows)
    statistics.csv            4 rows, one per policy
    convergence_comparison.png/.pdf   best run of each policy, labeled with its seed
    binary_simple/best_run/result.json + its plots and temporal diagnostics
    binary_complex/best_run/...
    continuous_simple/best_run/...
    continuous_complex/best_run/...
  mknapcb1/inst_01/ ...
```

`runs.csv` / `summary.csv` columns (summary repeats the rows of every `runs.csv` on purpose): `file, index, policy, seed, status, best_fitness, known_optimum, gap_pct, evaluations, generations, changes, elapsed_s`. Each row is appended right after its run; `changes` is the number of profile label changes; `elapsed_s` is the wall time of the run. `known_optimum` and `gap_pct` are blank (not 0) when the best known value is unknown.

`statistics.csv` columns: `file, index, policy, runs, best_fitness, worst_fitness, mean_fitness, median_fitness, std_fitness, best_seed, known_optimum, best_gap_pct, mean_gap_pct, mean_elapsed_s`. `std_fitness` is the sample standard deviation (n - 1), blank with a single seed; gap columns are blank when the reference is unknown.

`resumen.txt` is a human-readable Spanish summary for the same campaign. The top has configuration, status and progress. Each completed instance then lists, per mode, best, worst, mean, sample standard deviation, winning seed and its gap, average time, and every seed ranked best to worst. A final section ranks the four modes inside each instance. The ranking uses the best run (highest fitness, then the lower seed; if both tie, campaign mode order). The mean is shown and does not decide that order. An unknown reference is `no disponible`, not zero. A single-run sample standard deviation is `no aplica`. The file is rewritten after each completed instance, and again when the campaign finishes or stops, by writing a temporary file and replacing `resumen.txt` in one step. An interruption leaves the previous readable version; the temporary file is not left behind.

Best run = highest `best_fitness`; ties go to the lower seed, whatever the `--seeds` order. Only the current best record of each policy is kept in memory; at the end of each instance the four winners are written to `<policy>/best_run/` with all their plots and the shared comparison plot is drawn. Other runs keep only their CSV row (no record, no plots).

`campaign_config.json` is rewritten after every run (`completed_runs`). `resumen.txt` is rewritten after each completed instance, not after every run. `status` is `running` while it runs, `completed` at the end (exit 0), `failed` when a run or export raises (exit 1, `error: campaign failed at <file>[<index>] policy=<p> seed=<s>: <Exception>: <message>` on stderr; the failing run gets a `failed` row with blank metrics) and `interrupted` on Ctrl+C. The campaign stops at the first failure; finished rows and instances stay as written, the unfinished instance has only `runs.csv`. There is no resume: start a new campaign in a new folder. A process killed without Python cleanup leaves `status: running`.

Tests: `python -m unittest tests.test_gwo_de_policies tests.test_gwo_de_dtw tests.test_gwo_de_temporal tests.test_gwo_de_control_switch tests.test_gwo_de_grouped_output tests.test_gwo_de_campaign tests.test_gwo_de_sequential` from the repository root.

## Cluster (Slurm)

`run_gwo_de_hpc.sh` at the repository root runs this campaign on the cluster as one job with the header of `run_parallel_hpc.sh` (CPU partition, 9 CPUs, 32G, QOS long). From the repository root:

```bash
mkdir -p .hpc/logs .hpc/errors        # once; Slurm does not create them
bash run_gwo_de_hpc.sh --dry-run      # print the 27 instances and commands
sbatch run_gwo_de_hpc.sh
```

The job runs the sizes in order 100, 250, 500. For each size it starts nine processes in parallel, one per instance (the three files of that size, indices 0, 1 and 2), and waits for all nine before the next size. Each process calls this campaign with `--files <file> --indices <i> --seeds 1 .. 31`, so it runs 124 runs and keeps the best run of each policy. Seeds stay in one process because the best run is chosen among the seeds of one invocation. A failed instance does not stop the others or the next size; the job exits 1 at the end and lists the failed instances.

Output: `resultados/gwo_de_<SLURM_JOB_ID>/mknapcb<F>/inst_<II>/` with `runs.csv`, `statistics.csv`, `convergence_comparison.png/.pdf`, `campaign_config.json`, `resumen.txt`, `run.log` and `<policy>/best_run/`. Each process runs in `resultados/gwo_de_<SLURM_JOB_ID>/.staging/mknapcb<F>_inst_<II>/` and moves its `mknapcb<F>/inst_<II>/` into the experiment when it ends well (the campaign root `summary.csv` and `statistics.csv` of a one-instance campaign repeat the instance files and are dropped). A failed instance keeps its `.staging` folder with `run.log`. Job log: `.hpc/logs/gwo_de_<jobid>.log`, errors in `.hpc/errors/`. There is no resume. `MKP_CONDA_SH` sets the path of `conda.sh` (environment `mkp_env`). Tests: `python -m unittest tests.test_hpc_campaign_scripts`.

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

Latest default record: `results/mknapcb1_0_ddtw_s1_w40_p40_60.json`; its `_plots/` directory contains the current temporal figure. The 31-seed campaign runner exists (`campaign.py`, see Campaign) but has not been run yet.

The feature commit includes only `gwo_de_dtw/`, its two test modules, `instancias/mkcbres.txt`, and the feature task document. Root monitor and reference implementations remain unchanged. Python caches and unrelated presentations are excluded. Native review assessment was unavailable because an unrelated untracked directory contains a nested Git repository; functional verification is reported separately, not as review approval.
