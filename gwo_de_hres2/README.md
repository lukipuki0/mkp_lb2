# GWO–DE with DTW/DDTW profile control on HRES2 (WPEB)

Runs the GWO–DE hybrid of `gwo_de_dtw/` on the HRES2 problem of `HRES2-H2/` (minimize LCOE of a wind–PV–electrolyzer–battery system). Both packages are imported, never edited.
Design contract: `odd/tasks/gwo-de-hres2.md` (repository root).

## Usage

Run from the repository root with an interpreter that has `pandas` (the HRES2 model needs it), e.g. `.venv/Scripts/python.exe`:

```
python -m gwo_de_hres2.run [--policies p1,p2] [--signal ddtw|dtw] [--pop-size N] [--iterations T] [--seed S] [--output-dir DIR]
```

- `--policies`: comma list from the six `gwo_de_dtw` campaign policies (`binary_simple`, `binary_complex`, `continuous_simple`, `continuous_complex`, `vanilla_exploitation`, `vanilla_exploration`); default all six. An unknown name exits with code 2 (`unknown policy: <name>`).
- `--output-dir`: default `gwo_de_hres2/results/<signal>_s<seed>_<timestamp>`. A non-empty directory is rejected with exit code 2 (`output directory is not empty`) before any run.

Writes `manifest.json` (configuration, monitor, model config, bounds, runtime) and `runs.jsonl`, one record per policy, written as each run finishes: objective, feasibility, physical LCOE/LCOH/AGSR, solution, decoded capacities, evaluations, elapsed time, and per-generation series (`best_series`, `profile_series`, `parameter_series`, `intensity_series`, `changes`, `step_stats`).

## Campaign (31 seeds)

```
python -m gwo_de_hres2.campaign [--signals dtw ddtw] [--policies ...] [--seeds 1 2 ...] [--pop-size N] [--iterations T] [--out DIR] [--dry-run]
```

Defaults: both signals, the six policies, seeds 1..31 and the settings below, i.e. 2 × 6 × 31 = 372 runs. Unique seeds, `--pop-size >= 4`, `--iterations >= 1` and an absent or empty `--out` (default `gwo_de_hres2/results/campaign_<timestamp>`) are checked before anything is written; `--dry-run` prints the plan and stops.

Output folder:

| File | Content |
|---|---|
| `campaign_config.json` | settings, monitor per signal, total/completed runs, status (`running`, `completed`, `failed`) |
| `runs.csv`, `runs.jsonl` | one row/record per run (no per-generation series), written as each run finishes |
| `statistics.csv` | per signal and policy: runs, feasible runs, best, worst, mean, median, sample sd, best seed, mean time |
| `tests.csv` | two-sided Mann-Whitney U, Holm-adjusted per family: each signal (all 15 policy pairs) and `dtw_vs_ddtw` (each policy across signals) |
| `resumen.txt` | Spanish summary: ranking per signal and Holm-significant pairs |
| `<signal>/<policy>/best_run.json` | full record of the best run (lowest objective, ties to the lower seed) |
| `<signal>/convergence_comparison.png` | best-so-far LCOE of the best run of each policy |

Statistics use the objective (LCOE when feasible, the penalty otherwise; `feasible_runs` shows how many were feasible). A pair with identical samples has no p-value and stays out of its Holm family. A failing run stops the campaign with status `failed` and exit code 1; the rows already written stay, but no summary is produced.

## Defaults (`config.py`)

| Setting | Default |
|---|---|
| population `N` / generations / seed | 30 / 1000 / 1 |
| signal | DDTW (`--signal dtw` for DTW) |
| monitor | HRES2 paper nominal: W 40, band 4, `min_slope` 0.1, `plateau_max` 15, `patience` 8, adaptive thresholds 30/70 |
| profiles | from `gwo_de_dtw.config`: exploit `a=0.5, F=0.5, CR=0.9`; explore `a=2.0, F=0.9, CR=0.3` |

The HRES2 dispatch is compiled with numba: the first evaluation takes about 1 s (JIT), later ones about 0.25 ms, so N30 × 1000 generations (60 030 evaluations) takes on the order of 15–25 s per policy.

## How it differs from `gwo_de_dtw`

- No binarization. The population lives in `[-1, 1]^4` and is mapped linearly to the HRES2 bounds `[0, 10, 0, 0]..[200, 20, 50, 2]` (wind MW, electrolyzer units, battery MW, duration index); `HRES2Function` decodes the discrete parts. This lets `gwo_de_dtw.gwo.gwo_candidate` and `gwo_de_dtw.de.de_candidate` run unchanged with `v_max = 1`.
- Minimization: the lower objective wins and replaces the individual when `<=` the current one; leaders are the three lowest. The objective is `HRES2Function._evaluate`: LCOE when feasible, `100 + 10·AGSR` otherwise.
- The monitor maximizes, so it observes `-best` each generation (as the HRES2 metaheuristics do). The profile controller is `gwo_de_dtw.controller.ProfileController` unchanged; G3 is unused.
- Competitive mode only. Cost: N evaluations at initialization, 2N per generation.
- Weather is the synthetic TMY of `HRES2-H2/wpeb_model.py`, built once per invocation; all policies share it and the seed.

## Tests

```
.venv/Scripts/python.exe -m unittest tests.test_gwo_de_hres2 tests.test_gwo_de_hres2_campaign
```
