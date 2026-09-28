# Rotational DTW/DDTW sensitivity campaigns

Run isolated, reproducible sensitivity pilots for the manuscript's **rotational** MKP, CEC2022 and HRES2-H2 framework. This module reuses existing solvers; it does not run the separate WOA–ABC variant experiment or change its configuration.

## Quick path

Run these commands from the repository root:

```powershell
# Validate a protocol and inspect the number of tasks; creates no output files.
python -B -m sensitivity.run --config sensitivity/configs/rotational_pilot_1k.json --dry-run

# Tiny real-engine functionality check. NOT a scientific sensitivity study.
python -B -m sensitivity.run --config sensitivity/configs/smoke.json

# Resume the same campaign, skipping successful tasks and retrying failed ones.
python -B -m sensitivity.run --config sensitivity/configs/smoke.json --resume

# Generate summaries and figures from completed smoke records.
python -B -m sensitivity.analysis --campaign resultados/sensitivity/rotational_smoke

# Focused functional tests (no pytest dependency).
python -B -m unittest discover -s tests -p "test_sensitivity*.py" -v
```

Before launching the real pilot, review its case selection and settings. The 31-seed pilot can be expensive; execution is manual, never started by dry-run. Use `--limit N` to attempt only N pending tasks, then `--resume` to continue. Do not change source code or the protocol during a campaign: resume rejects incompatible provenance.

Use `-B` as shown: this repository already tracks some bytecode files, so ordinary imports can otherwise create unrelated working-tree changes despite the ignore rule. If an interrupted process leaves `.campaign.lock`, verify that no worker is running before removing only that stale lock; the runner deliberately does not guess that an existing worker is dead.

Dependencies: Python 3.11+, NumPy, SciPy and Matplotlib, plus the existing domain-engine dependencies (including Requests for the existing MKP loader module). Use a configured project environment; the runner never installs packages or downloads benchmark instances. Local MKP and CEC input files must exist.

## What is stored

```text
resultados/sensitivity/<protocol-name>/
  manifest.json           # Full protocol, seeds, source fingerprint and versions
  records/<task-id>.json   # Canonical atomic per-run records, including failures
  runs.jsonl              # Export of the canonical records
  runs.csv                # Flat export; nested metadata encoded as JSON
  summary.csv             # Separate case/mode/configuration summaries
  statistics.csv          # Same-mode baseline paired comparisons
  analysis_settings.json  # Explicit statistical settings and limitations
  figures/                # Separate case/mode pilot plots
```

The existing `resultados/` ignore rule keeps generated campaigns out of Git. Protocols and code remain versioned. Historical results and manuscript files are not modified. A failed run is not considered complete; interruptions do not turn partially written records into successful runs.

## Protocol and design

- `domains` selects cases and supplies a complete baseline monitor configuration and iteration budget for each domain.
- `seeds` is explicit and shared by every configuration. Legacy batch seed convention is 43–73 for 31 runs.
- `modes` runs both `dtw` and `ddtw`; `use_ddtw` is derived, not an independent sensitivity factor.
- `factors` expands a **one-factor-at-a-time (OAT)** pilot around each baseline. Resolved duplicates are removed; the baseline is retained exactly once per mode/case/seed.
- `band_ratio` is converted to the integer band `max(1, floor(ratio * window))`. Changing W also changes band at the fixed ratio. Therefore W results describe this coupled policy, not a pure fixed-band W effect.
- `percentiles` changes `[p_low, p_high]` as a pair. Other supported monitor factors can be added to the JSON protocol.
- Positive `min_slope` is fixed; zero selects the monitor's automatic rule, `0.01 * max(1, abs(last-first)) / W`.
- `max_epoch_iters: null` retains the legacy epoch lengths. A positive value opts into a cap on each epoch and remaining global iterations. The smoke protocol uses this cap explicitly and must not be compared to uncapped historical results.

The example pilot has only one MKP case, CEC F1 at D=10 and the HRES sizing case. It is **not** a claim of representative coverage. Expand MKP families and CEC F1–F12 before drawing benchmark-wide conclusions. Keep a separate held-out validation set if sensitivity results are used to select parameters. OAT does not estimate interactions such as W×band-ratio or plateau×patience; those require an explicitly designed follow-up protocol.

## Baselines reflect current code, not an assumed paper-wide table

| Domain | W | Band policy | Fixed/auto slope | Kmax | Nominal patience |
|---|---:|---|---:|---:|---:|
| MKP 1K | 40 | 10% of W | fixed 0.1 | 15 | 8 |
| CEC current 1K runner | 75 | 10% of W | fixed 0.1 | 15 | 25 |
| HRES current runner | 40 | 10% of W | auto (0) | 15 | 3 |

All use adaptive 30/70 percentile thresholds and the legacy zero improvement tolerance. The manuscript's generic 1K/3K table does not describe all current domain defaults. Reconcile the manuscript and the configuration that actually generated the historical runs before using pilot outputs as evidence for those exact reported experiments. The included pilot does not automatically add a 3K study; create a separate protocol/budget and do not pool it with 1K.

### Integration details that affect interpretation

- **MKP patience:** GA, ILS, WOA and GWO retain the existing minimum of 14. Each record includes effective per-solver settings. Nominal values below 14 do not vary patience for those four solvers.
- **MKP reference:** bundled OR-Library headers can contain zero. The example explicitly records the existing case0 best-known reference (24381) and its source. This is not a newly proven optimum. Without a positive reference, gap is absent, not replaced by the raw objective.
- **Iteration budget:** uncapped pipelines stop between complete epochs and may exceed the nominal budget. Records include actual history length and overshoot. Iterations are solver-specific steps/generations, **not equal objective-evaluation budgets**. Equal-evaluation studies need additional instrumentation.
- **HRES rotation:** an incompatible `pool_trayectoria` call was repaired by adding optional trajectory-pool support to the continuous orchestrator. Default CEC calls remain population-only; HRES explicitly alternates population and trajectory pools. This repairs the intended current integration, but does not prove historical HRES runs used identical scheduling.
- **HRES RNG:** case construction resets NumPy's RNG while generating synthetic weather. Optimization seeds are applied **after** case construction.
- **HRES quality:** the penalized objective, actual LCOE and feasibility are separate. The model's placeholder `optimum=0.30` is not a known optimum and is not used for a gap.
- **Switch count:** legacy `n_switches` counts epoch records, including the first. New output reports both epoch count and transitions (`max(0, epochs-1)`).

## Analysis contract

Quality metrics are lower-is-better: MKP reference gap (%), CEC objective minus official function bias, and feasible HRES LCOE (CNY/kWh). Summaries never pool raw scales across cases or domains. Feasibility rates, failures, missing runs and actual iteration counts are explicit.

Alternatives are compared to the **same-mode baseline**, pairing by seed within case and budget. Quality contrasts use only seeds where both solutions are feasible and have finite metrics; these conditional results must be interpreted alongside feasibility rates. DTW-vs-DDTW direct tests are not generated by this initial baseline-sensitivity report.

- Effect = baseline metric − alternative metric; positive values indicate improvement.
- Two-sided Wilcoxon uses SciPy's `auto` method; zero/tied differences affect its calculation. All-zero differences are reported as p=1, not equivalence. Fewer than two valid pairs yields no p-value.
- Holm families include all planned alternatives and cases in each domain/mode/budget/epoch-cap/metric. Missing tests remain in family size conservatively but receive no reported p-value. Partial output is labeled; it is not a final confirmatory analysis.
- Paired rank-biserial effect and deterministic 95% percentile bootstrap intervals for the mean difference accompany tests. Intervals are unadjusted, not simultaneous. Very small samples, particularly smoke runs, are not adequate inferential evidence.

References: [SciPy Wilcoxon documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html) and [SciPy bootstrap documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html). Signed-rank interpretation assumes a symmetric paired-difference distribution; same seeds do not guarantee identical random draws after solvers switch.

## Before updating the paper

- [ ] Reconcile the domain-specific baselines and current scheduling with historical experiments.
- [ ] Predefine the final cases, budgets, factor levels, comparisons and multiplicity families.
- [ ] Run all planned seeds and inspect failures, feasibility and actual computational costs.
- [ ] Add interaction/held-out validation experiments if making robustness or tuning claims.
- [ ] Report effects and uncertainty, not only p-values; lack of significance is not equivalence.
- [ ] Add verified tables/figures and a limitations paragraph to the manuscript.

The reviewer sensitivity item remains open until those empirical results exist. Implementing this module or passing smoke tests does not close it.
