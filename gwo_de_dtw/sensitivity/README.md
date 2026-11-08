# GWO-DE MKP sensitivity

Descriptive one-at-a-time monitor study for the binary MKP hybrid only.
It does not change algorithm formulas, the ordinary campaign, or manuscript files.

Approved design: DDTW, `binary_simple` and `binary_complex`, competitive mode, LB2,
N=30, T=1000, seeds 1..31, slope 2.0, improvement tolerance 0, adaptive thresholds on.
Baseline is W40, absolute band 4, p_low 40, p_high 60, plateau_max 15, patience 8.
Each alternative changes one of those monitor fields. Window alternatives keep band 4;
band is not recomputed as 10% of W.

Configurations (19): baseline, window_20, window_80, window_160, band_1, band_2, band_8,
p_low_20, p_low_30, p_low_50, p_high_50, p_high_70, p_high_80, plateau_max_5,
plateau_max_10, plateau_max_30, patience_2, patience_4, patience_16.

Instances: `instancias/mknapcb1.txt`, indices 0, 15 and 29.
Full plan: 19 x 2 x 3 x 31 = 3534 runs. Baseline is executed once per instance, policy
and seed, not once per varied parameter. Quality is not pooled across instances.
There are no inferential tests. Seed-paired initial conditions are not claimed.

Run from the repository root.

```text
python -m gwo_de_dtw.sensitivity.campaign --dry-run
python -m gwo_de_dtw.sensitivity.campaign --configurations baseline window_20 --indices 29 --seeds 1 --pop 4 --iterations 2 --out gwo_de_dtw/results/sensitivity_smoke
python -m gwo_de_dtw.sensitivity.campaign --workers 2 --indices 0 15 --seeds 1 --configurations baseline --pop 4 --iterations 1 --out gwo_de_dtw/results/sensitivity_smoke_parallel
```

`--workers` defaults to 3 and is capped at the number of selected indices, so the three
approved instances can run at the same time. Configurations, policies and seeds stay
sequential inside each instance. `--workers 1` keeps the in-process serial path.
Invalid `--workers` is rejected before any write. Child processes do not plot and do not
write `manifest.json`, `runs.csv` or `runs.jsonl`; the parent appends those as each
completed run is flushed. A failed optimization is not retried. Instances that have not
started are left unstarted. An instance that has already started is stopped by process
termination: `run_hybrid` has no cooperative cancel, so this is not an in-frame interrupt
and the in-flight run is not recorded as completed. Winner `result.json` files and rows
already flushed are kept. Final tables follow the selected index, policy, configuration
and seed order, not whichever instance finished first. Speedup is limited by CPU and
memory contention; it is not a claim that three instances finish in one third of the time.

`--dry-run` validates and prints the plan. It does not run the optimizer and does not
write. Invalid seeds, duplicate selections, budgets, indices, OAT invariants, unknown
configuration names, or a nonempty `--out` exit 2 and leave the output untouched.
A run or export failure records status `failed`, keeps completed rows, does not retry,
and exits 1. Interruption records status `interrupted`. There is no resume.
Status is `completed` only after winner records, tables and plots are exported.
Every winner `result.json` is written before the first plot.

Outputs, in the given directory or `gwo_de_dtw/results/sensitivity_<timestamp>/`:

- `manifest.json`: effective config, provenance, runtime, and known/unknown optimum evidence
- `runs.csv` and compact `runs.jsonl`, appended after each run
- `statistics.csv`, `comparisons.csv`, `resumen.txt`
- `plots/sensitivity_<policy>.png`: each parameter's alternatives against baseline zero, by instance
- `mknapcb1/inst_<NN>/<policy>/<configuration>/best_run/result.json` plus the existing per-run diagnostics

The winner is the highest fitness; ties keep the lower seed. Sample SD is blank when n=1.
An unknown reference leaves the gap blank, not zero. `elapsed_s` is wall-clock seconds.
Plots are descriptive and are not a robustness claim.
