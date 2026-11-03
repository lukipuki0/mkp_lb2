# Sequential GWO then DE

Standalone binary MKP search. Each generation runs one full greedy GWO sweep, then one full greedy DE sweep over the population that sweep produced. GWO and DE do not compete. This package does not import or modify the other hybrid package.

## Schedule

1. Copy alpha, beta and delta at the start of the generation. Those leaders stay frozen for every GWO move.
2. GWO sweep, in index order. A candidate replaces its individual only when its repaired fitness is `>=` the current one. Real vector, bits and fitness are written together, or not at all.
3. DE/rand/1/bin sweep over that post-GWO population, in place. A later DE move can use a row accepted earlier in the same sweep.
4. One shared profile (`a`, `F`, `CR`, `G3`) covers both sweeps.
5. The stagnation monitor observes once, after the generation, using the best-so-far fitness. A profile change applies on the next generation.

LB2 is the default. Initialization costs `N` repaired evaluations. Each generation costs `4N` with LB2 (two operators, two repaired LB2 candidates each) and `2N` with sigmoid. The global best improves only on a strict `>`.

## Policies

`binary_simple`, `binary_complex`, `continuous_simple`, `continuous_complex`, in that order.

Binary policies switch between the fixed exploit and explore profiles. Without Kirito (the default) a change replaces the monitor, so the next decision waits for a fresh window. `--dtw-kirito` keeps one monitor and, on every ready observation, explores only while the policy trigger holds.

Continuous policies never replace the monitor. They blend exploit and explore. The profile is named explore when the intensity is above 0.5.

Exploit: `a=0.5`, `F=0.5`, `CR=0.9`, `G3=0.1`. Explore: `a=2.0`, `F=0.9`, `CR=0.3`, `G3=0.1`.

## Dependency

`dtw_stagnation.StagnationConfig` and `StagnationMonitor` are imported from the repository root. The four policies read that monitor (`ready`, `fire`, `D2_vs_const`, `theta_c`, `delta`, `theta_delta`, `no_improve_len`). The module is not copied.

OR-Library parsing, the `mkcbres.txt` lookup and greedy repair live in `problem.py`. They are not imported from another package.

## Campaign

```text
python -m gwo_de_sequential.campaign
python -m gwo_de_sequential.campaign --dry-run --instances-dir instancias --out campaign_check
```

Defaults: `instancias/mknapcb1.txt` .. `mknapcb9.txt`, indices `0 1 2`, seeds `1` .. `31`, four policies, LB2, DDTW, window 40, population 30, 1000 generations, Kirito off. That is 3348 runs.

`--dry-run` validates and writes nothing. `--out` must not already exist; there is no overwrite and no resume. If `--out` is omitted, a new `gwo_de_sequential/results/campaign_<timestamp>` folder is created. A failing run sets `status=failed`, keeps finished rows, and stops. Launching again at the same path is rejected.

```text
campaign_config.json
summary.csv
statistics.csv
<stem>/inst_<NN>/runs.csv
<stem>/inst_<NN>/statistics.csv
<stem>/inst_<NN>/convergence_comparison.png
<stem>/inst_<NN>/convergence_comparison.pdf
<stem>/inst_<NN>/<policy>/best_run/result.json
<stem>/inst_<NN>/<policy>/best_run/trace.csv
<stem>/inst_<NN>/<policy>/best_run/fitness.png
```

`summary.csv` and `runs.csv` store metrics for every run. Traces and plots are written only for the best run of each policy. The higher fitness wins; a tie keeps the lower seed. The comparison legend is `<policy> (seed <seed>)`.
