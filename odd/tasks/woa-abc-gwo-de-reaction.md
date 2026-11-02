# WOA–ABC with the GWO–DE stagnation reaction

Branch `feat/binary-woa-abc-dtw`, worktree `.worktrees/woa-abc`. Keep WOA → ABC sequential; replace the a/rho recovery clock with the GWO–DE profile mechanisms (exploit/explore profiles, four policies, DTW_KIRITO).
Reference implementation (read-only, other branch): `gwo_de_dtw/controller.py`, `gwo_de_dtw/policies.py`, `gwo_de_dtw/config.py` on `feat/gwo-de-dtw-hybrid` (read with `git show feat/gwo-de-dtw-hybrid:<path>`).

## Specs
- S1. User: "necesito que la reaccion a estancamiento sea igual a  gwo de los mismos modos, que siga secuencial se entiende?" — the cycle order stays WOA → ABC employed → ABC onlookers → scouts.
- S2. User: "quiero los mismos mecanismos qye hay de binary y continous" — implement the four GWO–DE policies with the same rules:
  - `binary_simple`: exploit → explore when `D2_vs_const <= theta_c` on a ready observation.
  - `binary_complex` (default, as in GWO–DE): exploit → explore when the monitor fires.
  - Both binary policies: explore → exploit when `no_improve_len < plateau_max` on a ready observation, otherwise stay in explore; after every profile change the monitor is replaced (fresh window).
  - `dtw_kirito=True` (binary only, default False): monitor never replaced; every ready observation sets explore if the policy trigger holds, else exploit.
  - `continuous_simple`: intensity = 1 − clip(D2_vs_const / (theta_c·scale + eps), 0, 1).
  - `continuous_complex`: intensity = sigmoid(k·(delta / (theta_delta + eps) − center)).
  - Continuous policies never replace the monitor; intensity is 0 before the first ready observation; the active profile is the linear blend of exploit (0) and explore (1), named explore when intensity > 0.5.
  - Defaults as GWO–DE: scale 2.0, k 5.0, center 0.5.
- S3. Profiles control the WOA `a` and the ABC step amplitude `rho` (phi ~ U[−rho, rho]), plus G3 (LB2 base flip probability) for parity with GWO–DE. Within a profile `a` is constant (as in GWO–DE), not the old linear decay. Values are provisional — user: "despues lo discutimos". Defaults in one config place: exploit a=0.5, rho=1.0, G3=0.5; explore a=2.0, rho=2.0, G3=0.5.
- S4. The monitor observes the archived best once per complete cycle (after ABC/scouts); partial final cycles do not update it. A profile change decided after cycle g applies from cycle g+1. Population, failure counters and archive are never reset.
- S5. The old reaction (a/rho recovery clock and the variants that only alter it, `v2` and `v3`) is replaced. `v1` (scout limit 100) and `v4` (single-coordinate ABC step) are independent of the reaction and stay. `use_stagnation=False` / `--no-stagnation` keeps the exploit profile for the whole run.
- S6. Record per cycle: profile name, a, rho, G3, intensity (None for binary), changed flag, and monitor output; record the cycles where the profile changed. Plots/exports that showed recoveries show profile changes instead.
- S8. User: "que los de woa abc sean los mismo que gwo de" — the WOA–ABC plots are the GWO–DE plots: same files, same drawing code copied from `gwo_de_dtw/plots/` (convergence and delta segmented and coloured by profile with change lines, parameters per cycle, four-panel temporal with NPZ/CSV series, and the four-policy convergence comparison). Panels name WOA–ABC parameters (a, rho) instead of a, F, CR; titles say WOA–ABC; the x axis is cycles.
- S7. CLI, benchmark and campaign expose `--policy` (four choices, default binary_complex) and `--dtw-kirito`. Docs (README) describe the new reaction and drop the recovery-clock text.

## Tasks
| ID | Specs | Route | Commit |
|----|-------|-------|--------|
| T1 | S1–S7 | worker | f517717 |
| T2 | S1–S7 | verify inline (tests + smoke) | f517717 |
| T3 | S8 | worker | this commit |

## Log
- L1. User: "cambia a la rama de woa abc, necesito que la reaccion a estancamiento sea igual a  gwo de los mismos modos, que siga secuencial se entiende?"
- L2. User decisions: branch via separate worktree; reaction: "quiero los mismos mecanismos qye hay de binary y continous"; profile values: "despues lo discutimos" (provisional defaults, S3).
- L3. Parent interpretation to confirm with user at close: v2/v3 removed because they only modify the replaced recovery clock (S5).
- L4. Evidence: parent re-ran 103 targeted tests (101 pass; 2 pre-existing: `woa_abc.adaptive` import in legacy test_woa_abc_variants, TestPackageBoundary order-dependent, passes alone). Smokes seed 1, N6, 120 cycles: binary_simple best 23473 changes [18,40,63]; continuous_simple best 23423, 7 changes; worker: binary_complex 23462 (1), continuous_complex 23459 (7), kirito 23462 (3). Controller parity with gwo_de_dtw read line by line.
- L5. User: "que los de woa abc sean los mismo que gwo de" (after the parent showed the plot differences: profile-coloured convergence/delta, parameters, temporal panel, policy comparison vs WOA–ABC single segment and boxplot).
- L6. Evidence T3/T4: parent re-ran 110 targeted tests (only the 2 pre-existing failures); `python -m woa_abc.run` smoke (seed 1, N6, 120 cycles) wrote four policy folders with GWO–DE plot set and convergence_comparison.png/.pdf; parent inspected fitness_convergence and comparison. Removed unused woa_abc/mkp_plots.py; boxplot_runs.png and parameter_profiles.png dropped to match GWO–DE.
