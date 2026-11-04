# GWO-DE DTW/DDTW hybrid on the HRES2 (WPEB) problem
Branch `feat/gwo-de-hres2` from `main` @ c893092. New package `gwo_de_hres2/`; `gwo_de_dtw/` and `HRES2-H2/` are imported, never edited.
Interpreter: `.venv/Scripts/python.exe` (system Python 3.14 lacks pandas). Scope ends at a short smoke run; no full campaign, no push.

## Specs
S1. User: "quiero implementar lo del hres2 en el gwo_De_dtw". Chosen: "Correr GWO-DE-DTW sobre HRES2" (run the GWO-DE hybrid with its DTW/DDTW profile policies on the HRES2 WPEB problem, continuous, minimize LCOE, no binarization).
S2. Chosen location: "Paquete nuevo gwo_de_hres2/" (reuses gwo_de_dtw operators/controller without touching the MKP code).
S3. Chosen scope: "Rama nueva, smoke corto" (new branch, tests and a short smoke; no full campaign, no push).
S4. Design (agent, from exploration): search in normalized z in [-1, 1]^4 mapped linearly to `HRES2Function.lb_vector..ub_vector`, so `gwo_de_dtw.gwo.gwo_candidate` / `gwo_de_dtw.de.de_candidate` are reused unchanged with `v_max=1`. Competitive mode only (GWO vs DE per individual, greedy `<=` on LCOE objective). Cost: N at init, 2N per generation. Objective is `HRES2Function._evaluate` (LCOE, or the 100+10*AGSR penalty when infeasible). The monitor receives `-best` (it maximizes), as HRES2 MH do. Profiles a/F/CR from `gwo_de_dtw.config`; G3 unused. All six policies of `gwo_de_dtw.policies.CAMPAIGN_POLICIES` via `gwo_de_dtw.controller.ProfileController`.
S5. Defaults: N=30, 1000 generations, seed 1, monitor = HRES2 paper nominal (W40, band 4, min_slope 0.1, plateau 15, patience 8, adaptive 30/70), signal DDTW selectable DTW. Records per run: policy, signal, seed, objective, feasible, physical LCOE/LCOH/AGSR, solution, decoded capacities, evaluations, series, changes, elapsed. CLI writes `runs.jsonl` + `manifest.json` to a fresh directory and refuses a non-empty one.

## Tasks
T1 | S1-S5 | inline | done | Package gwo_de_hres2 (config, search, control, run CLI), tests, README. RED (module missing) then GREEN 12 tests; regression gwo_de_dtw+policies+hres2 101 tests OK. Commit: 1dfec2b.
T2 | S3 | inline | done | Short real smoke (tiny N/iterations, all six policies) and report runtime. Commit: n/a (results not committed).

## Log
L1. User: "quiero implementar lo del hres2 en el gwo_De_dtw okey preguntame si es necesario donde guardarlo etc"
L2. Answers: "Correr GWO-DE-DTW sobre HRES2"; "Paquete nuevo gwo_de_hres2/"; "Rama nueva, smoke corto".
L3. Measured one HRES2 evaluation ~60 ms (.venv); default N30 x 1000 gens = 60030 evals, about 1 h per run.
L4. Correction of L3: that measurement included numba JIT. First call ~1.2 s, warm ~0.24 ms/eval, so a default run is ~15-25 s per policy.
L5. Smoke job-1 exit 0: `--pop-size 10 --iterations 60 --seed 1 --signal ddtw`, output gwo_de_hres2/results/smoke_ddtw_n10_t60_s1, 1210 evals and 0.4-1.2 s per policy, all six feasible. LCOE: binary_simple 0.267379 (1 change), binary_complex 0.274486 (0), continuous_simple 0.267212 (2), continuous_complex 0.268867 (1), vanilla_exploitation 0.274486, vanilla_exploration 0.267193. binary_complex never fired in 60 gens so it equals vanilla_exploitation. One seed, tiny budget: no comparative claim.
