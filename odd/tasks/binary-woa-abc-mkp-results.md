# Binary WOA–ABC: first MKP run and familiar results

## Objective
Run one functional check on instancias/mknapcb1.txt, zero-based index0, and save raw data, CSV/TXT summaries and plots following the existing MKP batch layout and visual style. Do not modify prior result folders or solver behavior.

## User decisions and constraints
- Exactly ONE run for now, not31 and not a performance campaign.
-1000 complete hybrid-cycle maximum, population30, first conventional seed43. A hybrid cycle includes WOA plus complete ABC; do not imply equivalence to legacy single-MH iterations.
- Copy detector configuration from hybrid_mkp/batch_benchmark.py: DDTW, window40, band0(auto4), min_slope0.1, plateau_max15, patience8, adaptive thresholds, p_low30/p_high70, improvement_tol0.
- Instance file stores reference0: preserve unknown gap; do not silently inject separate BKS24381.
- Remain on feat/binary-woa-abc-dtw. NEVER create or switch branches. User subsequently explicitly authorized commit and push of the completed work on this same branch; no PR requested. Ignored runtime results remain local.
- Legacy results and sources remain intact. Pre-existing untracked .codegraph/ is not feature work.

## Scope and design
Original R1 introduced root runner binary_woa_abc_benchmark.py, tests/test_binary_woa_abc_benchmark.py and updated binary_woa_abc.md. User subsequently authorized grouping all WOA–ABC code, docs and results under woa_abc/: move existing solver to woa_abc/solver.py, exporter to woa_abc/benchmark.py and documentation to woa_abc/README.md; add lightweight __init__.py and __main__.py. Keep focused tests under tests/ with updated imports. No algorithm/shared-detector changes.

After user-requested reorganization, fresh ignored runtime output under woa_abc/resultados/run_ddtw_<timestamp>/mknapcb1/inst_00/ (derive default root from package location, not current working directory). Move the ONE completed run from resultados/binary_woa_abc/ to this location without re-running or altering its11 artifact bytes:

- manifest.json with effective configuration, seed, input identity, timings and completion metadata;
- runs_resultados.csv and resumen_pipeline.txt using familiar Spanish artifact labels;
- runs/run_01.json preserving full solver result for offline analysis;
- best_run_01/ with resumen_pipeline_best_run.txt, historial_dtw.csv, historial_dtw_detalle.txt, fitness_convergence.png, dtw_delta.png and parameter_recovery.png;
- boxplot_runs.png, explicitly labelled one run (not inferential evidence).

Join diagnostics to history by cycle (including possible partial cycle); don't zip. Keep unknown instantaneous-population fitness blank, never substitute archived best. Algorithm label WOA-ABC, no invented algorithm switches/Gantt timings. Distinguish n_fires from n_recoveries; n_switches0 only if preserving legacy summary compatibility. Cycle/evaluation indices remain truthful. Mark recoveries in plots rather than falsely indicating algorithm rotations. Use familiar large type,150dpi and layouts, adapted axes labelled cycles. Use headless Agg before pyplot; no SciPy/statistical tests needed.

Only local reads and new timestamped output writes, no remote downloads. Refuse overwrite. Keep raw JSON available if plotting fails; report partial status, don't rerun optimization automatically. One-run dispersion has no performance interpretation.

## Verification configuration
TDD not enabled by observed explicit project/session settings; ordinary functional regression tests, no strict RED/GREEN claim. All Python commands use PYTHONDONTWRITEBYTECODE=1 or -B to avoid tracked cache regeneration.
Exact focused runners:
- PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -p "test_binary_woa_abc_benchmark.py" -v
- PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -p "test_binary_woa_abc.py" -v
- PYTHONDONTWRITEBYTECODE=1 python -m woa_abc --help
- PYTHONDONTWRITEBYTECODE=1 python -m woa_abc.benchmark --help
- git diff --check and git status --short (new files need readback too).
Executed R2 command before relocation (historical provenance, DO NOT rerun): PYTHONDONTWRITEBYTECODE=1 python -m binary_woa_abc_benchmark --instance instancias/mknapcb1.txt --instance-index 0 --seed 43 --population-size 30 --iterations 1000
Future command after R4 uses python -m woa_abc.benchmark with the same flags.
Default solver evaluation ceiling resolves to210030 at N30/T1000; scouts count. Record actual consumption, don't describe ceiling as actual budget spent. No additional evaluation cap that prevents finishing1000 cycles without reporting it.

## Tasks
- [x] R1 — Add familiar single-run serialization/plots and focused tests. **done**
  - Check local input and all inherited effective settings; raw JSON/manifest/CSV/TXT consistency; safe fresh directory; cycle/diagnostic joins; labels; valid PNG files using tiny test instances in temporary directories.
- [x] R2 — Execute exactly one real1000-cycle run and save artifacts. **done**
  - Confirm selected file/index/seed/population/DDTW settings before run; preserve exact command/timing/output directory; verify1000 completed cycles, feasible best, phase accounting, parameter restarts and emitted files. Report any incomplete outcome honestly.
- [x] R4 — Group code/docs/results under woa_abc/ and update imports/commands. **done**
  - Reversible moves of own feature files only; no root compatibility wrappers or leftover loose WOA–ABC module/docs. Lightweight package exports and python -m woa_abc solver entry; python -m woa_abc.benchmark for saved runs.
  - Move existing run directory once, compare SHA256 for all11 files before/after, no new optimization. Existing .gitignore resultados/ already ignores nested package results; verify no change needed.
  - Update focused tests and README; validate package-relative output default, root files absent and CLI help entrypoints. No create/switch branch, commits or push.
- [x] R3 — Independently inspect saved run and source verification/review. **done**
  - Recheck raw saved binary solution/objective/capacities, accounting, metadata, events and parameter transitions, CSV/TXT matching JSON, and PNG dimensions/nonempty images. No second1000-cycle run.
  - Native assessment and configured review preflight over new runner/test plus existing doc changes; record any unavailable/skipped checks. No comparison campaigns.

## Delivery and progress
Initial forecast approximately500–800 authored lines; actual R1 is929 exporter +666 tests +23 doc additions =1618 lines, with serializers, headless plots and failure-path tests. Scope remains results support; no unrelated solver/legacy changes. About400/task is advisory, not a cap; don't minify/omit tests. Strategy ask-on-risk before future commits/PR, none currently authorized. Existing implementation was committed77e7955 and pushed; do not alter branch topology.
Mapping completed read-only: existing batch31runs/seeds43–73, actual saved result directory inspected. User narrowed execution to one run. Branch verified feat/binary-woa-abc-dtw and resultados/ ignored.
R1 completed by grok-4.7/high: exporter,16 tests and23-line guide addition. Exporter tests passed in1.364s (expected mocked plot-failure traceback, not a failed test); solver37 tests passed in0.212s; CLI help and git diff --check passed. No tracked cache regeneration. Native ASSESS is unassessable due untracked declaration handling and requires independent verifier; no risk approval inferred.
R2 completed with exactly one solver invocation: authorized command exited0. Saved directory resultados/binary_woa_abc/run_ddtw_20261019_071137/mknapcb1/inst_00. Manifest completed,11/11 artifacts, no plot error, one CSV row, raw result preserved. Solve58.011s; filesystem observed artifact span58.377s is not measured process walltime.
Run: N30/T1000/seed43/index0, inherited DDTW config verified;1000 complete cycles, best23974, reference0/gapnull,180030 evaluations =30init+60000WOA+60000employed+60000onlookers+0scouts. Ceiling210030, scoutlimit3000. Feasible100-bit result uses [11819,13470,11547,12729,13166] of [11927,13727,11551,13056,13460].17fires/17recoveries, no algorithm switches. Example fire55 -> cycle56 a2/rho2/s0; same fixed slope afterwards. Last fire984 -> final cycle999 s14/a1.971972/rho1.985986.
All4 PNG files have valid signatures/dimensions: boxplot900x750, convergence2100x900, delta2100x750, parameters2100x900. No tracked cache changes; outputs ignored, branch unchanged. No new solve authorized for R3/R4. Independent verifier subsequently passed all16+37 tests and checked all saved JSON/CSV/TXT/PNGs, input hash, feasibility/objective and17recovery transitions. Image read tool is disabled: no visual-aesthetic inspection claimed, PNGs decoded and labels/style checked from source. One read-only verifier script had a parse-time error before execution; smaller checks succeeded. No saved-run defect found.
R4 completed: source/docs moved to woa_abc/{solver.py,benchmark.py,README.md}, lightweight init/main added, tests/imports/CLI paths updated. Root loose modules/docs absent. Output default is package-relative woa_abc/resultados/, independent of CWD. Tests38solver+17exporter passed (0.206s/1.474s), both CLI help entrypoints passed. No algorithm changes, branch unchanged, no staged changes.
Completed run moved to woa_abc/resultados/run_ddtw_20261019_071137/mknapcb1/inst_00/. All11 before/afterSHA256 hashes matched (mismatch0); rawJSON hash f41280b111700bb66d2fbe77aabc7a3d92a5e472640838a306dc64924b03e657; manifest hash170519b190be0b1d2a2b4e1b6f93f3e39195d5cf06e109a4a88690ea7da53468. Historical artifacts unchanged, no rerun; only empty own old result container removed. Existing resultados/ gitignore matches package results. R3 independent package recheck and review now pending. Native ASSESS still unassessable for untracked declaration, separate verifier required. Final independent package recheck passed:38solver tests(0.179s),17exporter tests(1.322s), both CLI helps, package-relative root after changing CWD and lightweight init. Solver source diff against committed root module is only CLI prog string (no algorithm change). Independent checker matched persisted rawJSON/manifest hashes and rechecked11files/feasible23974/evals180030/1000cycles/17recoveries with no real rerun. All source/cache changes within expected feature surfaces, branch fixed, nothing staged.
Native review lineage review-1003f85e24a41299 approved by one review-reliability lens (medium,9changed paths/3722diff lines including relocation). Exact acknowledgement returned authority=burned for target sha256:173febbe8a85fb3866de13eddc788c010d406b6a8455d524c2aca218bf95e4a1, consumed revision sha256:84bed58f6d640aec846bf0e9055a69262c38d1952325226e98604a26fd7c5b43. No further STATUS after burn. Advisory R3-001 at tests/test_binary_woa_abc.py:811 is non-blocking (package-isolation assertions depend on shared sys.modules when suites combined); no correction offered, follow-up only. Focused suites run separately as documented and pass.
ASSESS remains unassessable because of its untracked declaration handling; no passed risk computation claimed. NativeReviewOutcome closed recorded only after exact acknowledgement; separate independent checks already performed. Skipped: full legacy suite (removed-module imports), visual aesthetic image inspection(tool disabled), additional optimizations/campaigns/statistics/commits/push. No functional failure pending. User subsequently authorized commit and push on the existing branch; exact commit identity is recorded in Engram after creation.

## Next step
Report final woa_abc/ layout, preserved saved run and55passing focused tests. User can inspect woa_abc/resultados/run_ddtw_20261019_071137/mknapcb1/inst_00/. Commit and push are now explicitly authorized on the same branch; no further run is authorized. Future optional follow-ups: test isolation warning and empirical inspection of frequent parameter resets, not automatic changes.
