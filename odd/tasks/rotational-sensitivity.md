# Rotational DTW/DDTW sensitivity tooling

## Objective
Provide reproducible local sensitivity campaigns for the manuscript's rotational MKP, CEC2022 and HRES2-H2 framework, reusing existing solvers.

## Problem and rationale
The reviewer asks for hyperparameter sensitivity evidence. Current batch runners hide configuration and seed metadata, and cannot isolate/resume a sensitivity campaign safely. Building tooling is not evidence of robustness; real study results remain pending.

## Authorized scope and constraints
- User selected the original rotational framework, not active WOA–ABC variant experiments.
- Implement configurations, thin adapters, execution/resume, analysis, tests and documentation.
- Preserve benchmark defaults, MKP effective patience overrides, existing paper edits and historical results.
- Repair the incompatible HRES trajectory-pool call with regression coverage. Any budget cap must be opt-in, explicitly recorded and never silently change legacy runs.
- Local functional/smoke verification only; no full campaign, network downloads, HPC submission, remote credentials, manuscript claims or reviewer checkbox completion.
- Initial one-factor-at-a-time examples are pilot protocols, not an exhaustive interaction study or final selection of representative cases.

## Checks and workflow
- No TDD mode is enabled in supplied project/session instructions or discovered local configuration. Use ordinary functional checks; do not claim RED/GREEN evidence.
- Runner: Python 3.11, `python -m unittest discover -s tests -p "test_sensitivity*.py" -v`.
- numpy/scipy/matplotlib available; pytest absent, so existing pytest suite cannot currently run.
- Native RDD is not enabled; status command printed off but failed repository identity resolution (access denied). Do not start or enable native review; delivery is unmanaged, with functional checks.
- CodeGraph previously returned no index and instructed no retries this session; use direct reads.

## Tasks
- [x] T01 — Validate versioned protocols; generate stable DTW/DDTW configuration IDs, paired seeds and OAT tasks; save manifests/raw records and safely resume.
  - Checks: invalid values, duplicate tasks, baseline identity, dry-run, configuration mismatch, failed-run retry and resume contracts.
  - Evidence: 14 campaign checks pass (13 delegated checks plus transitive lb2 provenance regression). Dry-runs: smoke6 tasks, pilot2046 tasks. Manifest includes scientific source/input hashes, versions, explicit protocol and atomic per-task records. Integer bands and numeric-equivalent settings deduplicate correctly. Resume rejects protocol/source/input changes and retries failed/incomplete tasks.
- [x] T02 — Integrate MKP/CEC/HRES single-run adapters and necessary backward-compatible orchestration seams.
  - Checks: seed after case construction, actual/effective parameters, feasibility/domain metric semantics, HRES population/trajectory rotation, opt-in budget behavior and legacy defaults.
  - Evidence: 15 adapter unittest checks passed, independently rerun by parent with 8 analysis checks (23 total). Six real-engine smoke cases (3 domains x DTW/DDTW), seed43, budget4/cap2, each completed 4 recorded iterations, 2 epochs, 1 transition and zero overshoot. Source-scoped diff hygiene passed. The coherent task exceeds the advisory400-line heuristic due to complete adapter/regression coverage and backward-compatible seams.
- [x] T03 — Produce domain-separated descriptive summaries, baseline paired Wilcoxon/Holm, paired effects/intervals and plots.
  - Checks: known synthetic comparisons, all-zero differences, missing pairs, feasibility filtering and no raw-scale pooling.
  - Evidence: `python -m unittest discover -s tests -p "test_sensitivity_analysis.py" -v`: 8 tests passed. Synthetic two-configuration plot generated under ignored resultados/sensitivity/tooling_analysis_qa and visually inspected (labels/axes readable). Full campaign CLI integration remains T04.
- [x] T04 — Document pilot protocols and commands; verify all three real adapters with tiny smoke cases and complete integration checks.
  - Checks: CLI dry-run/execution/resume/analysis, compile, diff hygiene and preserved user manuscript changes.
  - Evidence: final focused suite37 tests passed; all6 existing fixture-free WOA test functions passed via unittest.FunctionTestCase (pytest runner unavailable). CLI smoke: --limit2 then --resume completed6/6 real runs and analysis generated6 summaries/6 figures. Earlier first2 record hashes proved resume skipped successful tasks. Separate tiny CEC2-seed/2-configuration QA completed8/8 runs,4 summaries and2 paired statistical comparisons. Final source/input provenance guard rerun after including lb2. Compiled16 source files without generating bytecode. New/changed file whitespace scan and source-scoped git diff --check passed; global diff --check still flags pre-existing manuscript whitespace line164. Manuscript/action-matrix hashes unchanged. No full campaign or HPC job run.

## Acceptance criteria
- Both DTW/DDTW supported in each domain without duplicating optimization algorithms.
- Every result identifies case, seed, monitor mode, requested/resolved/effective configuration, nominal/actual iterations, objective, time, feasibility and applicable domain metric.
- Resume rejects incompatible protocol/provenance and never treats failed/partial runs as complete.
- Generated files remain in isolated ignored campaign directories.
- Documentation clearly distinguishes smoke/pilot output from publishable sensitivity evidence and explains current code/manuscript discrepancies.

## Progress and next step
Tooling implementation complete and checked; all4 tasks complete. Full empirical sensitivity study remains pending: reconcile historical domain configurations/scheduling, expand case coverage beyond the single-case pilot, finalize protocol and run31 seeds before changing paper claims/reviewer status. Pilot2046 tasks was validated only, not executed. Existing manuscript changes remain untouched (hashes 6de8d2b030f1bedd179b532765c2560ffc386429 and 760c21eb8ac769bc6ca48a3b5e86e5d52279345b). Global diff hygiene flags pre-existing manuscript whitespace line164; source-scoped checks pass. Restored21 originally-clean generated tracked .pyc files; remaining checks use python -B. No commits, network downloads or remote operations. RDD disabled/unmanaged; status repository identity lookup access denied, no review/enable attempted.

## Rollback boundary
Remove sensitivity/, the3 test_sensitivity*.py files and this task document; revert only the four source changes in hybrid_mkp/orchestrator.py, hybrid_mkp/mh/sa.py, continuous_benchmark/orchestrator.py and HRES2-H2/orchestrator.py. Generated ignored QA campaigns may be archived separately. Do not revert the user's manuscript or reviewer action matrix.

## Recovery locator
Repository-relative path: odd/tasks/rotational-sensitivity.md. Engram topic: odd/rotational-sensitivity/tasks.
