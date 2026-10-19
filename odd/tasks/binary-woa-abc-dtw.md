# Binary WOA–ABC with DTW parameter recovery

## Objective and rationale
Implement a standalone, reproducible binary MKP hybrid with one shared population and WOA followed by full ABC. Reuse the repository's LB2 probability transforms, greedy repair, and existing DTW/DDTW detector. Replace legacy epoch abortion/algorithm rotation with parameter recovery; do not modify legacy pipelines.

## Accepted design
- Shared state: continuous positions, feasible binary solutions, cached fitness and ABC failure counters. Initialize once using the WOA initializer (positions uniform in [-1, 1], sign bits, repair).
- Every complete cycle: WOA -> ABC employed -> ABC onlookers -> eligible scouts. Keep this phase structure fixed.
- Preserve legacy LB2: two independently sampled flip candidates, both repaired/evaluated, ties prefer L2. Fixed monitored defaults G1=G2=G3=0.5, v_max=6, spiral b=1. Use a local NumPy generator, never global RNGs.
- Preserve algorithm-specific acceptance: WOA accepts binary fitness ties and always updates latent positions; ABC accepts strict improvements only. Reset source failures on actual WOA/ABC improvements. Snapshot the WOA leader to avoid a mutable-view alias.
- WOA a decreases linearly from 2 to 0; ABC initially samples phi uniformly in [-1,1]. WOA branch p and a2/l behavior remain unchanged; only a and ABC amplitude rho are controlled.
- T is the configured total cycle count, T >= 2. Before any detection, s is the ordinary cycle index and rho=1. After a detector fire, the next cycle starts with s=0, a=2, rho=2. Then a=max(0,2-2s/(T-1)); rho=max(1,2-s/(T-1)). Keep the original T after every recovery, even late in a run.
- Observe archived best fitness once per complete cycle using StagnationMonitor. On fire, record diagnostics, reset only the detector and recovery clock, preserve population/trials/archive. Detector window, plateau, thresholds and patience are unchanged. No extra hysteresis, edge requirement, or algorithm selection.
- A strict objective-evaluation ceiling counts initialization, both LB2 candidates, and scouts. Do not start an LB2 proposal with fewer than two evaluations left; a terminal unused evaluation is permitted. Incomplete final cycles are recorded but do not update DTW.
- Separate total cycle and evaluation limits; either can stop a run. No hidden extra objective calls or evaluation caching.

## Scope and constraints
New files only: binary_woa_abc.py, tests/test_binary_woa_abc.py, binary_woa_abc.md. Task document is parent-owned. No edits to shared detector, repair, legacy modules, papers or existing results. No campaigns, comparisons, scientific superiority claims, external downloads or dependency installation.

Authoritative branch: feat/binary-woa-abc-dtw, base 48bf26f. Initial pre-existing untracked .codegraph/ is not part of this feature. Commits/push/PR are not authorized: preserve changes uncommitted under the explicit no-commit safety policy; commit IDs remain pending user authorization.

## Verification configuration
- TDD: not enabled by any observed explicit project/session setting; use ordinary functional regression verification, not a claimed strict RED/GREEN workflow. Exploration found no project AGENTS.md, .pi/settings.json, pyproject.toml or pytest.ini prescribing TDD. Tests existing does not enable it.
- Exact focused runner: PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -p "test_binary_woa_abc.py" -v
- Writer self-verification runs synchronously. Independent command verification is delegated on demand; native ASSESS is run after writer return.
- Structural check: git diff --check (new files also inspected through readback).
- CLI checks after T2: python -m binary_woa_abc --help; local small-instance smoke command is recorded when the local fixture path is established. Never fetch remote data.

## Tasks
- [x] T1 — Implement shared-population WOA/full-ABC baseline and focused tests. **done**
  - Seeded local RNG, feasible archive, strict accounting and termination, full phase order, scouts, and baseline a schedule; adaptive behavior belongs to T2.
  - Checks: focused unittest runner; cover deterministic runs/global RNG isolation, phase execution, exact objective counts including scouts and terminal budgets, feasible binary archive preservation and config validation.
- [x] T2 — Add existing-detector recovery control, local CLI and user documentation. **done — terminal-budget metadata corrected**
  - Record per-cycle a/rho/evaluation/best/DTW diagnostics and repeated fires. Add tests for initial rho, next-cycle reset, same-slope decay, monitor rearming and fresh warm-up, persistent populations, and partial-cycle handling.
  - Local-only OR-Library input, selected instance index, seed, population/cycle/evaluation options, DTW/DDTW selection and JSON result.
  - Checks: focused unittest runner, CLI help and a bounded local smoke. Documentation describes both budget stops and lack of performance claims.
- [x] T3 — Verify integrated implementation and inspect native review authority. **done**
  - Read changed surfaces; exercise focused tests and real local smoke through a read-only verifier if needed; native ASSESS followed by configured review preflight.
  - Record passed, failed, pending/skipped checks and exact review result. No campaigns, commits or delivery operations.

## Delivery forecast
Initial forecast 700–1000 authored lines; T1 actually produced 1224 (625 solver + 599 tests) with comprehensive accounting/validation coverage. Revised forecast approximately 1600–2000 including adaptation, CLI and documentation. Keep coherent baseline and adaptive/CLI work units rather than unrelated repository changes. About 400 lines/task is advisory, not a cap; do not minify or omit tests. Delivery strategy: ask-on-risk before any future commit/PR; no commit/PR currently authorized. Native review must not be mistaken for commit permission.

## Progress and evidence
- Design and reusable seams explored read-only; implementation explicitly authorized by user.
- T1 completed by delegated writer (grok-4.7, high). New baseline solver and tests read back; 26 unittest tests passed in 0.080s. git diff --check passed (tracked diffs); writer also inspected new files for whitespace/conflict markers.
- Strict local-RNG shared population and complete WOA/employed/onlooker/scout phase sequence implemented. LB2 pairs are atomic with a possible one-evaluation terminal remainder; complete-cycle hook is ready for T2.
- Native ASSESS after T1: unassessable because untracked scope is undeclared; plan requires independent verifier. No review authority or approval inferred.
- Tooling incident: tests rewrote exactly five previously clean tracked .pyc files. Separate read-only diagnosis confirmed generated caches only. Parent restored only those five from HEAD; status now contains only new feature files and pre-existing .codegraph/. All later Python commands must set PYTHONDONTWRITEBYTECODE=1.
- T2 completed by same delegated writer (grok-4.7, high): existing run-local detector, next-cycle recovery, local-only CLI and documentation implemented. 36 unittest tests passed in 0.164s; CLI help and git diff --check passed.
- Real local smoke: PYTHONDONTWRITEBYTECODE=1 python -m binary_woa_abc --instance instancias/mknapcb1.txt --instance-index 0 --seed 1 --population-size 4 --iterations 6 --max-evaluations 200. Passed: feasible n100/m5 result, best 22229, 148 evaluations, 6 completed cycles, reference0 => gap null. No DTW fire expected before default window40.
- Real detector regression covers repeated flat-signal fires at cycles4 and9 with fresh window/patience after reset, next-cycle a/rho recovery and original slope. No population/trial reset or shared monitor state.
- Current authored size: 915 solver/CLI + 886 tests + 104 docs = 1905 lines. No tracked source/cache changes. No performance superiority claim.
- Independent verifier ran all 36 tests (0.176s), CLI help, baseline smoke twice with identical JSON, and DDTW real-instance smoke (N4/T16/budget500/window4/band1/slope2/plateau2/patience2). All commands passed; latter produced 388 evaluations, best22443, fires7/12 and next-cycle resets8/13; objective and feasibility independently checked.
- Verifier identified a source-level defect: recovery_applied checks remaining cycle indices but not remaining evaluation budget. On a fire with 0 or1 evaluations left, it can claim a recovery on a nonexecuting next cycle. T2 reopened; add regressions for both remainders and fix the truthful metadata/state condition. T3 pending rerun; no complete/approval claim.
- Native inspect ready with only implementation/test/doc paths selected; no lineage started. ASSESS still unassessable for untracked declaration and requires independent verifier. Re-inspect after correction to avoid stale candidate.
- Bounded correction completed: recovery_applied now requires a later cycle AND >=2 remaining evaluations, while every fire still resets detector. Reproducer before: 62 5 False 2.0 2.0 4 True True; after: 62 5 False 0.0 1.0 4 True False. Added budgets62/63/64 regression; 37 tests passed in 0.173s; caches remain unchanged.
- Independent re-verification passed: 37 tests in0.227s, exact budgets62/63/64 repros (recovery false/false/true), real DDTW smoke repeated best22443/evals388/fires7,12, no tracked source/cache modifications. One prior verifier continuation timed out before model execution; a fresh foreground verifier completed successfully.
- Native review completed: lineage review-28bf62d0db8f4636, medium tier, one review-reliability lens, 3 changed files and1946 authored lines. Native last event approved; exact acknowledgement returned authority=burned for target sha256:6c6eb957330ec9f39ebc03264edb93b6760ec8517a57b81d614ac5c8f910e4bf, consumed revision sha256:6400eba75bda736ed1aea80e47c4b5a0f334786dbd09686375d98798e6b7f168. No further lifecycle STATUS after acknowledgement.
- Non-blocking native advisory R3-roulette-overflow at binary_woa_abc.py:348–355 (roulette fitness sum numerical overflow). No correction offered; retain as separate potential hardening, not part of this implementation. No severe findings remain.
- ASSESS after exact acknowledgement recorded nativeReviewOutcome=closed; it remains risk-unassessable due its handling of undeclared untracked paths, but independent verification was already completed. No unavailable assessment is reported as a passed risk calculation.
- Skipped intentionally: broad legacy suite (deleted-module imports), large campaigns, comparative/statistical validation, external research, commits/push/PR. No scientific performance claim.
- Work-unit commit identities: pending explicit user authorization (no commits made).

## Next step
Report implemented API/CLI,37 passing focused tests and successful local DDTW smoke. Next user-controlled step: inspect parameter/event histories on a small run. Numerical-extreme roulette hardening is a separate non-blocking follow-up; no changes or campaigns without authorization.
