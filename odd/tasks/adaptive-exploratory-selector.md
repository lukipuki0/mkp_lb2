Objective: Add an experimental adaptive selector that redistributes exploration probability toward less-used MHs, compare it against the existing selector, and run the user-requested example.
Branch: `feat/adaptive-exploratory-selector` | Delivery: one focused work-unit commit; no push/PR | Tests: `.venv/Scripts/python.exe -B -m unittest`

## Specs

### S1 — User request
> "dale implementalo y ejecuta una version como antes a ver como da todo"

Acceptance: preserve the existing `adaptive` selector; add the discussed experimental variant; run one comparable experiment and report fitness, gap, use/probability plots. The discussed comparison configuration is the prior run: official `mknapcb1`, index 0, seed 60, 1000 requested iterations, DDTW enabled, window 20.

### S2 — Algorithm assumption accepted from preceding proposal
Assumption: use the variant name `adaptive_exploratory`. Retain the existing EWMA reward score update for the selected MH (so zero reward decays its score); compute the variant's exploration component from inverse square-root per-pool use counts, and mix it with exploitation according to existing `epsilon`. Normalize probabilities; do not change baseline `adaptive` behavior. Do not treat an unselected MH as empirically improving.

Acceptance: deterministic tests show the selected no-reward MH's next probability decreases relative to less-used pool members, unselected/never-used MHs gain relative exploration share, probabilities sum to 1, floors remain positive, and baseline selector behavior is unchanged.

## Tasks

- T1 (S2) — Done; delegated worker; RED observed then GREEN, 21 focused selection/pipeline tests passed. Included in the feature delivery commit.
- T2 (S1-S2) — Done; delegated worker; configuration, both CLI/help and docs integrated; four baseline dry-run stdout hashes unchanged, exit 0, empty stderr. Included in the feature delivery commit.
- T3 (S1) — Done; inline; job-20 exit 0. Fitness 24280, gap 0.414257%, 1034 iterations. Probability PNG visually inspected. Included with earlier authorized session work in the feature delivery commit.

## Log

- L1: "dale implementalo y ejecuta una version como antes a ver como da todo"
- L2: Prior proposal accepted: add a separate variant without changing `adaptive`; its experimental probability mass favors under-used MHs, compare under matched conditions, and keep the original as baseline unless results justify promotion.
- L3: Pilot matched baseline fitness/gap/iterations (24280 / 0.414257% / 1034). Both used VNS 10 times; experimental final VNS probability 82.3163% versus baseline 85%. Experimental chose TS once instead of ILS once. Elapsed solver time 626.312s versus baseline 89.407s; cause not established. No default promotion. No full-suite rerun; 21 focused tests passed. Changes remain uncommitted because shared files contain prior work.

- L4: Final suites: adaptive 86/87 passed, general 41/41 passed (127/128 total). Sole failure: preexisting CRLF in submit_adaptive_mkp_arrays.sh. User explicitly selected "Subir sin corregir": publish with that known failure unchanged. Include authorized session code/tests/docs; exclude a/, tracked bytecode, preexisting odd/tasks/time-series-presentation.md changes and experiment outputs.
