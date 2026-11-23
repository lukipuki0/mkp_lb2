# MKP gap and temporal-diagnostic figures

Figures for Section 6 of `overleaf/gwo_de_dtw_paper`. They replace, in the text, the single
27-case figure `figura/mkp-gap-by-case.pdf`. That file remains in place, unchanged.

## Reproduce

From the repository root, using the installed matplotlib 3.10, numpy 2.2 and pandas 2.3:

```
python figuras/resultados-gwo-de/generate_gap_figures.py --check-only   # checks only, writes nothing
python figuras/resultados-gwo-de/generate_gap_figures.py                # checks, then writes the 10 files
```

Options: `--results` (campaign root, default `gwo_de_dtw/results/resultados gwo de dtw`),
`--tex` (results.tex used for parity checks), `--out` (default `overleaf/gwo_de_dtw_paper/figura`),
`--only STEM` (repeatable; run every check, then write only the named figure, e.g.
`--only mkp-temporal-mknapcb1-0-binary-complex`).

The Binary-Complex figure was written once with
`python -B figuras/resultados-gwo-de/generate_gap_figures.py --only mkp-temporal-mknapcb1-0-binary-complex`,
so the gap and Binary-Simple files were not rewritten. A full run to a temporary `--out` directory
reproduced those eight files byte for byte (hashes below).
Two consecutive runs gave byte-identical outputs: PDF dates and producer metadata are suppressed,
and fonts are embedded as TrueType (DejaVu Sans).

## Gap figures (`mkp-gap-n100`, `-n250`, `-n500`)

- Data: `runs.csv` of `gwo_de_22349` (four adaptive policies) and `gwo_de_vanilla_22875` (two
  vanilla policies), 27 cases x 6 policies x 31 seeds. Gap = 100 (f* - mean profit) / f*, computed
  per case. Profits are never pooled across cases.
- Checks run before any file is written (the script exits on the first failure): every source file
  matches `SHA256SUMS` (raw bytes, or LF-normalised bytes when Git's `core.autocrlf` changed the
  checkout); each case-policy pair has 31 runs with seeds 1-31 and status `ok`; f* equals the
  Table `tab:mkp_best` column; the raw mean matches `statistics.csv` (profit within 1e-6, gap within
  1e-9); every gap rounded to two decimals equals its cell in Table `tab:mkp_gap`. Result: 54 points
  per figure and 162 in total, all 162 equal to the table. Unrounded ranges: n=100 0.8114-3.4376 %,
  n=250 2.5463-5.8151 %, n=500 4.0048-6.8282 %.
- Design: one point per case and policy, with a fixed horizontal offset inside each case column.
  No lines are drawn, because the three instance indices are unrelated problems and a line would
  suggest a trajectory. Policies are told apart by Okabe-Ito colour and marker shape, with the same
  mapping as the original figure. The x labels are the index `[i]` under each `mknapcbg (m = ...)`
  group. All three y axes span 4 percentage points from different starting values, so slopes
  compare across figures without forcing one range on all of them. Size 5.4 x 3.0 in (13.7 cm wide,
  for the 394 pt MDPI `\textwidth`), with 8-8.5 pt text at print size.
- Not shown: dispersion. The standard deviations are in Table `tab:mkp_mean`.

## Temporal diagnostics (`mkp-temporal-mknapcb1-0-binary-simple`, `-binary-complex`)

Only the saved best run of each adaptive policy and case keeps a temporal series (108 runs in
`gwo_de_22349`; the vanilla runs have no monitor).

Candidates inspected as rendered images (`best_run/temporal_diagnostics.png`):

| Case | Policy | Seed | Changes | Observation |
|---|---|---:|---:|---|
| mknapcb1[0] | Binary-Simple | 31 | 5 | Clear rise/return/resume sequence; long exploration plateau after generation 436 |
| mknapcb1[0] | Continuous-Simple | 5 | 9 | Four improvement-driven drops of I_t; intensity shows partial levels |
| mknapcb1[0] | Binary-Complex | 26 | 11 | Readable, but its rule adds no-improvement, D_R and Delta conditions; denser |
| mknapcb1[0] | Continuous-Complex | 17 | 1 | Single crossing (as in every Continuous-Complex run); little temporal structure |
| mknapcb4[0] | Binary-Complex | 1 | 7 | Best single run of the campaign (gap 0.09 %); similar pattern to seed 26, different case |

A scan of all 108 saved series (number of profile changes, last improving generation) came before
the visual inspection.

Selected: mknapcb1[0] Binary-Simple seed 31 and Binary-Complex seed 26. The first version paired
Binary-Simple with Continuous-Simple seed 5; the author asked for the two binary policies instead,
so Continuous-Simple was replaced by Binary-Complex. Reasons for this pair:

1. Same case as the n=100 panel of Figure `fig:mkp_convergence`, which already plots these two
   runs' best-so-far curves, so no new case is added. Only the best run is saved, so these are the
   only traces of the two binary policies on this case (seed 26 is `best_seed` in `statistics.csv`).
2. Both policies switch between the same two fixed profiles, restart the monitor after every change
   and return to exploitation with the same rule (no-improvement counter below `plateau_max`). They
   differ in the entry rule: Binary-Simple reads D_C against theta_C only; Binary-Complex needs
   C_noimp, C_plateau and C_progress (D_R >= theta_R or Delta >= theta_Delta) on `patience` = 8
   consecutive ready observations. Each figure shows only the signals its rule uses.
3. The two runs have different seeds and the same length (1000 generations). They are not a paired
   comparison, and the text does not compare the timing of their changes.

mknapcb4[0] seed 1 was not used: it would add a case that no other figure shows.

Why they were re-rendered: the saved diagnostics are 12 x 13 in pages with six panels and about
10 pt text. At 13.9 cm width the text would print at about 4.5 pt. In addition, their distance panels
draw straight segments across the monitor warm-up gaps of binary runs and, for Binary-Complex, plot
only the samples with percentile thresholds, hiding the fixed warm-up thresholds that the rule used.
The new figures plot `temporal_series.csv` columns as saved. Warm-up NaNs stay as gaps, and isolated
ready samples are drawn as dots. Distance axes are symlog (linear below 10), because the distances
run from 0 to about 1,400.

- Binary-Simple (3 panels, 5.4 x 4.1 in): `f_star`, `D_C` and `theta_C`, `mode_used`.
- Binary-Complex (4 panels, 5.4 x 5.2 in): `f_star`; `D_C`, `theta_C`, `D_R`, `theta_R`;
  `delta` and `theta_delta` (linear, 0-100); and one row per condition, from `ready`,
  `no_improve_len >= plateau_max` (C_noimp), `D_C <= theta_C` (C_plateau),
  `D_R >= theta_R or delta >= theta_delta` (C_progress) and the saved `fire` flag
  (S_t >= tau_pat). Dark = holds, light grey = evaluated and fails, blank = monitor not ready.
  The conditions are comparisons of saved columns with saved thresholds; no distance or threshold
  is recomputed. The rule constants (`plateau_max` 15, `patience` 8) come from
  `temporal_metadata.json`. The streak and `fire` stay set during exploration, where the rule does
  not read them; they are plotted as saved.

Checks on the selected runs: `SHA256SUMS` matches; metadata seed and policy match; the seed equals
`best_seed` in `statistics.csv`; final `f_star` equals the `runs.csv` and metadata best (24,264 and
24,252); the number of `mode_used` changes equals `runs.csv` `changes` (5 and 11); `use_ddtw` is
true; `mode_used[t] = s[t-1]`. For Binary-Complex also: distances and thresholds are present
exactly at ready generations; `delta = D_R - D_C`; the streak rebuilt from C_stag equals the saved
`trigger_streak`; `streak >= patience` equals the saved `fire`; and the rule (enter on `fire` in
exploitation, stay in exploration while `no_improve_len >= plateau_max`, no decision when not
ready) reproduces the saved `s` at all 1000 generations. Changing `patience` to 7 or one saved
`theta_C` value makes these checks fail.

Observed in the Binary-Complex run (used in the text): exploitation 1-56, 98-144, 185-237,
605-651, 717-763, 882-937; exploration in between and from 938 to the end. C_progress holds on all
532 ready observations. Entries at 98, 605 and 717 come 47 generations after the interval starts
(39 warm-up + 8 confirmations). C_plateau first holds at 230 after the improvement at 191 leaves
the window, and at 49 and 930 when percentile thresholds replace the warm-up values
(D_C 698 <= 722 and 63 <= 63). The five returns follow improvements at 97, 184, 604, 716 and 881,
all on ready observations. The improvement at 243 precedes the first ready observation (277,
counter 34). Final profit 24,252 at generation 907.

Source SHA-256 (from `SHA256SUMS`, all under `gwo_de_22349/mknapcb1/inst_00/`):

| File | SHA-256 |
|---|---|
| `binary_simple/best_run/temporal_series.csv` | `a12c8a8d4963755ab04b0a2480a095b47e00d3b4678b39f8887819580ad9b554` |
| `binary_simple/best_run/temporal_metadata.json` | `90be89dd8c02450b2d2c19a15726727d95d472ed95b1b26639d43f75a79656fc` |
| `binary_complex/best_run/temporal_series.csv` | `c0370c9bf6a435af31324f19c5c6b88ac30ee2615638ff58bc228059ba1441ad` |
| `binary_complex/best_run/temporal_metadata.json` | `c564804f65876f8d4f35e057f5a469b17f35d23c64de0cd56be893bc4f285138` |
| `binary_complex/best_run/result.json` | `7d87d1c2948cc2cb038ec9c2ef23f97eb61101bed226e286677d74896e8b9e8e` |
| `runs.csv` | `1ac2ed8aa57ecdb5a8ce6cadbabd5ca0eb921910d3d96883fb6beffd13b27608` |
| `statistics.csv` | `3d6bad884569b6b2bdc33b6af0738f2d8517a3252d2c160bfffb52f18e0c4e8f` |

## Outputs (SHA-256)

| File | SHA-256 |
|---|---|
| `mkp-gap-n100.pdf` | `a01db6f8be0e37d9fbbe7ff83201d47d5bf920e7c544fe1906d906610d3f4c1e` |
| `mkp-gap-n100.png` | `3b1273d30f60212d9fbc9f59c4181bd996ac89ac794e6aefbd0f1c04c3fca797` |
| `mkp-gap-n250.pdf` | `876ad4e409f2e021d339a816ab0276d3d7a6d22e15ce4141d71881e03a64e9b1` |
| `mkp-gap-n250.png` | `815a1cb40ff036a40a9f54a341b142ee64153da11a042f7ce361b951ae19ea96` |
| `mkp-gap-n500.pdf` | `cc06dd9b2bc7a276503f4af5f3f475d249cfe7e6807eca49d8535278cf0d9cbf` |
| `mkp-gap-n500.png` | `209cb758355d2f957672336900439dc5b24f7c8a665092d002eee75d956d9f30` |
| `mkp-temporal-mknapcb1-0-binary-simple.pdf` | `7a42e937729e25ba9839da1c3cef9b247c607b0f9b0e7428bb827893ee54b72b` |
| `mkp-temporal-mknapcb1-0-binary-simple.png` | `95fc120e8c4c77c4bbb000433b8f5f4affad706dd3b10e823dda8b8674160c99` |
| `mkp-temporal-mknapcb1-0-binary-complex.pdf` | `52cce3d0d0bdb5ab26d0b76919c01fdea10e9d1d35d4ef48528c7f5e904f6b29` |
| `mkp-temporal-mknapcb1-0-binary-complex.png` | `4a787af7c87fe75c4257a8409c466a44ad6f36a28ad653a18c9ec5c07aaf87d4` |

The earlier `mkp-temporal-mknapcb1-0-continuous-simple.pdf/.png` (SHA-256 `893935212fe1...` and
`7b6f6bd18003...`) remain in `figura/` unchanged, but the script no longer writes them and the text
no longer includes them.

## Limits

- The temporal figures show two winning runs with different seeds. They are not representative, do
  not summarise the 31 seeds, are not paired, and do not show that a profile change caused an
  improvement.
- The no-improvement counter is plotted only as the C_noimp row of the Binary-Complex figure, not
  as a count; the Binary-Simple figure does not show it. Its effect appears in the text as the
  return in the generation after each improvement.
- `fig:mkp_gap` is kept as a second label on the n=100 figure, because `discussion.tex` cites it
  for a statement about n=100 versus n=500. That sentence could cite all three figures instead.
