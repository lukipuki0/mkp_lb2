# Saved GWO-DE DDTW results

Verbatim saved artifacts from three selected campaigns. The folder keeps the requested DTW name; the recorded signal is **DDTW** (`signal: ddtw`, `use_ddtw: true`).

| Campaign | Scope | Completed runs | Scientific files |
|---|---|---:|---:|
| `gwo_de_22349/` | Four adaptive policies: `binary_simple`, `binary_complex`, `continuous_simple`, `continuous_complex` | 3,348 | 1,161 |
| `gwo_de_vanilla_22875/` | Two fixed policies: `vanilla_exploitation`, `vanilla_exploration` | 1,674 | 675 |
| `gwo_de_sensitivity_22767/` | Descriptive one-at-a-time sensitivity: 19 configurations (baseline + 18 variations), two binary policies | 3,534 | 1,034 |

## Reading the campaigns

The first two campaigns together cover **six policies and 5,022 runs**, matched by 27 cases (`mknapcb1`–`mknapcb9`, instance indices 0/1/2) and seeds 1–31. They are separate campaigns, not six modes from one original campaign. The vanilla policies are fixed-profile **GWO-DE hybrid** variants, not standalone GWO or standalone DE. Matching case/seed identifiers does not certify identical initial populations.

Sensitivity covers only `mknapcb1`, indices **0/15/29**, seeds 1–31, and `binary_simple`/`binary_complex`: 19 × 2 × 3 × 31 = **3,534 runs**. Its descriptive OAT design varies window, band, thresholds, plateau limit, or patience; it is not a factorial or inferential study.

## Contents and limits

All 2,870 scientific files (385,164,090 bytes) are retained: run CSVs, available JSONL, statistics/comparisons, original configs and manifest, winner JSON, saved winner temporal CSV/NPZ traces, PNG/PDF plots, summaries, and case run logs where present. Nonwinner traces were not retained by the sources; their saved run records are CSV-only in the ordinary campaigns (sensitivity also has aggregate JSONL).

Temporary `.staging` copies are excluded; caches, matplotlib configuration/cache folders, `__pycache__`, and `.pyc` artifacts are also excluded if present. No other campaigns are included. Original configs/logs are unchanged, including system paths and host metadata. A text credential heuristic found no matches; that is not a security guarantee.

Historical configs and reported optimum references do not certify exact source-code revision provenance or every optimum. This archive makes no claim of statistical significance or guaranteed speedup. No experiments were rerun for packaging.

## Integrity

`SHA256SUMS` covers every scientific file and this README using paths relative to this folder; it excludes itself. From this folder, run `sha256sum -c SHA256SUMS`. All scientific copies were checked against their original source SHA-256 hashes.
