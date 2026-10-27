# Inferencia HRES2-H2

Promoted comparison: independent-sample Mann-Whitney U (two-sided) with Holm across comparisons against this reference variant. W/T/L requires lower physical LCOE and adjusted p < 0.05. Every expected record of the reference and of that comparator must be finite physical feasible. An incomplete reference skips every comparison. A feasible subset is descriptive only: no p-value, W/T/L, imputation, or feasible-only superiority is produced. Records stay in the counts.

Paired Wilcoxon and Friedman are not valid without proof of paired initial conditions and are not used for inference.
Paired diagnostics are not valid.

- **Número de registros de la referencia:** 31
- **Algoritmo de Control (Referencia):** `Hybrid DDTW`

| Algoritmo | best (min) | median | n_records | n_finite_feasible | Holm p | WTL |
|---|---|---|---|---|---|---|
| Hybrid DDTW | 0.267159 | 0.267159 | 31 | 31 | null | reference |
| PSO | 0.267159 | 0.274579 | 31 | 31 | 8.481791e-04 | W |
| GWO | 0.267159 | 0.267160 | 31 | 31 | 1.979123e-11 | W |
| WOA | 0.267159 | 0.267525 | 31 | 31 | 8.481791e-04 | W |
| EHO | 0.267159 | 0.274579 | 31 | 31 | 6.353696e-07 | W |
| ACO | 0.267159 | 0.267159 | 31 | 31 | 1.035454e-10 | W |
| ILS | 0.267160 | 0.267176 | 31 | 31 | 1.979123e-11 | W |
| SA | 0.267170 | 0.281419 | 31 | 31 | 1.979123e-11 | W |
| TS | 0.267160 | 0.267181 | 31 | 31 | 1.979123e-11 | W |
| VNS | 0.267160 | 0.267165 | 31 | 31 | 1.979123e-11 | W |
