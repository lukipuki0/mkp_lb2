# DTW Optimization — Academic Paper Writing Context

> **Purpose**: This file provides full context for continuing the academic paper writing. Read this first before any writing work.

---

## 1. Project Summary

**Research topic**: Trajectory-driven online parameter control framework based on Dynamic Time Warping (DTW) for metaheuristic adaptation on the Multidimensional Knapsack Problem (MKP).

**Core idea**: Instead of fixed explore/exploit schedules, use DTW to detect when the fitness convergence curve resembles a plateau (stagnation), then dynamically switch MH parameters between exploration and exploitation modes.

**Key files for reference**:
- `REVIEW_PROJECT.md` — Full project review with architecture, code analysis, results
- `mkp_common/monitor.py` — DTW stagnation monitor implementation
- `mkp_common/config.py` — All parameters and configurations
- `contexto/oficial/` — Theoretical documents (DTW fundamentals, strategy descriptions, MH docs)

---

## 2. Paper Structure (Agreed)

```
1. Introduction ✅ (written by user in English)
2. Background
   2.1 MKP ✅ (corrected)
   2.2 Metaheuristics ✅ (corrected)
   2.3 DTW as Shape Sensor ✅ (corrected)
3. Proposed Approach
   3.1 Framework Overview ✅ (corrected)
   3.2 DTW-based Stagnation Sensor
       3.2.1 Reference Patterns ✅ (corrected)
       3.2.2 Distance Metrics ✅ (verified)
       3.2.3 Adaptive Thresholds ✅ (verified)
   3.3 Decision Strategies ✅ (corrected, all 6 strategies)
4. Experimental Setup (PENDING)
5. Results & Analysis (PENDING)
6. Conclusions (PENDING)
```

---

## 3. Writing Principles Established

These rules MUST be followed throughout the paper:

1. **Only document what's necessary and useful** — Don't explain implementation details that don't affect results
2. **Go directly to what's used** — Don't document unused configuration options or alternative modes
3. **Verify all formulas against code** — Every formula must match the actual implementation in `monitor.py`
4. **No alphanumeric codes** — Use full strategy names:
   - ~~A3~~ → **Binary-Simple**
   - ~~A4~~ → **Binary-Complex**
   - ~~B3~~ → **Continuous-Simple**
   - ~~B1~~ → **Continuous-Complex**
5. **Background = existing work, Methodology = your contribution** — DTW fundamentals go in Background, the stagnation sensor goes in Proposed Approach
6. **Use original paper notation** — DTW formulas match Sakoe & Chiba (1978) and Keogh & Pazzani (2001)
7. **Language** — Paper body in Spanish (user will translate to English at the end)

---

## 4. Section-by-Section Status

### 2.1 MKP ✅
- Mathematical formulation included
- References Chu & Beasley (1998)
- Justification as benchmark added

### 2.2 Metaheuristics ✅
- Table with 4 MHs: BPSO, GA, BGWO, BDE
- Columns: Family, Controlled Parameters, Mechanism, Binarization
- GA uses native binary encoding; others use sigmoid transfer
- Reference to Syswerda (1989) added for GA crossover uniforme

### 2.3 DTW as Shape Sensor ✅
- 1-based indexing (matching original papers)
- Local cost: d(a_i, b_j) = (a_i - b_j)² (Keogh & Pazzani)
- Normalized distance: D[n,m] / (n+m) (Sakoe & Chiba)
- DDTW formula with discrete derivative
- Sakoe-Chiba band with w=2
- Final paragraph links to Section 3.2

### 3.1 Framework Overview ✅
- Title: "Framework de Control de Parámetros Basado en Trayectoria"
- Three components: MH, DTW monitor, decision module
- Three families: Vanilla, Binary, Continuous
- Warm-up period explained (first W iterations, no signals)
- References to subsections added

### 3.2.1 Reference Patterns ✅
- Ramp: r_i = x_1 + s_min · (i-1), with s_min = 2.0
- Constant: c_i = x_1
- No auto-calculation mode documented (not used in experiments)

### 3.2.2 Distance Metrics ✅
- D1 = DDTW(X, R) — distance to ramp
- D2 = DDTW(X, C) — distance to plateau
- Δ = D1 - D2 — net balance
- Interpretation of each metric

### 3.2.3 Adaptive Thresholds ✅
- θ_c = P_30(H_{D2}) — 30th percentile of D2 history
- θ_r = P_70(H_{D1}) — 70th percentile of D1 history
- θ_δ = P_70(H_Δ) — 70th percentile of Δ history
- No intermediate variables (p_low, p_high) — directly use percentiles

### 3.3 Decision Strategies ✅
- All 6 strategies documented with formulas
- Summary table included
- Controlled comparisons explained (Binary-Simple vs Continuous-Simple, etc.)

---

## 5. Verified Formulas (from code)

### DTW Configuration (actual values)
| Parameter | Value |
|---|---|
| window (W) | 20 |
| band (w) | 2 |
| min_slope (s_min) | 2.0 |
| use_ddtw | True |
| adapt_thresholds | True |
| p_low | 30 |
| p_high | 70 |

### Decision Strategy Parameters
| Strategy | Key Parameters |
|---|---|
| Binary-Simple | fire = D₂ ≤ θ_c |
| Binary-Complex | plateau_max=4, patience=2 |
| Continuous-Simple | σ=2.0 |
| Continuous-Complex | k=5.0, c=0.5, ε=10⁻¹⁰ |

### MH Parameters (Exploit → Explore)
| MH | Exploit | Explore |
|---|---|---|
| BPSO | w=0.729, c1=1.49445, c2=1.49445 | w=0.9, c1=2.5, c2=0.5 |
| GA | cx=0.9, mut=0.01 | cx=0.6, mut=0.15 |
| BGWO | a=0.5 | a=2.0 |
| BDE | F=0.5, CR=0.9 | F=0.9, CR=0.3 |

---

## 6. Next Steps

### Immediate (Section 4 — Experimental Setup)
- [ ] Define instances (Chu & Beasley mknapcb1-9)
- [ ] Document MH configurations with exact parameter values
- [ ] Define evaluation metrics (fitness, gap to optimal, convergence)
- [ ] Document statistical methodology (Wilcoxon + Holm-Bonferroni)
- [ ] Number of epochs, seeds, computational budget

### After (Section 5 — Results & Analysis)
- [ ] Present results across all 9 instances
- [ ] Compare strategies within each MH
- [ ] Analyze convergence behavior (not just final fitness)
- [ ] Statistical significance analysis

### Final (Section 6 — Conclusions)
- [ ] Summarize findings
- [ ] Limitations
- [ ] Future work

---

## 7. Key References

- Sakoe & Chiba (1978) — DTW original
- Keogh & Pazzani (2001) — DDTW
- Chu & Beasley (1998) — MKP instances
- Kennedy & Eberhart (1997) — PSO
- Holland (1975), Goldberg (1989) — GA
- Syswerda (1989) — Uniform crossover
- Mirjalili et al. (2014) — GWO
- Emary (2016) — Binary GWO
- Storn & Price (1997) — DE
- Pampara (2006) — Binary DE

---

## 8. Code Reference for Verification

When verifying formulas against code, check these files:

```
mkp_common/monitor.py     — DTW monitor (D1, D2, Δ, thresholds)
mkp_common/config.py      — All parameter values
mkp_common/mh/pso.py      — BPSO implementation
mkp_common/mh/ga.py       — GA implementation
mkp_common/mh/gwo.py      — BGWO implementation
mkp_common/mh/de.py       — BDE implementation
mkp_common/runner.py      — Main execution loop
mkp_common/problem.py     — MKP instance loader + repair
```

---

## 9. Current Results Summary (for reference)

Results exist for all 9 Chu & Beasley instances (mknapcb1-9). Key finding from mknapcb4[0]:

- No DTW strategy is statistically significantly better than exploration-only after Holm-Bonferroni correction
- Exploitation-only is significantly worse than exploration-only in PSO (p=0.000***)
- Small deltas (+11 to +47 fitness over ~23,000 optimal)

Full results in: `results/estadistico/`
