# DTW Optimization — Project Review & Context

> Review date: 2026-07-28
> Scope: Full codebase + experimental results + theoretical context

---

## 1. Project Overview

**Goal**: Compare DTW-based stagnation detection strategies for adapting metaheuristic (MH) parameters on the Multidimensional Knapsack Problem (MKP).

**Core idea**: Instead of fixed explore/exploit schedules, use Dynamic Time Warping to detect when the fitness convergence curve resembles a plateau (stagnation), then dynamically switch MH parameters between exploration and exploitation modes.

**Problem**: Multidimensional Knapsack Problem (MKP) — Chu & Beasley instances from OR-Library (mknapcb1–9).

**Metaheuristics used**:
| MH | Type | Explore params | Exploit params |
|---|---|---|---|
| BinaryPSO | Swarm | w=0.9, c1=2.5, c2=0.5 | w=0.729, c1=1.49445, c2=1.49445 |
| GeneticAlgorithm | Evolutionary | cx=0.6, mut=0.15 | cx=0.9, mut=0.01 |
| BinaryGWO | Swarm | a=2.0 | a=0.5 |
| BinaryDE | Evolutionary | F=0.9, CR=0.3 | F=0.5, CR=0.9 |

---

## 2. Strategy Versions (6 total)

| # | Version | Type | Strategy | Description |
|---|---|---|---|---|
| 1 | Vanilla-Explotación | Baseline | — | MHs forced to pure exploit mode entire run |
| 2 | Vanilla-Exploración | Baseline | — | MHs forced to pure explore mode entire run |
| 3 | Binary-Simple | DTW Binary | A3 — Fire D₂ | Boolean decision: `fire = D₂ ≤ θ_c`. Simplest question. |
| 4 | Binary-Complex | DTW Binary | A4 — 3 conditions + patience | Baseline DTW: plateau + D₂ + D₁/Δ + temporal confirmation. Max robustness. |
| 5 | Continuous-Simple | DTW Continuous | B3 — D₂ direct | Continuous intensity: `intensity = 1 − clip(D₂/(θ_c × scale))`. Simplest regulator. |
| 6 | Continuous-Complex | DTW Continuous | B1 — Sigmoid Δ | Sigmoidal intensity on normalized delta. Non-linear response with dead zone. |

---

## 3. DTW Monitor — How It Works

### 3.1 Core Mechanism

The `StagnationMonitor` (in `mkp_common/monitor.py`) compares the fitness convergence curve against two reference patterns:

- **Ramp (ideal progress)**: Linear increase with slope `min_slope`
- **Constant (ideal stagnation)**: Flat line at starting value

It computes:
- **D₁** = DTW(observed_window, ramp) → distance to progress
- **D₂** = DTW(observed_window, constant) → distance to plateau
- **Δ** = D₁ − D₂ → net balance (positive = stagnation, negative = progress)

### 3.2 DDTW (Derivative DTW)

By default (`use_ddtw=True`), the monitor uses Derivative DTW which operates on first differences instead of raw values. This makes the metric **scale-invariant**: a plateau has derivative ≈ 0 regardless of whether fitness is 10 or 10,000.

### 3.3 Adaptive Thresholds

Thresholds are computed as moving percentiles of the metric history:
- **θ_c** = percentile(D₂_history, p_low=30) → "what is abnormally low for D₂?"
- **θ_r** = percentile(D₁_history, p_high=70) → "what is abnormally high for D₁?"
- **θ_δ** = percentile(Δ_history, p_high=70) → "what is abnormally biased?"

This auto-calibrates to each MH, instance, and search phase.

### 3.4 DTW Configuration (shared across strategies)

| Parameter | Value | Justification |
|---|---|---|
| `window` | 20 | Balance reactivity vs stability |
| `band` | 2 | Sakoe-Chiba band ±2 positions, O(n) cost |
| `min_slope` | 2.0 | Steep ramp → D₂ dominates detection |
| `use_ddtw` | True | Scale invariance via derivatives |
| `adapt_thresholds` | True | Auto-calibrate via percentiles |
| `p_low` | 30 | 30th percentile for θ_c |
| `p_high` | 70 | 70th percentile for θ_r, θ_δ |

---

## 4. Decision Strategies

### 4.1 Binary Strategies (boolean fire)

**A3 — Binary-Simple (Fire D₂)**:
```python
fire = D₂ ≤ θ_c
```
Single condition: "is the curve flat?" No patience, no plateau check.

**A4 — Binary-Complex (3 conditions + patience)**:
```python
cond_plateau = no_improve_len >= plateau_max  # plateau_max=4
cond_constant = D₂ ≤ θ_c
cond_ramp = (D₁ ≥ θ_r) OR (Δ ≥ θ_δ)
trigger = cond_plateau AND cond_constant AND cond_ramp
fire = trigger_streak >= patience  # patience=2
```
Triple-gated with temporal confirmation.

### 4.2 Continuous Strategies (intensity ∈ [0,1])

**B3 — Continuous-Simple (D₂ direct)**:
```python
ratio = D₂ / (θ_c × scale + eps)  # scale=2.0
intensity = 1 − clip(ratio, 0, 1)
```
Low D₂ (plateau) → high intensity → explore. High D₂ (active) → low intensity → exploit.

**B1 — Continuous-Complex (Sigmoid Δ)**:
```python
r_balance = Δ / (θ_δ + eps)
intensity = sigmoid(K × (r_balance − center))  # K=5.0, center=0.5
```
Non-linear response: small Δ changes near center produce sharp intensity shifts.

---

## 5. Architecture & Code Quality

### 5.1 Strengths

1. **Clean Strategy pattern**: `runner.py` accepts `fire_fn` injection → strategies are decoupled from execution loop
2. **Centralized config**: `mkp_common/config.py` controls all strategies → single change point
3. **Proper statistical analysis**: Wilcoxon signed-rank (paired by seed) + Holm-Bonferroni correction for multiple comparisons
4. **Scale invariance**: DDTW + adaptive percentiles → same config works across instances (n=100 to n=500)
5. **Reproducibility**: Fixed seeds per epoch, deterministic repair operator

### 5.2 Observations / Potential Improvements

1. **Duplicated runners**: `continuous_simple/runner.py` and `continuous_complex/runner.py` duplicate the generic runner logic (~250 lines). Could generalize with an `intensity_fn` parameter.

2. **`fire_count = 0` in continuous strategies**: Hardcoded in continuous runners. Consider counting `mode == "explore"` transitions for consistency.

3. **DDTW `first_diff` prepend**: `first_diff` prepends `x[0]` so derivative has n elements with first = 0. Valid convention but should be documented.

4. **Repair complexity**: `reparar()` checks `np.all(r @ x <= b)` in each greedy-add iteration → O(m·n) per item. Could optimize with accumulated consumption for large instances.

5. **GA elitism**: Hardcoded `elitism = 2`. Consider making configurable for research flexibility.

6. **Baseline choice**: Analysis compares against `vanilla_exploracion` (always-explore). Paper must be explicit about this choice.

---

## 6. Experimental Results — mknapcb4[0]

### 6.1 Statistical Comparison (Wilcoxon + Holm-Bonferroni)

```
Holm-Bonferroni alpha = 0.0500 (20 comparisons)

PSO  Exploration-only: 22919.3 ± 68.9
     Exploitation-only: Δ=-96.0, p=0.000***  (WORSE)
     Binary-Simple:     Δ=-19.0, p=0.272
     Binary-Complex:    Δ=+29.0, p=0.252
     Continuous-Simple: Δ=+25.0, p=0.203
     Continuous-Complex:Δ=+21.0, p=0.822

GA   Exploration-only: 22879.9 ± 81.5
     Exploitation-only: Δ=-26.0, p=0.139
     Binary-Simple:     Δ=+47.0, p=0.078
     Binary-Complex:    Δ=+11.0, p=0.445
     Continuous-Simple: Δ=+32.0, p=0.499
     Continuous-Complex:Δ=+27.0, p=0.769

GWO  Exploration-only: 22593.8 ± 59.1
     Exploitation-only: Δ=-10.0, p=0.969
     Binary-Simple:     Δ=+27.0, p=0.018¹   (nominally significant, NOT after Holm)
     Binary-Complex:    Δ=+0.0,  p=0.547
     Continuous-Simple: Δ=+0.0,  p=0.394
     Continuous-Complex:Δ=+22.0, p=0.462

DE   Exploration-only: 22910.4 ± 63.4
     Exploitation-only: Δ=-29.0, p=0.024¹   (nominally significant, NOT after Holm)
     Binary-Simple:     Δ=+11.0, p=0.196
     Binary-Complex:    Δ=+25.0, p=0.799
     Continuous-Simple: Δ=+4.0,  p=0.487
     Continuous-Complex:Δ=+31.0, p=0.576
```

### 6.2 Key Findings (mknapcb4[0])

- **No DTW strategy is statistically significantly better** than exploration-only after Holm-Bonferroni correction
- Deltas are small (+11 to +47 fitness over ~23,000 optimal)
- `Binary-Simple` on GWO has p=0.018 but doesn't survive Holm correction (marked ¹)
- `Exploitation-only` is significantly **worse** than exploration-only in PSO (p=0.000***)
- Exploration-only is a strong baseline for this instance

### 6.3 Interpretation Notes

1. Results are instance-specific (mknapcb4[0] only). Need to check all 9 instances.
2. The advantage of DTW may be in **convergence speed** (not just final fitness) or on harder instances.
3. Exploitation-only being worse confirms the hypothesis that exploration is necessary for MKP.

---

## 7. Project Structure Reference

```
DTW_optimization/
├── mkp_common/              # Shared code
│   ├── mh/                  # Metaheuristics: BinaryPSO, GA, BinaryGWO, BinaryDE
│   ├── config.py            # Central config (population, iterations, epochs, DTW params)
│   ├── base.py              # BaseMH interface + adapt_continuous
│   ├── monitor.py           # StagnationMonitor: DTW + 3 conditions + adaptive thresholds
│   ├── problem.py           # OR-Library instance loader + greedy repair
│   ├── runner.py            # Generic loop with injectable fire_fn (Strategy pattern)
│   ├── stats.py             # Wilcoxon + Holm-Bonferroni + table formatters
│   └── results.py           # JSON result save/load
├── vanilla_explotacion/     # Version 1: Baseline — pure exploit
├── vanilla_exploracion/     # Version 2: Baseline — pure explore
├── binary_simple/           # Version 3: A3 — Fire D₂
├── binary_complex/          # Version 4: A4 — 3 conditions + patience
├── continuous_simple/       # Version 5: B3 — D₂ direct continuous
├── continuous_complex/      # Version 6: B1 — Sigmoid Δ
├── contexto/oficial/        # Theoretical documents (paper references)
├── instances/               # Chu & Beasley OR-Library: mknapcb1..9
├── analisis/                # Statistical analysis + boxplots
├── results/                 # Results per strategy per instance
│   └── estadistico/         # Statistical comparison outputs (all 9 instances)
├── run_all.py               # Sequential execution of all strategies
├── run_all_hpc.py           # Parallel execution for HPC/SLURM
└── run_dtw.sh               # SLURM submission script
```

---

## 8. References (from contexto/oficial/)

- **DTW**: Sakoe & Chiba (1978) — Dynamic Programming Algorithm Optimization for Spoken Word Recognition
- **DDTW**: Keogh & Pazzani (2001) — Derivative Dynamic Time Warping
- **Sakoe-Chiba band**: Sakoe & Chiba (1971) — constraint for alignment window
- **MKP instances**: Chu & Beasley (1998) — OR-Library benchmark

---

## 9. Next Steps for Academic Writing

- [ ] Review results across all 9 instances (not just mknapcb4[0])
- [ ] Identify patterns: which MH benefits most from DTW? which strategy works best?
- [ ] Analyze convergence curves (not just final fitness) — DTW may improve speed
- [ ] Draft methodology section based on contexto/oficial/ documents
- [ ] Prepare tables/figures for paper
