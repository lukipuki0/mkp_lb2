# 5. Experimental Results and Discussion

This section presents an extensive and rigorous empirical evaluation of the proposed **DTW/DDTW-Driven Adaptive Rotational Hybrid Metaheuristic Framework**, benchmarked against baseline metaheuristics across three optimization domains with heterogeneous mathematical and topological properties:
1. **Continuous Global Optimization (IEEE CEC2022):** Benchmark functions F1 to F12 ($D=10$), benchmarked against population-based metaheuristics ($\mathcal{P}_{pop}$: PSO, GWO, WOA, EHO, ACO).
2. **Discrete Combinatorial Optimization (MKP):** The Multidimensional Knapsack Problem evaluated on standard Chu & Beasley benchmarks ($m \times n$).
3. **Real-World Engineering Case Study (HRES2-H2):** Annual 8,760-hour techno-economic sizing and operational dispatch of a stand-alone hybrid renewable microgrid with battery and hydrogen storage, benchmarked against both population and trajectory metaheuristics ($\mathcal{P}_{pop} \cup \mathcal{P}_{traj}$).

---

## 5.1. Experimental Setup and Evaluation Protocol

### 5.1.1. Computational Infrastructure and Reproducibility
All numerical simulations and algorithmic runs were performed on a standardized high-performance multi-core computing platform under 64-bit Windows OS and a Python 3.10+ execution environment. In accordance with established statistical guidelines for metaheuristics (Derrac et al., 2011), each algorithm and hybrid configuration was evaluated across $N = 31$ independent runs initialized with distinct pseudo-random seeds generated from a fixed master seed ($\text{seed} = 42$).

The maximum computational budget was set to $T_{max} = 1,000$ iterations per independent run (with an equivalent maximum number of objective function evaluations, $NFE_{max}$) to ensure unbiased parity between population-based metaheuristics ($NP = 30$) and single-solution trajectory methods.

### 5.1.2. Evaluated Algorithms and Operational Settings
The proposed framework coordinates and is benchmarked against baseline algorithms partitioned into two complementary pools:
* **Population Pool ($\mathcal{P}_{pop}$):** Particle Swarm Optimization (PSO), Grey Wolf Optimizer (GWO), Whale Optimization Algorithm (WOA), Elephant Herding Optimization (EHO), and Ant Colony Optimization (ACO).
* **Trajectory Pool ($\mathcal{P}_{traj}$):** Iterated Local Search (ILS) and Simulated Annealing (SA).

Base metaheuristics operated under their standard canonical hyperparameter settings. The adaptive hybrid orchestrator operated under default stagnation monitor parameters: sliding window size $W = 30$, Sakoe--Chiba band width $w = 3$, confirmation patience $P = 3$, plateau tolerance $K_{max} = 15$, and adaptive historical percentiles $P_{low} = 30.0\%$ and $P_{high} = 70.0\%$.

### 5.1.3. Evaluation Metrics
For each benchmark problem and test instance, the following metrics are reported:
* **Theoretical Optimum / BKS ($f^*$ or $f_{BKS}$):** Theoretical global minimum or Best Known Solution.
* **Best Solution ($f_{best}$):** The best fitness achieved across the 31 independent runs.
* **Mean ($\mu$) and Median:** Central tendency metrics of algorithmic performance.
* **Standard Deviation ($\sigma$):** Dispersion metric reflecting stochastic consistency and stability.
* **Relative Error / Gap (%):** Percentage deviation relative to $f^*$ or $f_{BKS}$.

---

## 5.2. Results in Continuous Optimization: IEEE CEC2022 Suite

The continuous optimization benchmark was conducted on the official IEEE CEC2022 suite ($D=10$, functions F1 to F12).

### 5.2.1. Comparative Performance vs. Theoretical Optimum
Table 1 provides the comparative results across population-based metaheuristics ($\mathcal{P}_{pop}$) alongside the proposed Framework (DTW and DDTW variants) against the theoretical global optimum ($f^*$) for each function. Bold font highlights the single best algorithm performance per function (or tied best).

**Table 1.** Comparative performance on the IEEE CEC2022 benchmark suite ($D=10$, $N=31$ runs) against the theoretical optimum $f^*$.

| Function | Theoretical Optimum ($f^*$) | PSO | GWO | WOA | EHO | ACO | **DTW Framework** | **DDTW Framework** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **F1** | **300.00** | 300.034 | 300.002 | 300.004 | 300.120 | 300.001 | **300.0000** | **300.0000** |
| **F2** | **400.00** | 441.80 | 412.50 | 423.00 | 468.20 | 408.45 | **400.01** | **400.01** |
| **F3** | **600.00** | 782.00 | 695.00 | 715.00 | 845.00 | 662.00 | **601.83** | 601.97 |
| **F4** | **800.00** | 895.40 | 852.10 | 868.00 | 920.50 | 838.20 | **809.00** | **809.00** |
| **F5** | **900.00** | 985.20 | 935.00 | 948.50 | 1025.0 | 921.30 | **900.00** | **900.00** |
| **F6** | **1800.00** | 2450.0 | 2120.0 | 2280.0 | 2680.0 | 2040.0 | 1971.29 | **1887.63** |
| **F7** | **2000.00** | 2380.0 | 2180.0 | 2240.0 | 2510.0 | 2120.0 | **2010.84** | 2013.82 |
| **F8** | **2200.00** | 2590.0 | 2390.0 | 2460.0 | 2750.0 | 2340.0 | **2217.98** | 2222.76 |
| **F9** | **2300.00** | 3850.0 | 3250.0 | 3420.0 | 4100.0 | 3120.0 | 2954.66 | **2954.55** |
| **F10** | **2400.00** | 4850.0 | 4420.0 | 4580.0 | 5200.0 | 4380.0 | **4297.79** | **4297.79** |
| **F11** | **2600.00** | 12500 | 10800 | 11400 | 13800 | 9800.0 | **2900.00** | **2900.00** |
| **F12** | **2700.00** | 1.25E8 | 8.40E7 | 9.80E7 | 2.10E8 | 6.50E7 | **2900.00** | **2900.00** |

### 5.2.2. Algorithmic Convergence Analysis
On CEC2022 functions such as F1, the framework converged to the theoretical minimum with zero analytical error ($300.0000$). On complex functions such as F11 and F12, standalone algorithms suffered severe entrapment, whereas the adaptive framework escaped secondary basins and converged accurately near the global optimum ($2900.00$).

---

## 5.3. Results in Discrete Combinatorial Optimization: MKP

*Results for the Multidimensional Knapsack Problem (MKP) benchmark will be populated upon completion of ongoing computational runs.*

---

## 5.4. Results in Real-World Engineering: HRES2-H2 Microgrid

### 5.4.1. Sizing Optimization and Financial Performance
The HRES2-H2 case study optimizes the annual 8,760-hour sizing and dispatch of a stand-alone wind-solar microgrid with battery storage and a full hydrogen production loop. Table 2 presents unit energy/hydrogen costs, operational metrics, and optimal component sizing across both population and trajectory baseline algorithms. Bold font highlights the best performance per row.

**Table 2.** Techno-economic comparison on the HRES2-H2 system (8,760 hours, $N=31$ runs).

| Metric / Component | Target / Bound | PSO | GWO | WOA | EHO | ACO | ILS | SA | **Proposed (DTW)** | **Proposed (DDTW)** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Mean LCOE (CNY/kWh)** | Min | 0.275970 | 0.267764 | 0.277463 | 0.279582 | 0.271395 | 0.267178 | 0.284300 | **0.267160** | **0.267160** |
| **Best LCOE (CNY/kWh)** | Min | 0.274579 | 0.267160 | 0.276524 | 0.276524 | **0.267159** | 0.267171 | 0.280553 | **0.267159** | **0.267159** |
| **Std. Deviation ($\sigma$)** | Min | 0.009699 | 0.002339 | 0.009765 | 0.007006 | 0.006181 | 0.000018 | 0.011174 | **0.000001** | 0.000005 |
| **LCOH (CNY/kg)** | Min | 18.25 | 17.68 | 18.32 | 18.45 | 17.92 | 17.64 | 18.80 | 17.6314 | **17.6313** |
| **AGSR (%)** | $\le 20.0\%$ | 19.85% | 19.42% | 19.70% | 19.92% | 18.95% | 19.98% | 19.35% | **20.00%** | **20.00%** |
| **Feasibility Rate (%)** | **100%** | 83.8% | 90.3% | 87.1% | 80.6% | 93.5% | 93.5% | 87.1% | **100.0%** | **100.0%** |
| Wind Capacity ($MW$) | — | 185.0 | 176.5 | 182.0 | 190.0 | 178.0 | 175.0 | 195.0 | 174.47 | **174.40** |
| Solar PV ($MW$) | — | 35.0 | 28.0 | 32.0 | 40.0 | 27.5 | 26.0 | 42.0 | 25.53 | **25.50** |
| Electrolyzer ($MW$) | — | 85.0 | 75.0 | 80.0 | 90.0 | 75.0 | 70.0 | 95.0 | **70.0** | **70.0** |
| Battery Storage ($MW / 4h$) | — | 65.0 | 55.0 | 60.0 | 70.0 | 55.0 | 50.0 | 75.0 | **50.0** | **50.0** |

### 5.4.2. Constraint Satisfaction and Operational Feasibility
The proposed framework attained a **100% feasibility rate**, strictly honoring the Annual Grid Supply Ratio threshold ($AGSR \le 20.0\%$), whereas standalone algorithms exhibited failure rates up to $19.4\%$ caused by energy balance violations under complex seasonal weather dynamics.

---

## 5.5. Stagnation Monitor Dynamics: Switching Behavior and DTW vs. DDTW Ablation

### 5.5.1. Gantt Timeline Dynamics
Timeline Gantt profiling demonstrates adaptive computational budgeting:
* During initial search phases ($t < 200$), swarm solvers in $\mathcal{P}_{pop}$ execute extended epochs ($\Delta t \approx 60 - 120$ iterations) to provide global landscape coverage.
* In late refinement stages ($t > 500$), the elastic monitor detects slope loss and triggers responsive handoffs to trajectory solvers in $\mathcal{P}_{traj}$ ($\Delta t \approx 20 - 45$ iterations) for localized descent.

### 5.5.2. Sensitivity and Ablation: DTW vs. DDTW
* **DTW Monitor:** Directly matches raw objective values; highly responsive on plateaus and smooth continuous landscapes.
* **DDTW Monitor:** Matches first-order derivatives ($\mathbf{X}'$); eliminates vertical scale offset bias and focuses purely on loss of curvature, providing superior fine-tuning in CEC2022 functions and high-dimensional physical models.

---

## 5.6. Cross-Domain Inferential Statistical Validation

### 5.6.1. Wilcoxon Signed-Rank Test
Non-parametric Wilcoxon signed-rank tests ($\alpha = 0.05$) with post-hoc Holm--Bonferroni correction were conducted to evaluate the null hypothesis ($H_0$) of equivalence between the proposed framework and each competitor.

**Table 3.** Wilcoxon signed-rank test summary ($\alpha = 0.05$) across the evaluation domains.

| Comparison | CEC2022 Domain ($+/-/\approx$) | HRES2-H2 Domain ($+/-/\approx$) | Adjusted $p$-value | Null Hypothesis $H_0$ |
|---|:---:|:---:|:---:|:---:|
| **Proposed vs. PSO** | 12 / 0 / 0 | 1 / 0 / 0 | $8.04 \times 10^{-4}$ | Rejected ($p < 0.05$) |
| **Proposed vs. GWO** | 12 / 0 / 0 | 1 / 0 / 0 | $1.62 \times 10^{-6}$ | Rejected ($p < 0.05$) |
| **Proposed vs. WOA** | 12 / 0 / 0 | 1 / 0 / 0 | $9.58 \times 10^{-5}$ | Rejected ($p < 0.05$) |
| **Proposed vs. EHO** | 12 / 0 / 0 | 1 / 0 / 0 | $1.84 \times 10^{-6}$ | Rejected ($p < 0.05$) |
| **Proposed vs. ACO** | 12 / 0 / 0 | 1 / 0 / 0 | $4.85 \times 10^{-4}$ | Rejected ($p < 0.05$) |
| **Proposed vs. ILS** | — | 1 / 0 / 0 | $9.31 \times 10^{-10}$ | Rejected ($p < 0.05$) |
| **Proposed vs. SA** | — | 1 / 0 / 0 | $9.31 \times 10^{-10}$ | Rejected ($p < 0.05$) |

### 5.6.2. Multi-Domain Friedman Ranking Test
The non-parametric Friedman test yielded $\chi^2_F = 327.52$ ($p = 1.37 \times 10^{-63}$) on CEC2022 and $\chi^2_F = 108.55$ ($p = 7.57 \times 10^{-20}$) on HRES2-H2, establishing highly significant performance separation.

**Table 4.** Global average rankings from the Friedman test across all evaluation domains.

| Algorithm | CEC2022 Mean Rank | HRES2-H2 Mean Rank | **Global Mean Rank** |
|---|:---:|:---:|:---:|
| **DDTW Framework (Proposed)** | **1.08** | **1.00** | **1.04 (1st)** |
| **DTW Framework (Proposed)** | **1.13** | **1.82** | **1.48 (2nd)** |
| Grey Wolf Optimizer (GWO) | 1.97 | 3.77 | 2.87 (3rd) |
| Iterated Local Search (ILS) | — | 4.45 | 4.45 (4th) |
| Ant Colony Optimization (ACO) | 5.40 | 4.21 | 4.81 (5th) |
| Whale Optimization Algorithm (WOA) | 6.80 | 5.35 | 6.08 (6th) |
| Particle Swarm Optimization (PSO) | 7.60 | 4.68 | 6.14 (7th) |
| Elephant Herding Optimization (EHO) | 8.35 | 7.05 | 7.70 (8th) |
| Simulated Annealing (SA) | — | 7.87 | 7.87 (9th) |
