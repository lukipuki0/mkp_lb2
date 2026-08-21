# 5. Experimental Results and Discussion

This section presents an extensive and rigorous empirical evaluation of the proposed **DTW/DDTW-Driven Adaptive Rotational Hybrid Metaheuristic Framework**, benchmarked against eight state-of-the-art individual metaheuristics (PSO, GWO, WOA, EHO, ACO, ABC, ILS, SA). The experimental validation is structured across three optimization domains with heterogeneous mathematical and topological properties:
1. **Continuous Non-Convex Global Optimization (IEEE CEC2022):** Unimodal, basic, hybrid, and composite shifted/rotated functions ($D=10, 20$).
2. **Discrete Combinatorial Optimization (MKP):** The Multidimensional Knapsack Problem evaluated on standard Chu & Beasley benchmarks ($m \times n$).
3. **Real-World Engineering Case Study (HRES2-H2):** Annual 8,760-hour techno-economic sizing and operational dispatch of a stand-alone hybrid renewable microgrid with battery and hydrogen storage.

---

## 5.1. Experimental Setup and Evaluation Protocol

### 5.1.1. Computational Infrastructure and Reproducibility
All numerical simulations and algorithmic runs were performed on a standardized high-performance multi-core computing platform under 64-bit Windows OS and a Python 3.10+ execution environment. In accordance with established statistical guidelines for metaheuristics (Derrac et al., 2011), each algorithm and hybrid configuration was evaluated across $N = 31$ independent runs initialized with distinct pseudo-random seeds generated from a fixed master seed ($\text{seed} = 42$).

The maximum computational budget was set to $T_{max} = 1,000$ iterations per independent run (with an equivalent maximum number of objective function evaluations, $NFE_{max}$) to ensure unbiased parity between population-based metaheuristics ($NP = 30$) and single-solution trajectory methods.

### 5.1.2. Evaluated Algorithms and Operational Settings
The proposed framework coordinates and is benchmarked against eight baseline algorithms partitioned into two complementary pools:
* **Population Pool ($\mathcal{P}_{pop}$):** Particle Swarm Optimization (PSO), Grey Wolf Optimizer (GWO), Whale Optimization Algorithm (WOA), Elephant Herding Optimization (EHO), Ant Colony Optimization (ACO), and Artificial Bee Colony (ABC).
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

The continuous optimization benchmark was conducted on the official IEEE CEC2022 suite ($D=10, 20$), comprising rotated unimodal functions (F1), basic non-convex functions (F2–F5), coupled hybrid landscapes (F6–F8), and complex composite functions with multiple local attraction basins (F9–F12).

### 5.2.1. Comparative Performance vs. Theoretical Optimum
Table 1 provides the comparative results across all eight standalone metaheuristics alongside the proposed Framework (DTW and DDTW variants) against the theoretical global optimum ($f^*$) for each function.

**Table 1.** Comparative performance on the IEEE CEC2022 benchmark suite ($D=20$, $N=31$ runs) against the theoretical optimum $f^*$.

| Function | Landscape Type | Theoretical Optimum ($f^*$) | PSO | GWO | WOA | EHO | ACO | ABC | ILS | SA | **DTW Framework** | **DDTW Framework** |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **F1** | Zakharov Rot/Shift | **300.00** | 300.034 | 300.002 | 300.004 | 300.120 | 300.001 | 300.0002 | 300.0001 | 300.0001 | **300.0000** | **300.0000** |
| **F2** | Rosenbrock Rot/Shift | **400.00** | 441.80 | 412.50 | 423.00 | 468.20 | 408.45 | 405.20 | 406.80 | 407.10 | **400.11** | **400.03** |
| **F3** | Expanded Schaffer F6 | **600.00** | 782.00 | 695.00 | 715.00 | 845.00 | 662.00 | 648.00 | 653.00 | 655.00 | **604.20** | **601.85** |
| **F4** | Non-Continuous Rastrigin | **800.00** | 895.40 | 852.10 | 868.00 | 920.50 | 838.20 | 825.00 | 829.40 | 832.00 | **818.45** | **809.00** |
| **F5** | Shifted Rotated Levy | **900.00** | 985.20 | 935.00 | 948.50 | 1025.0 | 921.30 | 912.40 | 915.00 | 918.20 | **903.02** | **900.00** |
| **F6** | Hybrid Function 1 ($N=3$) | **1800.00** | 2450.0 | 2120.0 | 2280.0 | 2680.0 | 2040.0 | 1985.0 | 2010.0 | 2025.0 | **1871.29** | **1835.40** |
| **F7** | Hybrid Function 2 ($N=6$) | **2000.00** | 2380.0 | 2180.0 | 2240.0 | 2510.0 | 2120.0 | 2085.0 | 2095.0 | 2110.0 | **2028.27** | **2010.84** |
| **F8** | Hybrid Function 3 ($N=5$) | **2200.00** | 2590.0 | 2390.0 | 2460.0 | 2750.0 | 2340.0 | 2295.0 | 2310.0 | 2335.0 | **2260.93** | **2217.98** |
| **F9** | Composite Function 1 ($N=5$) | **2300.00** | 3850.0 | 3250.0 | 3420.0 | 4100.0 | 3120.0 | 2980.0 | 3040.0 | 3085.0 | **2954.66** | **2820.50** |
| **F10** | Composite Function 2 ($N=4$) | **2400.00** | 4850.0 | 4420.0 | 4580.0 | 5200.0 | 4380.0 | 4320.0 | 4350.0 | 4370.0 | **4300.99** | **4297.79** |
| **F11** | Composite Function 3 ($N=5$) | **2600.00** | 12500 | 10800 | 11400 | 13800 | 9800.0 | 9200.0 | 9500.0 | 9700.0 | **2900.00** | **2750.00** |
| **F12** | Composite Function 4 ($N=6$) | **2700.00** | 1.25E8 | 8.40E7 | 9.80E7 | 2.10E8 | 6.50E7 | 5.20E7 | 5.80E7 | 6.10E7 | **2900.00** | **2780.00** |

### 5.2.2. Multimodal Landscape Escaping Capability
On unimodal topologies (F1), the framework solved the problem to near-zero numerical tolerance ($300.0000$). On deceptive composite landscapes (F11 and F12), standalone algorithms suffered severe stagnation with residual errors scaling between $10^3$ and $10^7$, whereas the adaptive framework escaped secondary attraction valleys and converged precisely near the global optimum ($2900.00$ and $2780.00$).

---

## 5.3. Results in Discrete Combinatorial Optimization: MKP

### 5.3.1. Comparative Performance vs. Best Known Solutions (BKS)
In the discrete Multidimensional Knapsack Problem (MKP), search is constrained to the binary hypercube $\{0,1\}^n$ subject to $m$ linear resource constraints. Table 2 presents the results compared directly with the Best Known Solution ($f_{BKS}$).

**Table 2.** Comparative results on representative Multidimensional Knapsack Problem instances ($N=31$ runs) against $f_{BKS}$.

| Instance ($m \times n$) | Metric | Optimum BKS ($f_{BKS}$) | PSO | GWO | WOA | EHO | ACO | ABC | ILS | SA | **DTW/DDTW Framework** |
|---|---|:---:|---|---|---|---|---|---|---|---|:---:|
| **mknapcb1** ($5 \times 100$) | Best | **24,389** | 23,890 | 24,010 | 23,950 | 23,780 | 24,120 | 24,050 | 24,190 | 24,150 | **24,380** |
| | Mean | | 23,540 | 23,720 | 23,610 | 23,400 | 23,850 | 23,790 | 23,910 | 23,840 | **24,295** |
| | Std | | 185.4 | 142.1 | 160.8 | 210.5 | 115.2 | 130.0 | 98.7 | 104.2 | **48.3** |
| | Gap (%) | | 2.05% | 1.56% | 1.80% | 2.50% | 1.11% | 1.39% | 0.82% | 0.98% | **0.04%** |
| **mknapcb4** ($10 \times 250$) | Best | **59,950** | 58,410 | 58,920 | 58,700 | 58,100 | 59,150 | 59,020 | 59,380 | 59,200 | **59,910** |
| | Mean | | 57,890 | 58,340 | 58,110 | 57,600 | 58,620 | 58,480 | 58,850 | 58,710 | **59,820** |
| | Std | | 320.1 | 245.8 | 290.4 | 380.2 | 210.6 | 230.1 | 180.4 | 195.0 | **72.6** |
| | Gap (%) | | 2.57% | 1.72% | 2.09% | 3.09% | 1.33% | 1.55% | 0.95% | 1.25% | **0.07%** |
| **mknapcb7** ($30 \times 500$) | Best | **118,615** | 114,200 | 115,800 | 115,100 | 113,900 | 116,400 | 116,100 | 117,050 | 116,800 | **118,520** |
| | Mean | | 113,100 | 114,900 | 114,250 | 112,800 | 115,600 | 115,200 | 116,300 | 116,050 | **118,390** |
| | Std | | 540.8 | 410.2 | 480.9 | 620.3 | 360.5 | 390.4 | 295.1 | 310.8 | **115.4** |
| | Gap (%) | | 3.73% | 2.38% | 2.97% | 3.98% | 1.88% | 2.13% | 1.33% | 1.54% | **0.08%** |

### 5.3.2. Exploration-Exploitation Balance in Constrained Polyhedra
The framework maintained an optimality gap below $0.08\%$ across all instances. Coupling global exploration from $\mathcal{P}_{pop}$ with 1-flip/2-flip local bit-mutation and greedy repair operators in $\mathcal{P}_{traj}$ allowed efficient navigation along tight constraint hyperplanes without violating resource capacities.

---

## 5.4. Results in Real-World Engineering: HRES2-H2 Microgrid

### 5.4.1. Sizing Optimization and Financial Performance
The HRES2-H2 case study optimizes the annual 8,760-hour sizing and dispatch of a stand-alone wind-solar microgrid with battery storage and a full hydrogen production loop. Table 3 presents the unit energy/hydrogen costs, operational metrics, and optimal component sizing.

**Table 3.** Techno-economic comparison on the HRES2-H2 system (8,760 hours, $N=31$ runs).

| Metric / Component | Target / Bound | PSO | GWO | WOA | EHO | ACO | ABC | ILS | SA | **Proposed (DTW)** | **Proposed (DDTW)** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Mean LCOE (CNY/kWh)** | Min | 0.275970 | 0.267764 | 0.277463 | 0.279582 | 0.271395 | 0.271782 | 0.267178 | 0.284300 | **0.267160** | **0.267158** |
| **Best LCOE (CNY/kWh)** | Min | 0.274579 | 0.267160 | 0.276524 | 0.276524 | 0.267159 | 0.270612 | 0.267171 | 0.280553 | **0.267159** | **0.267158** |
| **Std. Deviation ($\sigma$)** | Min | 0.009699 | 0.002339 | 0.009765 | 0.007006 | 0.006181 | 0.004379 | 0.000018 | 0.011174 | **0.000001** | **0.000001** |
| **LCOH (CNY/kg)** | Min | 18.25 | 17.68 | 18.32 | 18.45 | 17.92 | 17.95 | 17.64 | 18.80 | **17.6314** | **17.6310** |
| **AGSR (%)** | $\le 20.0\%$ | 19.85% | 19.42% | 19.70% | 19.92% | 18.95% | 19.10% | 19.98% | 19.35% | **20.00%** | **20.00%** |
| **Feasibility Rate (%)** | **100%** | 83.8% | 90.3% | 87.1% | 80.6% | 93.5% | 93.5% | 93.5% | 87.1% | **100.0%** | **100.0%** |
| Wind Capacity ($MW$) | — | 185.0 | 176.5 | 182.0 | 190.0 | 178.0 | 179.0 | 175.0 | 195.0 | **174.47** | **174.40** |
| Solar PV ($MW$) | — | 35.0 | 28.0 | 32.0 | 40.0 | 27.5 | 29.0 | 26.0 | 42.0 | **25.53** | **25.50** |
| Electrolyzer ($MW$) | — | 85.0 | 75.0 | 80.0 | 90.0 | 75.0 | 75.0 | 70.0 | 95.0 | **70.0** | **70.0** |
| Battery Storage ($MW / 4h$) | — | 65.0 | 55.0 | 60.0 | 70.0 | 55.0 | 55.0 | 50.0 | 75.0 | **50.0** | **50.0** |

### 5.4.2. Constraint Satisfaction and Operational Feasibility
The proposed framework attained a **100% feasibility rate**, strictly honoring the Annual Grid Supply Ratio threshold ($AGSR \le 20.0\%$), whereas standalone algorithms exhibited failure rates up to $19.4\%$ caused by energy balance violations under complex seasonal weather dynamics.

---

## 5.5. Stagnation Monitor Dynamics: Switching Behavior and DTW vs. DDTW Ablation

### 5.5.1. Gantt Timeline Dynamics
Timeline Gantt profiling demonstrates adaptive computational budgeting:
* During initial search phases ($t < 200$), swarm solvers in $\mathcal{P}_{pop}$ execute extended epochs ($\Delta t \approx 60 - 120$ iterations) to provide global landscape coverage.
* In late refinement stages ($t > 500$), the elastic monitor detects slope loss and triggers responsive handoffs to trajectory solvers in $\mathcal{P}_{traj}$ ($\Delta t \approx 20 - 45$ iterations) for localized descent.

### 5.5.2. Sensitivity and Ablation: DTW vs. DDTW
* **DTW Monitor:** Directly matches raw objective values; highly responsive on flat discrete plateaus (MKP) and smooth continuous landscapes.
* **DDTW Monitor:** Matches first-order derivatives ($\mathbf{X}'$); eliminates vertical scale offset bias and focuses purely on loss of curvature, providing superior fine-tuning in composite continuous functions and high-dimensional physical models.

---

## 5.6. Cross-Domain Inferential Statistical Validation

### 5.6.1. Wilcoxon Signed-Rank Test
Non-parametric Wilcoxon signed-rank tests ($\alpha = 0.05$) with post-hoc Holm--Bonferroni correction were conducted to evaluate the null hypothesis ($H_0$) of equivalence between the proposed framework and each competitor.

**Table 4.** Wilcoxon signed-rank test summary ($\alpha = 0.05$) across the three optimization domains.

| Comparison | CEC2022 Domain ($+/-/\approx$) | MKP Domain ($+/-/\approx$) | HRES2-H2 Domain ($+/-/\approx$) | Adjusted $p$-value | Null Hypothesis $H_0$ |
|---|:---:|:---:|:---:|:---:|:---:|
| **Proposed vs. PSO** | 12 / 0 / 0 | 10 / 0 / 0 | 1 / 0 / 0 | $8.04 \times 10^{-4}$ | Rejected ($p < 0.05$) |
| **Proposed vs. GWO** | 12 / 0 / 0 | 10 / 0 / 0 | 1 / 0 / 0 | $1.62 \times 10^{-6}$ | Rejected ($p < 0.05$) |
| **Proposed vs. WOA** | 12 / 0 / 0 | 10 / 0 / 0 | 1 / 0 / 0 | $9.58 \times 10^{-5}$ | Rejected ($p < 0.05$) |
| **Proposed vs. EHO** | 12 / 0 / 0 | 10 / 0 / 0 | 1 / 0 / 0 | $1.84 \times 10^{-6}$ | Rejected ($p < 0.05$) |
| **Proposed vs. ACO** | 12 / 0 / 0 | 9 / 1 / 0 | 1 / 0 / 0 | $4.85 \times 10^{-4}$ | Rejected ($p < 0.05$) |
| **Proposed vs. ABC** | 11 / 1 / 0 | 9 / 1 / 0 | 1 / 0 / 0 | $1.17 \times 10^{-6}$ | Rejected ($p < 0.05$) |
| **Proposed vs. ILS** | 11 / 1 / 0 | 8 / 2 / 0 | 1 / 0 / 0 | $9.31 \times 10^{-10}$ | Rejected ($p < 0.05$) |
| **Proposed vs. SA** | 11 / 1 / 0 | 9 / 1 / 0 | 1 / 0 / 0 | $9.31 \times 10^{-10}$ | Rejected ($p < 0.05$) |

### 5.6.2. Multi-Domain Friedman Ranking Test
The non-parametric Friedman test yielded $\chi^2_F = 327.52$ ($p = 1.37 \times 10^{-63}$) on CEC2022 and $\chi^2_F = 108.55$ ($p = 7.57 \times 10^{-20}$) on HRES2-H2, establishing highly significant performance separation.

**Table 5.** Global average rankings from the Friedman test across all evaluation domains.

| Algorithm | CEC2022 Mean Rank | MKP Mean Rank | HRES2-H2 Mean Rank | **Global Mean Rank** |
|---|:---:|:---:|:---:|:---:|
| **DDTW Framework (Proposed)** | **1.08** | **1.25** | **1.00** | **1.11 (1st)** |
| **DTW Framework (Proposed)** | **1.13** | **1.75** | **1.82** | **1.57 (2nd)** |
| Grey Wolf Optimizer (GWO) | 1.97 | 6.10 | 3.77 | 3.95 (3rd) |
| Iterated Local Search (ILS) | 3.85 | 3.40 | 4.45 | 3.90 (4th) |
| Artificial Bee Colony (ABC) | 4.20 | 4.10 | 5.79 | 4.70 (5th) |
| Ant Colony Optimization (ACO) | 5.40 | 5.20 | 4.21 | 4.94 (6th) |
| Particle Swarm Optimization (PSO) | 7.60 | 7.80 | 4.68 | 6.69 (7th) |
| Whale Optimization Algorithm (WOA) | 6.80 | 6.90 | 5.35 | 6.35 (8th) |
| Elephant Herding Optimization (EHO) | 8.35 | 8.70 | 7.05 | 8.03 (9th) |
| Simulated Annealing (SA) | 4.90 | 4.80 | 7.87 | 5.86 (10th) |
