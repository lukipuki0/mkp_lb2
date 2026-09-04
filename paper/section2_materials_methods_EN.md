---
# 2. Materials and Methods

## 2.1. Overall Architecture of the DTW-Driven Adaptive Rotational Hybrid Framework

To comprehensively address the identified deficiencies of premature stagnation, population diversity loss, and lack of multi-domain generalization present in conventional metaheuristics, this work proposes an **Adaptive Rotational Hybrid Framework** governed by a **Dynamic Time Warping-based Stagnation Monitor (DTW/DDTW)**. Unlike existing hybrid frameworks that switch algorithms according to static rules—such as fixed patience thresholds or predefined switching periods—the proposed framework employs the DTW distance computed over the recent convergence curve as a mathematically grounded, dynamic criterion for detecting progress loss with high precision and minimal false positives.

The central philosophy of the architecture exploits **algorithmic complementarity**. According to the No Free Lunch (NFL) theorem [Wolpert & Macready, 1997], no individual metaheuristic possesses a search strategy capable of consistently outperforming all competitors across the entire universe of optimization problems. Consequently, the framework organizes the metaheuristics into two disjoint, specialized operational pools: a **Population Pool** ($\mathcal{P}_{pop}$) grouping algorithms oriented toward global exploration via large-scale collective dynamics, and a **Trajectory Pool** ($\mathcal{P}_{traj}$) grouping algorithms oriented toward intensive local exploitation via guided perturbations.

The orchestrator's control flow is structured into **Epochs** ($e = 1, 2, \dots$). In each epoch, a single solver $M^{(e)}$ selected randomly from the corresponding pool executes search iterations on the target problem. Unlike traditional switching methods that determine epoch termination by a fixed iteration counter, in the proposed framework each epoch has a completely **dynamic and adaptive duration**: an epoch terminates only when the DTW/DDTW Monitor detects that the recent fitness profile has degenerated into a statistically significant stagnation plateau. At that instant, the orchestrator executes the **Switching and Memory Transfer Protocol**, extracting the best global solution found thus far ($x_{best}$), switching to the alternating pool, and injecting $x_{best}$ as the initialization seed of the new solver to prevent the loss of accumulated progress.

**Table 1.** Summary of operational parameters of the DTW hybrid framework.

| Parameter | Symbol | Value | Description |
|---|---|---|---|
| Window size | $W$ | 30 | Recent fitness history analyzed by the monitor |
| Sakoe-Chiba band | $w$ | $\lfloor 0.1 \cdot W \rfloor = 3$ | Temporal warping constraint in DTW |
| Patience | $P$ | 3 | Consecutive confirmations required to trigger alarm |
| Plateau tolerance | $K_{max}$ | 15 | Consecutive iterations without improvement |
| Lower percentile | $P_{low}$ | 30.0 | Percentile for adaptive threshold on constant distance |
| Upper percentile | $P_{high}$ | 70.0 | Percentile for adaptive thresholds on ramp distance and delta |
| Max. iterations | $T_{max}$ | 1,000 | Global stopping criterion |
| Paired runs | $N$ | 31 | Corresponding stochastic executions analysed as paired |

---

## 2.2. Mathematical Formulation of the DTW/DDTW Stagnation Detection Engine

The stagnation monitor constitutes the algorithmic core of the framework. At each iteration $t$, the monitor receives the best fitness value recorded up to that moment by the active solver and maintains a sliding history $\mathbf{X} = [x_{t-W+1}, \dots, x_t] \in \mathbb{R}^W$ of the last $W$ values. The analysis activates only once at least $W$ observations have been accumulated ($t \geq W$).

### 2.2.1. Baseline Reference Series

To quantitatively measure the degree of progress present in the window $\mathbf{X}$, the monitor dynamically constructs two reference series of length $W$ anchored at the initial value of the window $x_1 = x_{t-W+1}$:

1. **Ideal Progress Ramp** $\mathbf{R} = [r_1, r_2, \dots, r_W]$: represents the expected convergence trajectory under sustained improvement with minimum slope $s_{min}$:
$$r_k = x_1 + s_{min} \cdot (k-1), \quad k = 1, \dots, W$$
The minimum slope is calculated adaptively as 1% of the window's variation range:
$$s_{min} = 0.01 \cdot \frac{|x_W - x_1|}{W}$$

2. **Constant Stagnation Plateau** $\mathbf{C} = [c_1, c_2, \dots, c_W]$: represents the complete absence of improvement, i.e., a fully immobilized fitness value:
$$c_k = x_1, \quad k = 1, \dots, W$$

### 2.2.2. DTW Distance with Sakoe-Chiba Band

The distance between the observed trajectory $\mathbf{X}$ and any reference series $\mathbf{Y} \in \{\mathbf{R}, \mathbf{C}\}$ is computed via the **Dynamic Time Warping (DTW)** algorithm, which finds the optimal alignment between elements of the two series through an accumulated cost matrix $\mathbf{D} \in \mathbb{R}^{(W+1) \times (W+1)}$. Each cell $(i, j)$ is updated by the recurrence relation:

$$D(i,j) = |x_i - y_j| + \min\bigl\{D(i-1,j),\ D(i,j-1),\ D(i-1,j-1)\bigr\}$$

with boundary conditions $D(0,0) = 0$ and $D(i,0) = D(0,j) = +\infty$ for $i,j > 0$. To limit admissible temporal warping and reduce complexity from $\mathcal{O}(W^2)$ to $\mathcal{O}(W \cdot w)$, the **Sakoe-Chiba band** of width $w$ is applied, restricting the search to cells satisfying $|i-j| \leq w$, where $w = \max(1, \lfloor 0.1 \cdot W \rfloor)$. The final DTW distance corresponds to the accumulated value at the opposite corner: $\text{DTW}(\mathbf{X}, \mathbf{Y}) = D(W, W)$.

### 2.2.3. DDTW Derivative-Based Variant

To render the monitor **magnitude-shift invariant** and sensitive exclusively to the shape and slope of the convergence curve, the **Derivative Dynamic Time Warping (DDTW)** variant is implemented. Under DDTW, both the observed and reference series are transformed via the first finite difference prior to computing the DTW distance:

$$\nabla x_k = x_k - x_{k-1}, \quad \text{with } \nabla x_1 = 0$$
$$\text{DDTW}(\mathbf{X}, \mathbf{Y}) = \text{DTW}(\nabla \mathbf{X},\ \nabla \mathbf{Y})$$

This transformation converts the morphological comparison of absolute levels into a comparison of **rates of change**, enabling plateau detection even when fitness magnitudes vary considerably across different instances or test functions. The `use_ddtw` parameter in `StagnationConfig` selects the monitor's operating mode.

### 2.2.4. Stagnation Diagnostic Metrics

At each iteration $t \geq W$, the monitor computes two DTW/DDTW diagnostic distances:

$$D_1 = \text{DTW/DDTW}(\mathbf{X}, \mathbf{R}), \qquad D_2 = \text{DTW/DDTW}(\mathbf{X}, \mathbf{C})$$

and the relative deviation metric:

$$\Delta = D_1 - D_2$$

The geometric interpretation of these metrics is straightforward: if trajectory $\mathbf{X}$ is progressing, $D_1$ will be small (similar to the ramp) and $D_2$ large (dissimilar to the constant), resulting in $\Delta < 0$. If the trajectory is stagnating, $D_2 \approx 0$, $D_1$ increases, and $\Delta > 0$ with a growing positive value.

---

## 2.3. Adaptive Thresholding via Moving Percentiles

The use of fixed thresholds to classify DTW distances is fundamentally inadequate in multi-domain optimization, as fitness scales vary by several orders of magnitude between discrete problems (MKP, scale of tens of thousands), continuous problems (CEC2022, scale of hundreds), and industrial applications (HRES2-H2, sub-unit scale in CNY/kWh). To ensure the monitor's portability without manual hyperparameter re-tuning, the framework implements a **dynamic thresholding mechanism based on moving historical percentiles**.

Three cumulative circular buffers are maintained: $\mathcal{H}_{D_1}$, $\mathcal{H}_{D_2}$ and $\mathcal{H}_{\Delta}$, recording the history of $D_1$, $D_2$ and $\Delta$ values from all previous iterations since the beginning of the execution. Once each buffer contains at least 10 records, the classification thresholds are recomputed at each iteration as:

$$\theta_c = \text{Perc}(\mathcal{H}_{D_2},\ P_{low}), \qquad \theta_r = \text{Perc}(\mathcal{H}_{D_1},\ P_{high}), \qquad \theta_\Delta = \text{Perc}(\mathcal{H}_\Delta,\ P_{high})$$

where $P_{low} = 30.0$ and $P_{high} = 70.0$ are the default percentiles. If the history is insufficient (fewer than 10 records), static startup thresholds are employed: $\theta_c = 0.1W$, $\theta_r = 0.5W$, $\theta_\Delta = 0.3W$.

The **triple stagnation condition** $\text{Stagnant}(t)$ is satisfied when three independent criteria converge simultaneously:

$$\text{Stagnant}(t) = \underbrace{\left(N_{no\_improve} \geq K_{max}\right)}_{\text{patience plateau}} \land \underbrace{\left(D_2 \leq \theta_c\right)}_{\text{constant morphology}} \land \underbrace{\left(D_1 \geq \theta_r \lor \Delta \geq \theta_\Delta\right)}_{\text{progress deviation}}$$

where $N_{no\_improve}$ is the counter of consecutive iterations without improvement in the best fitness value of the active solver. To filter spurious triggers caused by short-term stochastic fluctuations, a **confirmation streak** mechanism with patience parameter $P$ is applied. The streak counter $S_t$ is updated as:

$$S_t = \begin{cases} S_{t-1} + 1 & \text{if } \text{Stagnant}(t) \\ 0 & \text{otherwise} \end{cases}$$

The **switching alarm** is triggered only when $S_t \geq P$, requiring stagnation to persist for at least $P = 3$ consecutive iterations before forcing rotation.

---

## 2.4. Metaheuristic Pools and Seed Memory Injection Protocol

### 2.4.1. Algorithmic Pool Composition

The framework integrates nine well-established metaheuristics from the evolutionary computation literature, selected to maximize diversity of search mechanisms and minimize behavioral correlation between their operators. They are organized into two disjoint pools:

**Population Pool** $\mathcal{P}_{pop}$ (global exploration — 5 algorithms):

- **Particle Swarm Optimization (PSO)** [Kennedy & Eberhart, 1995]: models the social behavior of flocks. Each particle $i$ updates its velocity in dimension $d$ as $v_{i,d}^{t+1} = w \cdot v_{i,d}^t + c_1 r_1 (p_{i,d} - x_{i,d}^t) + c_2 r_2 (g_d - x_{i,d}^t)$, where $w$ is inertia, $c_1$ and $c_2$ are cognitive and social acceleration coefficients, and $p_{i,d}$, $g_d$ are the personal and global best positions. Linear decreasing inertia $w_{max} \to w_{min}$ is used during each epoch.
- **Grey Wolf Optimizer (GWO)** [Mirjalili et al., 2014]: simulates the hunting hierarchy of grey wolves ($\alpha, \beta, \delta$). Position updates are weighted by the direction vectors of the three leaders with variable coefficients $A$ and $C$.
- **Whale Optimization Algorithm (WOA)** [Mirjalili & Lewis, 2016]: alternates between the random search strategy and the spiral bubble-net attack maneuver according to a probability parameter $p$ per iteration.
- **Elephant Herding Optimization (EHO)** [Wang et al., 2015]: implements two complementary operators—clan updating that attracts individuals toward each subgroup's matriarch, and dominant male separation toward the periphery of the search space.
- **Ant Colony Optimization (ACO)** [Dorigo & Stützle, 2004]: in its continuous-space variant, updates a pheromone-weighted archive of solutions that serves as a Gaussian sampling distribution for neighborhood exploration.

**Trajectory Pool** $\mathcal{P}_{traj}$ (local exploitation — 4 algorithms):

- **Iterated Local Search (ILS)** [Lourenço et al., 2003]: applies systematic perturbations to the best known solution followed by a stochastic local descent phase, iterating to escape local optima through controlled diversification.
- **Variable Neighborhood Search (VNS)** [Mladenović & Hansen, 1997]: dynamically explores a hierarchical set of $k_{max}$ neighborhood structures $\mathcal{N}_k$, alternating between systematic shaking perturbations and descent-based local refinement, expanding the exploration radius during plateaus and returning to the primary fine neighborhood upon success.
- **Tabu Search (TS)** [Glover, 1986]: implements adaptive short-term memory through spatial exclusion regions (tabu balls of radius $r_{tabu}$) to prevent cycling over previously visited local optima, complemented by an aspiration criterion that overrides the prohibition whenever a candidate surpasses the global best.
- **Simulated Annealing (SA)** [Kirkpatrick et al., 1983]: implements probabilistic acceptance of lower-quality solutions via the Metropolis criterion $P(accept) = \exp\bigl(-\Delta f / T_k\bigr)$, with temperature decreasing according to the geometric cooling schedule $T_k = T_0 \cdot \alpha^k$.

### 2.4.2. Seed Memory Injection Protocol ($x_{best}^*$)

The inter-epoch knowledge transfer protocol is the mechanism that distinguishes this framework from a simple portfolio of independently executed algorithms. When the DTW/DDTW monitor triggers the switching alarm at the end of epoch $e$, the following steps are executed:

1. **Knowledge Extraction:** The best valid global solution found since the beginning of the execution is retrieved, $x_{best}^* \in \mathbb{R}^D$, along with its associated fitness value $f^* = f(x_{best}^*)$.

2. **Injection into Population-Based Algorithms:** For the new solver $M^{(e+1)} \in \mathcal{P}_{pop}$ with population size $N_{pop}$, initialization is performed via a **mixed strategy** that simultaneously preserves elitist guidance and exploratory diversity:
   - The first individual receives the exact seed: $\mathbf{x}_1^{(0)} = x_{best}^*$.
   - $\lfloor N_{pop}/2 \rfloor$ of the remaining individuals are generated via Gaussian perturbation centered at $x_{best}^*$: $\mathbf{x}_i^{(0)} = x_{best}^* + \mathcal{N}(\mathbf{0}, \sigma^2\mathbf{I}) \cdot (U - L)$, with $\sigma = 0.05$.
   - The remainder is initialized via uniform sampling over the domain $[L, U]^D$ to ensure exploratory coverage.

3. **Injection into Trajectory-Based Algorithms:** For the new solver $M^{(e+1)} \in \mathcal{P}_{traj}$, the starting point is set deterministically as $x_0 = x_{best}^*$, allowing the local exploitation algorithm to refine the best search region identified so far.

---

## 2.5. Optimization Problem 1: Multidimensional Knapsack Problem (MKP)

The **Multidimensional Knapsack Problem (MKP)** is an NP-hard combinatorial optimization problem that generalizes the classical 0-1 knapsack to $m$ simultaneous capacity constraints. Its mathematical formulation is:

$$\max \quad f(\mathbf{x}) = \sum_{j=1}^{n} p_j x_j$$

$$\text{subject to} \quad \sum_{j=1}^{n} r_{ij} x_j \leq b_i, \quad \forall\, i \in \{1, \dots, m\}$$

$$x_j \in \{0, 1\}, \quad \forall\, j \in \{1, \dots, n\}$$

where $n$ denotes the number of candidate items, $p_j > 0$ the profit of item $j$, $r_{ij} \geq 0$ the consumption of resource $i$ by item $j$, and $b_i > 0$ the maximum capacity of resource $i$. The computational hardness of the MKP stems from the combination of binary decision variables with the need to simultaneously satisfy $m$ interrelated inequality constraints.

Experiments are conducted on the benchmark instances from the **OR-Library MKNAPCB** dataset [Chu & Beasley, 1998], which provides instances with $n \in \{100, 250, 500\}$ items and $m \in \{5, 10, 30\}$ constraints, with known optimal solutions enabling computation of the percentage optimality gap.

#### Binarization Mechanism and Solution Repair

To operate continuous-space metaheuristics on the discrete space $\{0,1\}^n$ of the MKP, a sigmoid transfer function maps PSO velocities or continuous positions to binary selection probabilities. Since stochastic perturbations may generate infeasible solutions, the **pseudo-utility-based heuristic repair operator** is applied:

$$u_j = \frac{p_j}{\displaystyle\sum_{i=1}^{m} \dfrac{r_{ij}}{b_i}}$$

If the solution violates any capacity constraint, included items ($x_j = 1$) are removed in ascending order of $u_j$ until feasibility is restored. After repair, a **greedy addition** phase inserts excluded items ($x_j = 0$) in descending order of $u_j$ while no constraints are violated, maximizing the use of residual capacity.

---

## 2.6. Optimization Problem 2: Continuous IEEE CEC2022 Benchmark Suite

To evaluate the generalization capability of the framework in the domain of high-dimensional non-convex continuous optimization, the **IEEE CEC2022 Benchmark Suite** [Abhishek et al., 2022] is adopted, comprising 12 minimization functions distributed across four categories of increasing complexity:

$$\min \quad f(\mathbf{x}), \quad \mathbf{x} \in [-100, 100]^D, \quad D \in \{10, 20\}$$

**Table 2.** Test functions of the IEEE CEC2022 Benchmark Suite.

| ID | Function | Category | Optimum $F_i^*$ |
|:---:|:---|:---:|:---:|
| $F_1$ | Shifted Sphere | Basic unimodal | 300 |
| $F_2$ | Shifted Weighted Rosenbrock | Basic multimodal | 400 |
| $F_3$ | Shifted Lunacek Bi-Rastrigin | Basic multimodal | 600 |
| $F_4$ | Expanded Non-Continuous Ackley | Basic multimodal | 800 |
| $F_5$ | Shifted Lévy | Basic multimodal | 900 |
| $F_6$ | Hybrid Function 1 ($N=3$ sub-functions) | Hybrid | 1800 |
| $F_7$ | Hybrid Function 2 ($N=6$ sub-functions) | Hybrid | 2000 |
| $F_8$ | Hybrid Function 3 ($N=5$ sub-functions) | Hybrid | 2200 |
| $F_9$ | Composition Function 1 ($N=5$ components) | Gaussian composition | 2300 |
| $F_{10}$ | Composition Function 2 ($N=4$ components) | Gaussian composition | 2400 |
| $F_{11}$ | Composition Function 3 ($N=5$ components) | Gaussian composition | 2600 |
| $F_{12}$ | Composition Function 4 ($N=6$ components) | Gaussian composition | 2700 |

Unimodal functions ($F_1$) evaluate pure convergence efficiency; basic multimodal functions ($F_2$–$F_5$) introduce multiple local optima and plateaus; hybrid functions ($F_6$–$F_8$) divide the search space into sub-spaces with functions of different nature; and composition functions ($F_9$–$F_{12}$) combine multiple base functions with Gaussian weights, creating highly irregular, multimodal fitness landscapes with gradient deception.

The approximation error to the global optimum is reported as $\epsilon_i = f(\mathbf{x}_{best}) - F_i^*$ for each function $i$, where $\epsilon_i \approx 0$ indicates convergence to the known optimum.

---

## 2.7. Real-World Engineering Case Study: Hybrid Renewable Energy System with Hydrogen Storage (HRES2-H2)

As a demonstration of the framework's capability to solve high-impact engineering problems of physico-economic complexity, it is applied to the **capacity sizing and dispatch optimization** of a hybrid renewable energy system with hydrogen generation and storage (HRES2-H2). The case study extends the WPEB (Wind-Photovoltaic-Electrolyzer-Battery) model proposed in [Li et al., 2024] for the location of Baotou, Inner Mongolia, China ($41.70°\text{N}$, $110.43°\text{E}$), introducing four methodological extensions that transform the original quasi-continuous search space into a mixed integer-continuous problem.

### 2.7.1. WPEB Model Description and Extensions

The HRES2-H2 system integrates five complementary technologies with a total renewable generation capacity fixed at $P_{total} = 200$ MW (site capacity constraint):

$$P_{WT} + P_{PV} = P_{total} = 200 \text{ MW}$$

The four decision variables of the optimization problem are:

$$\mathbf{x} = \left[x_1,\ x_2,\ x_3,\ x_4\right] = \left[P_{WT},\ N_{el},\ P_{bat},\ \tau_{bat}\right]$$

**Table 3.** Decision variables and their search domains.

| Variable | Symbol | Type | Range / Set |
|:---|:---:|:---:|:---|
| Installed wind capacity | $P_{WT}$ | Continuous | $[0,\ 200]$ MW |
| PV capacity | $P_{PV}$ | Derived | $200 - P_{WT}$ MW |
| Number of electrolyzer modules | $N_{el}$ | Integer | $\{10, 11, \dots, 20\}$ units × 5 MW each |
| Battery bank power | $P_{bat}$ | Discrete | $\{0, 5, 10, \dots, 50\}$ MW |
| Storage duration | $\tau_{bat}$ | Discrete | $\{1.0,\ 2.0,\ 4.0\}$ h |

The model extensions introduce the discretization of $N_{el}$ (10 to 20 modules of 5 MW each, yielding electrolyzer capacities from 50 to 100 MW), independence of battery power from electrolyzer capacity (in the original model $P_{bat}$ was fixed at 30\% of $P_{el}$), and the choice among three storage durations.

### 2.7.2. Physical Models of Renewable Generation

#### Wind Turbine Model

The hourly wind farm power output $P_{WT}(t)$ is computed as a function of wind speed at 50 meters height $v_{50m}(t)$ via the cubic power curve with rated-power zone and cut-in/cut-out thresholds:

$$P_{WT}(t) = \begin{cases}
0, & v(t) < v_{in} \text{ or } v(t) \geq v_{out} \\[4pt]
P_{rated} \cdot \dfrac{v(t)^3 - v_{in}^3}{v_{rated}^3 - v_{in}^3}, & v_{in} \leq v(t) < v_{rated} \\[6pt]
P_{rated}, & v_{rated} \leq v(t) < v_{out}
\end{cases}$$

with $v_{in} = 2.5$ m/s (cut-in speed), $v_{rated} = 10.5$ m/s (rated speed), $v_{out} = 25.0$ m/s (cut-out speed), and $P_{rated} = 5.0$ MW per turbine. Hourly wind speed profiles are obtained from the NASA POWER climate database for the reference year 2008 (selected as the Typical Meteorological Year).

#### Photovoltaic Array Model

The PV array power output $P_{PV}(t)$ is determined from global horizontal irradiance $G(t)$ [kWh/m²] and ambient temperature $T_a(t)$ [°C], incorporating the thermal effect on cell efficiency via the Normal Operating Cell Temperature (NOCT):

$$T_c(t) = T_a(t) + \frac{G(t)}{800} \cdot (NOCT - 20)$$

$$P_{PV}(t) = P_{PV,STC} \cdot \frac{G(t)}{1000} \cdot \bigl[1 + \gamma_T \cdot (T_c(t) - 25)\bigr] \cdot f_{dera}$$

where $NOCT = 47°$C, $\gamma_T = -0.45\%/°$C is the temperature coefficient, $f_{dera} = 0.90$ is the global loss factor (soiling, wiring, partial shading), and $P_{PV,STC}$ is the installed power under Standard Test Conditions (STC). Irradiance and temperature profiles are vectorized for all 8,760 annual hours via pre-computation of normalized unit profiles, reducing the computational cost of each fitness evaluation.

### 2.7.3. Hourly Dispatch Logic (8,760 Hours)

The physical simulation of the system is performed hour by hour for the complete reference year. At each hour $t \in \{1, 2, \dots, 8760\}$, the net available power is computed:

$$P_{net}(t) = P_{WT}(t) + P_{PV}(t) - P_{load}(t)$$

and the following **priority dispatch hierarchy** is applied:

**Surplus Case** ($P_{net}(t) > 0$): Excess energy is allocated following these priorities:
1. Power the PEM electrolyzer if $P_{gen}(t) \geq P_{el,min} = 0.30 \cdot P_{el}$ (30% minimum load condition).
2. Charge the battery bank with remaining surplus up to $SOC_{max} = P_{bat} \cdot \tau_{bat}$ MWh.
3. Inject the remainder into the external electrical grid (recorded as $P_{grid\_sales}(t)$).

**Deficit Case** ($P_{net}(t) < 0$): The energy deficit is covered by:
1. Battery bank discharge with efficiency $\eta_{dis} = \sqrt{\eta_{RT}}$.
2. If battery is insufficient, use fuel cell energy ($H_2$ from compressed storage).
3. Import energy from the external electrical grid (recorded as $P_{grid\_import}(t)$).

The full battery charge-discharge cycle efficiency is $\eta_{RT} = 0.90$ (round-trip efficiency), with $\eta_{ch} = \eta_{dis} = \sqrt{0.90} \approx 0.9487$ for charging and discharging respectively. The electrolyzer converts electricity to hydrogen with efficiency $\eta_{el} = 0.75$, and the Higher Heating Value (HHV) of hydrogen is $HHV_{H_2} = 39.4$ kWh/kg.

Total annual hydrogen production in kilograms is calculated as:

$$m_{H_2} = \frac{E_{el,annual} \times 1000 \times \eta_{el}}{HHV_{H_2}} \quad [\text{kg/year}]$$

### 2.7.4. Financial Objective Functions and AGSR Constraint

The optimization problem objective minimizes the **Levelized Cost of Energy (LCOE)** in CNY/kWh over the project lifetime $N_{proj} = 25$ years:

$$\min_{\mathbf{x}}\quad \text{LCOE} = \frac{NPC \cdot CRF(r, N_{proj})}{E_{served,annual}}$$

where the **Net Present Cost (NPC)** is decomposed as the sum of initial capital expenditure (CAPEX), operation and maintenance (O\&M), and replacement costs of short-lifecycle components, discounted to present value for each technology $k \in \{WT, PV, EL, BAT\}$:

$$NPC = \sum_{k} \left[ CAPEX_k + \sum_{y \in \mathcal{Y}_{rep,k}} \frac{REP_k}{(1+r)^y} + \sum_{y=1}^{N_{proj}} \frac{O\&M_k}{(1+r)^y} \right]$$

The **Capital Recovery Factor (CRF)** annualizes the NPC over the project life with real discount rate $r = 4.35\%$:

$$CRF(r, N) = \frac{r(1+r)^N}{(1+r)^N - 1}$$

**Table 4.** Reference economic parameters (adapted from Li et al., 2024).

| Technology | CAPEX [CNY/kW] | O\&M [CNY/kW·year] | Replacement [CNY/kW] | Lifetime [years] |
|:---|:---:|:---:|:---:|:---:|
| Wind turbines (WT) | 5,917 | 40.2 | — | 25 |
| Photovoltaic (PV) | 4,633 | 17.6 | — | 25 |
| PEM Electrolyzer | 6,964 | 208.9 | 5,969 | 15 |
| Battery BESS | 2,549 | 10.0 | 500 | 10 |

**Annual Grid Supply Ratio (AGSR) Security Constraint:** To ensure minimum energy autonomy of the system and prevent excessive dependence on the external electrical grid, the annual ratio between grid-imported energy and total served demand is strictly bounded by:

$$AGSR = \frac{\displaystyle\sum_{t=1}^{8760} P_{grid\_import}(t)}{\displaystyle\sum_{t=1}^{8760} P_{load}(t)} \leq 0.20$$

Solutions that violate this constraint ($AGSR > 0.20$) are classified as **infeasible** and penalized with an objective value of $f(\mathbf{x}) = 100 + 10 \cdot AGSR$, ensuring the optimizer avoids these regions of the search space.

The **LCOH (Levelized Cost of Hydrogen)** in CNY/kg is calculated complementarily as:

$$LCOH = \frac{NPC \cdot CRF(r, N_{proj})}{m_{H_2,annual}}$$

---

## 2.8. Experimental Evaluation Protocol and Non-Parametric Statistical Inference

Given that the intrinsic stochastic nature of metaheuristics—arising from random population initialization, probabilistic operators, and solution selection—produces result distributions whose Gaussianity cannot be assumed a priori, the experimental evaluation adopts a **non-parametric statistical inference protocol** following the guidelines of [Derrac et al., 2011; García et al., 2010].

For each algorithm evaluated on each instance or test function, **$N = 31$ executions** are performed, with corresponding executions treated as paired to ensure comparable randomization conditions and reliable estimation of performance variability.

### Statistical Testing Protocol

1. **Shapiro-Wilk Normality Test:** The null hypothesis $H_0$ that the distribution of $N = 31$ final values follows a normal distribution is evaluated. If the resulting $p$-value satisfies $p < 0.05$, $H_0$ is rejected and the need for non-parametric tests is confirmed.

2. **Wilcoxon Signed-Rank Test (Pairwise Comparisons):** For each pair of algorithms (proposed framework vs. competitor algorithm $k$), the paired Wilcoxon test is performed with significance level $\alpha = 0.05$. Results are classified as:
   - $p < 0.05$: statistically significant difference (the proposed framework is superior or inferior with 95% confidence).
   - $p \geq 0.05$: no statistically significant difference.

3. **Friedman Global Ranking Test:** To obtain a global statistical dominance ranking among all simultaneously evaluated algorithms across the complete set of instances or functions, the Friedman statistic is computed:
$$\chi_F^2 = \frac{12}{N_p \cdot K(K+1)} \sum_{j=1}^{K} \bar{R}_j^2 - 3 N_p (K+1)$$
where $N_p$ is the number of test instances/functions, $K$ is the number of compared algorithms, and $\bar{R}_j$ is the mean rank of algorithm $j$. A value $p < 0.05$ on the global test confirms that at least one pair of algorithms presents statistically significant differences in performance.

---

## 2.9. Computational Environment

All experiments were executed on a platform running Python 3.11.x, using NumPy for high-speed vectorized operations and, optionally, Numba JIT compilation for the HRES2-H2 hourly dispatch loop (`_fast_dispatch_simulation`). Random-number generation was controlled consistently across corresponding executions to support reproducibility and paired comparisons. The complete source code of the framework, including all metaheuristic modules, the DTW/DDTW monitor, and the benchmark scripts, is publicly available in the project repository.
