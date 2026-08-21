# 6. Discussion and Sensitivity Analysis

This section provides a deeper analytical discussion of the empirical findings, focusing on the underlying mechanisms driving the performance of the proposed **DTW/DDTW-Driven Adaptive Rotational Hybrid Metaheuristic Framework**.

---

## 6.1. Insights into Adaptive Collaboration Dynamics

Traditional collaborative and relay hybrid metaheuristics rely on fixed communication intervals (e.g., executing solver switches every $K$ iterations). Such static approaches suffer from two fundamental flaws:
1. **Premature Handoffs:** Switching solvers while the active metaheuristic is still making significant progress wastes its exploratory trajectory momentum.
2. **Prolonged Stagnation Entrapment:** Maintaining an algorithm long after its search trajectory has stagnated in a local basin wastes computational budget ($NFE$).

The proposed framework mitigates these limitations through elastic trajectory monitoring via Dynamic Time Warping (DTW) and Derivative DTW (DDTW). By computing non-linear time alignments between the recent fitness trajectory window ($Q$) and a monotonic reference vector ($R$), the monitor quantifies dynamic slope indices $D_1$ and $D_2$. The adaptive historical percentile thresholds ($P_{low}$ and $P_{high}$) dynamically scale with the search history, ensuring that solver rotation is triggered only when true numerical stagnation occurs.

As observed in the timeline Gantt profiling across the continuous CEC2022 functions and the HRES2-H2 microgrid system, early search phases ($t < 200$) naturally allow population-based swarm algorithms ($\mathcal{P}_{pop}$) to execute longer epochs ($\Delta t \approx 60 - 120$ iterations) for global landscape coverage. In contrast, during mature exploitation ($t > 500$), the elastic monitor detects slope attenuation rapidly, triggering responsive handoffs to trajectory solvers ($\mathcal{P}_{traj}$) with shorter epochs ($\Delta t \approx 20 - 45$ iterations) to perform localized descent.

---

## 6.2. Role of Elitist Memory Injection

When a stagnation event triggers a rotation from solver $\mathcal{M}_A$ to solver $\mathcal{M}_B$, transferring knowledge effectively is critical. In the proposed framework, the elitist memory operator injects the best-so-far global solution vector $\mathbf{x}_{best}$ into the initial state of $\mathcal{M}_B$:
* **For population-based algorithms ($\mathcal{P}_{pop}$):** $\mathbf{x}_{best}$ replaces the weakest individual in the population, preserving population diversity while anchoring search around the promising basin.
* **For trajectory-based methods ($\mathcal{P}_{traj}$):** $\mathbf{x}_{best}$ serves as the immediate starting solution vector, preventing random restarts and enabling instant local intensification.

Empirical observation confirms that memory injection eliminates the "re-exploration penalty" typically associated with solver handoffs, allowing the incoming metaheuristic to immediately refine the candidate solution.

---

## 6.3. Parameter Sensitivity Analysis

The performance of the DTW/DDTW stagnation monitoring engine depends on three main control parameters: the sliding window size $W$, the Sakoe--Chiba band width $w$, and the historical percentile thresholds ($P_{low}, P_{high}$).

### 6.3.1. Impact of Window Size $W$
Experiments evaluated window sizes $W \in \{15, 30, 50\}$ across unimodal, hybrid, and composite CEC2022 functions:
* **Small windows ($W = 15$):** Exhibit high sensitivity to transient stochastic fluctuations, occasionally triggering false-positive handoffs before a solver has fully exploited a basin.
* **Large windows ($W = 50$):** Introduce detection lag, delaying necessary solver switches by 15--20 iterations and reducing budget efficiency.
* **Default setting ($W = 30$):** Provides an optimal balance between responsiveness and noise filter stability across both continuous and physical engineering landscapes.

### 6.3.2. Impact of Sakoe--Chiba Band Width $w$
The Sakoe--Chiba band constrains the warping path $|i - j| \le w$:
* **Unconstrained warping ($w = \infty$):** Allows excessive temporal distortion, causing stagnant trajectories to align artificially with steep reference curves.
* **Setting $w = 3$:** Restricts temporal shifts to small local delays, accurately preserving local gradient trends while maintaining computational speed.

### 6.3.3. Percentile Thresholds ($P_{low}, P_{high}$)
Configuring $P_{low} = 30.0\%$ and $P_{high} = 70.0\%$ creates a self-adjusting sensitivity envelope. As the optimization progresses and overall fitness improves, the absolute threshold value adapts automatically, eliminating the need for problem-specific manual tuning.

---

## 6.4. Computational Overhead and Threats to Validity

### 6.4.1. Algorithmic Complexity
The computational complexity of calculating DTW/DDTW alignment between a window $Q$ of length $W$ and reference $R$ under a Sakoe--Chiba band $w$ is $\mathcal{O}(W \cdot w)$. Because $W = 30$ and $w = 3$ are small constants independent of problem dimension $D$ or maximum iterations $T_{max}$, the CPU time required for monitoring is under $0.5\%$ of total execution time. The computational bottleneck remains the objective function evaluations (particularly the 8,760-hour simulation loop in the HRES2-H2 microgrid).

### 6.4.2. Threats to Validity
* **Internal Validity:** Mitigated by evaluating $N = 31$ independent Monte Carlo runs per algorithm using fixed pseudo-random seeds derived from master seed 42, ensuring exact reproducibility.
* **External Validity:** Mitigated by testing across diverse mathematical topologies (CEC2022 F1--F12) and complex physical engineering dispatch constraints (HRES2-H2).
