# Especificación Matemática y Arquitectura Algorítmica del Framework DTW

**Proyecto:** Framework Híbrido Metaheurístico Adaptativo por Rotación Basado en DTW  
**Autor:** Equipo de Investigación  

---

## 1. Monitor de Estancamiento Basado en DTW/DDTW

### 1.1. Series de Referencia Dinámicas

Para una ventana deslizante de fitness $\mathbf{X} = [x_{t-W+1}, \dots, x_t]$ de tamaño $W=30$:

1. **Rampa Ideal de Progreso ($\mathbf{R}$):**
   $$r_k = x_1 + s_{min} \cdot (k-1), \quad k=1, \dots, W$$
   con pendiente mínima adaptativa:
   $$s_{min} = 0.01 \cdot \frac{|x_W - x_1|}{W}$$

2. **Meseta de Estancamiento Constante ($\mathbf{C}$):**
   $$c_k = x_1, \quad k=1, \dots, W$$

### 1.2. Algoritmo DTW con Banda de Sakoe-Chiba

La alineación óptima entre $\mathbf{X}$ y $\mathbf{Y} \in \{\mathbf{R}, \mathbf{C}\}$ se calcula con la matriz de costo acumulado $D(i,j)$:

$$D(i,j) = |x_i - y_j| + \min\left\{D(i-1,j),\ D(i,j-1),\ D(i-1,j-1)\right\}$$

restringido por la ventana temporal $|i-j| \le w$, donde $w = \lfloor 0.1 \cdot W \rfloor = 3$. La distancia final es $DTW(\mathbf{X}, \mathbf{Y}) = D(W, W)$.

### 1.3. Variante DDTW (Basada en Derivadas)

Aplica diferencias finitas previas al cálculo del DTW para lograr **invariancia ante escalas de magnitud**:
$$\nabla x_k = x_k - x_{k-1}, \quad \text{con } \nabla x_1 = 0$$
$$\text{DDTW}(\mathbf{X}, \mathbf{Y}) = \text{DTW}(\nabla \mathbf{X}, \nabla \mathbf{Y})$$

### 1.4. Umbrales Adaptativos mediante Percentiles Múltiples

Se mantienen tres buffers circulares históricos de métricas ($\mathcal{H}_{D_1}, \mathcal{H}_{D_2}, \mathcal{H}_\Delta$). Los umbrales de decisión se recalculan dinámicamente como:

$$\theta_c = \text{Perc}(\mathcal{H}_{D_2}, 30.0), \qquad \theta_r = \text{Perc}(\mathcal{H}_{D_1}, 70.0), \qquad \theta_\Delta = \text{Perc}(\mathcal{H}_\Delta, 70.0)$$

Condición triple de estancamiento:
$$\text{Stagnant}(t) = \left(N_{no\_improve} \ge 15\right) \land \left(D_2 \le \theta_c\right) \land \left(D_1 \ge \theta_r \lor \Delta \ge \theta_\Delta\right)$$

La alarma de rotación se activa tras una racha de confirmación de $P=3$ iteraciones consecutivas cumpliendo la condición triple.

---

## 2. Pool de Metaheurísticas y Protocolo de Inyección

### 2.1. Pool Poblacional ($\mathcal{P}_{pop}$ - Exploración Global)
1. **Particle Swarm Optimization (PSO):** Inercia decreciente lineal $w: 0.9 \to 0.4$, $c_1=2.0, c_2=2.0$.
2. **Grey Wolf Optimizer (GWO):** Jerarquía de caza $\alpha, \beta, \delta$ con coeficiente de decaimiento $a: 2 \to 0$.
3. **Whale Optimization Algorithm (WOA):** Alternancia entre maniobra de red de burbujas en espiral y búsqueda aleatoria.
4. **Elephant Herding Optimization (EHO):** Operadores de actualización de clan y separación de machos dominantes.
5. **Ant Colony Optimization (ACO):** Muestreo gaussiano sobre archivo de soluciones ponderadas por feromonas.

### 2.2. Pool de Trayectoria ($\mathcal{P}_{traj}$ - Explotación Local)
1. **Iterated Local Search (ILS):** Perturbaciones sistemáticas seguidas de descenso estocástico local.
2. **Variable Neighborhood Search (VNS):** Cambio jerárquico de $k_{max}$ estructuras de vecindario con fases de shaking y búsqueda local.
3. **Tabu Search (TS):** Memoria de corto plazo mediante lista tabú con radio de exclusión $r_{tabu}$ y criterio de aspiración.
4. **Simulated Annealing (SA):** Aceptación probabilística con enfriamiento geométrico $T_k = T_0 \cdot \alpha^k$.

### 2.3. Protocolo de Transferencia de Memoria ($x_{best}^*$)
Al activarse la rotación de solvers:
- **Rescate de Memoria:** Se recupera la mejor solución global válida $x_{best}^*$.
- **Inyección Poblacional:**
  - Solución #1: $x_{best}^*$ exacto.
  - 50% de la población: Perturbación gaussiana $x_{best}^* + \mathcal{N}(0, \sigma^2 \mathbf{I})$ con $\sigma=0.05$.
  - 50% restante: Muestreo uniforme exploratorio sobre $[L, U]^D$.
- **Inyección Trayectoria:** Punto de partida inicial $x_0 = x_{best}^*$.
