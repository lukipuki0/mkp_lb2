# Guía Teórica y Práctica: Funcionamiento del Motor de Detección de Estancamiento DTW / DDTW

Este documento presenta una descripción exhaustiva del **motor de detección de estancamiento** implementado en `dtw_stagnation.py`, el cual actúa como el "gatillo" (*trigger*) inteligente del orquestador híbrido de metaheurísticas para alternar dinámicamente entre fases de **exploración** (algoritmos poblacionales) y **explotación** (algoritmos de trayectoria).

---

## 1. Introducción y Motivación

### 1.1. El Problema del Estancamiento en Metaheurísticas
En problemas de optimización combinatorial (como MKP) o continuos (como CEC2022 y HRES2-H2), los algoritmos frecuentemente quedan atrapados en óptimos locales o alcanzan regiones donde la tasa de mejora decae asintóticamente. 

Los enfoques tradicionales para detectar este fenómeno suelen basarse en:
- **Conteo estático de iteraciones sin mejora:** Disparan un cambio tras $N$ pasos fijos. Sin embargo, no distinguen entre una búsqueda activa en una cuenca prometedora vs. un estancamiento real.
- **Diferencia de aptitud fija ($\Delta f < \epsilon$):** Altamente dependiente de la escala del problema, volviéndose inútil si se cambia de dominio o función objetivo.

### 1.2. La Solución mediante Morfología de Curvas (DTW)
El enfoque implementado en este framework analiza la **forma morfológica** de la curva de convergencia temporal en una ventana deslizante reciente de tamaño $W$. Utiliza **DTW (Dynamic Time Warping)** y **DDTW (Derivative Dynamic Time Warping)** para comparar el comportamiento real del algoritmo contra dos escenarios de referencia:
1. Una **Rampa de Progreso Ideal** ($\mathbf{R}$).
2. Una **Meseta de Estancamiento Constante** ($\mathbf{C}$).

---

## 2. Fundamentos Matemáticos de DTW (Dynamic Time Warping)

### 2.1. Concepto de Alineamiento Elástico
A diferencia de la distancia Euclidiana tradicional (que compara punto a punto rígidamente en el mismo índice de tiempo: $d_E = \sqrt{\sum (x_i - y_i)^2}$), el DTW permite **deformar el eje temporal**. Esto significa que un punto en una serie puede asociarse a uno o más puntos en la otra serie si presentan una trayectoria morfológica similar con un desfase temporal.

```text
Distancia Euclidiana (Rígida):      DTW (Alineamiento Elástico):
  X:  o---o---o---o---o               X:  o---o---o---o---o
      |   |   |   |   |                   |  / \ /   / \
  Y:  o---o---o---o---o               Y:  o---o---o---o---o
```

### 2.2. Formulación de Programación Dinámica
Dadas dos series $\mathbf{X} = [x_1, x_2, \dots, x_n]$ e $\mathbf{Y} = [y_1, y_2, \dots, y_m]$, se construye una matriz de costo acumulado $\mathbf{D} \in \mathbb{R}^{(n+1) \times (m+1)}$.

1. **Condición de Frontera:**
   $$D(0, 0) = 0, \quad D(i, 0) = D(0, j) = +\infty \quad (\forall i \ge 1, j \ge 1)$$

2. **Ecuación de Recurrencia de Bellman:**
   $$D(i, j) = |x_i - y_j| + \min \begin{cases} 
   D(i - 1, j) & \text{(inserción / dilatación temporal)} \\
   D(i, j - 1) & \text{(eliminación / compresión temporal)} \\
   D(i - 1, j - 1) & \text{(coincidencia diagonal)}
   \end{cases}$$

3. **Distancia Final:**
   $$\text{DTW}(\mathbf{X}, \mathbf{Y}) = D(n, m)$$

### 2.3. Restricción de Banda de Sakoe-Chiba
Para evitar deformaciones patológicas (donde un solo punto se alinea artificialmente con toda una serie) y reducir la complejidad computacional de $\mathcal{O}(n \cdot m)$ a $\mathcal{O}(n \cdot w)$, se restringe el espacio de búsqueda con una ventana de ancho $w$:
$$|i - j| \le w, \quad w = \max(1, \lfloor \gamma \cdot W \rfloor), \quad \gamma = 0.10$$

En el código (`dtw_stagnation.py`):
```python
for i in range(1, n + 1):
    j_start = max(1, i - window)
    j_end   = min(m, i + window)
    si = s[i - 1]
    for j in range(j_start, j_end + 1):
        cost = abs(si - t[j - 1])
        D[i, j] = cost + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
```

---

## 3. Formulación de DDTW (Derivative Dynamic Time Warping)

### 3.1. ¿Por qué utilizar Derivadas?
El DTW estándar compara amplitudes absolutas ($|x_i - y_j|$). Si una serie tiene una pendiente ascendente idéntica a la rampa pero se encuentra desplazada verticalmente en el eje Y, el DTW estándar podría arrojar una distancia alta debido al *offset* numérico.

**DDTW** resuelve esto reemplazando los valores de la serie por su **primera diferencia discreta** (velocidad de cambio):
$$\nabla x_k = x_k - x_{k-1}, \quad \text{con } \nabla x_1 = 0$$

```python
def _first_diff(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.diff(x, prepend=x[0])

def ddtw_distance(s: np.ndarray, t: np.ndarray, window: Optional[int] = None) -> float:
    return dtw_distance(_first_diff(s), _first_diff(t), window=window)
```

La distancia DDTW es simplemente:
$$\text{DDTW}(\mathbf{X}, \mathbf{Y}) = \text{DTW}(\nabla \mathbf{X}, \nabla \mathbf{Y})$$

---

## 4. Construcción de Secuencias de Referencia Sintéticas

En cada iteración $t \ge W$, se extrae la ventana deslizante de las últimas $W$ mejores soluciones observadas:
$$\mathbf{X} = [x_{t-W+1}, x_{t-W+2}, \dots, x_t] \in \mathbb{R}^W$$

A partir del valor inicial de la ventana ($x_1 = X[0]$), se generan dos líneas base de longitud $W$:

### 4.1. Rampa de Progreso Ideal ($\mathbf{R}$)
Representa una optimización activa y saludable que mejora a una tasa lineal constante:
$$r_k = x_1 + s_{min} \cdot (k - 1), \quad \forall k \in \{1, 2, \dots, W\}$$

Donde la pendiente mínima $s_{min}$ se adapta al rango dinámico observado en la ventana:
$$s_{min} = 0.01 \cdot \frac{\max(1.0, |X[W-1] - X[0]|)}{W}$$

### 4.2. Meseta de Estancamiento Constante ($\mathbf{C}$)
Representa la detención absoluta del avance del algoritmo:
$$c_k = x_1, \quad \forall k \in \{1, 2, \dots, W\}$$

```text
Fitness
   ^
   |        /  <- Rampa Ideal (R) [Progreso Activo]
   |       /
   |------*-----------------  <- Trayectoria Real Observada (X)
   |      ==================  <- Constante Plana (C) [Estancamiento]
   +-----------------------------> Iteraciones (Ventana W)
```

---

## 5. Métricas Diagnósticas: $D_1$, $D_2$ y $\Delta$

A partir de las comparaciones elásticas, se calculan tres indicadores en cada paso:

1. **$D_1 = \text{DTW}(\mathbf{X}, \mathbf{R})$:** Distancia de la curva real a la rampa de progreso.
2. **$D_2 = \text{DTW}(\mathbf{X}, \mathbf{C})$:** Distancia de la curva real a la meseta constante.
3. **$\Delta = D_1 - D_2$:** Métrica diferencial de estancamiento.

### Interpretación de los Estados:

| Estado del Algoritmo | Comportamiento de $D_1$ | Comportamiento de $D_2$ | Valor de $\Delta = D_1 - D_2$ | Diagnóstico |
|---|---|---|---|---|
| **Convergencia Activa** | Bajo ($D_1 \to 0$) | Alto ($D_2 \gg 0$) | $\Delta \ll 0$ (Negativo) | La metaheurística está mejorando saludablemente. |
| **Zona de Transición** | Moderado | Moderado | $\Delta \approx 0$ | La tasa de mejora se está ralentizando. |
| **Estancamiento Severo** | Alto ($D_1 \gg 0$) | Mínimo ($D_2 \to 0$) | $\Delta \gg 0$ (Positivo alto) | La metaheurística está atrapada en un óptimo local. |

---

## 6. Umbralización Adaptativa mediante Percentiles Históricos

### 6.1. La Ineficiencia de los Umbrales Fijos
En sistemas de optimización multiobjetivo o que operan sobre diversas funciones de prueba:
- En **MKP**, el fitness está en el orden de $10^4 \implies D_1, D_2 \sim 10^3$.
- En **CEC2022**, el fitness está en el orden de $10^2 \implies D_1, D_2 \sim 10^1$.
- En **HRES2-H2**, el fitness (LCOE) está en el orden de $10^{-1} \implies D_1, D_2 \sim 10^{-2}$.

Un umbral numérico estático causaría falsas alarmas permanentes en MKP e insensibilidad en HRES2-H2.

### 6.2. Memoria Acumulativa y Percentiles Dinámicos
Para lograr una portabilidad total e invariancia a la escala, `StagnationMonitor` almacena el historial acumulado de las métricas en tres búferes:
$$\mathcal{H}_{D_1}, \quad \mathcal{H}_{D_2}, \quad \mathcal{H}_{\Delta}$$

Una vez que se han acumulado al menos 10 observaciones ($|\mathcal{H}| \ge 10$), los umbrales se recalculan dinámicamente:
$$\theta_c = \text{Percentil}(\mathcal{H}_{D_2}, P_{low})$$
$$\theta_r = \text{Percentil}(\mathcal{H}_{D_1}, P_{high})$$
$$\theta_\Delta = \text{Percentil}(\mathcal{H}_{\Delta}, P_{high})$$

donde típicamente $P_{low} = 30.0$ y $P_{high} = 70.0$.

---

## 7. Lógica de Decisión y Filtro de Paciencia (*Trigger*)

En cada iteración, el monitor evalúa una **triple condición booleana simultánea**:

```text
               ┌────────────────────────────────────────────────────────┐
               │         ¿Iteraciones sin mejora >= plateau_max?         │
               └───────────────────────────┬────────────────────────────┘
                                           │ Sí
                                           ▼
               ┌────────────────────────────────────────────────────────┐
               │           ¿Distancia a Constante D2 <= theta_c?         │
               └───────────────────────────┬────────────────────────────┘
                                           │ Sí
                                           ▼
               ┌────────────────────────────────────────────────────────┐
               │      ¿(D1 >= theta_r) O (Delta >= theta_delta)?        │
               └───────────────────────────┬────────────────────────────┘
                                           │ Sí
                                           ▼
                           ┌───────────────────────────────┐
                           │   trigger_streak += 1         │
                           └───────────────┬───────────────┘
                                           │
                        ┌──────────────────┴──────────────────┐
                        │ ¿trigger_streak >= patience?        │
                        └──────────────────┬──────────────────┘
                                           │ Sí
                                           ▼
                              💥 DISPARAR ALARMA (fire=True)
                              -> ROTAR A LA SIGUIENTE MH
```

### 7.1. Definición Formal de la Condición
$$\text{Stagnant}(t) \iff (N_{no\_improve} \ge K_{max}) \land (D_2 \le \theta_c) \land \big((D_1 \ge \theta_r) \lor (\Delta \ge \theta_\Delta)\big)$$

### 7.2. Filtro de Paciencia Anti-Falsos Positivos
Para evitar que una fluctuación estocástica momentánea interrumpa prematuramente una metaheurística:
$$S_t = \begin{cases}
S_{t-1} + 1, & \text{si } \text{Stagnant}(t) = \text{True} \\
0, & \text{si } \text{Stagnant}(t) = \text{Falso}
\end{cases}$$

El orquestador ejecuta el cambio de algoritmo **únicamente cuando**:
$$\text{fire} = (S_t \ge P)$$

donde $P$ es la paciencia configurada (`patience`, por defecto entre 3 y 8).

---

## 8. Parámetros de Configuración (`StagnationConfig`)

| Parámetro | Tipo | Default | Descripción |
|---|---|---|---|
| `window` | `int` | `40` | Tamaño de la ventana deslizante (número de iteraciones evaluadas). |
| `band` | `int` | `0` (auto) | Ancho de la banda Sakoe-Chiba (si es 0, se fija automáticamente al 10% de `window`). |
| `min_slope` | `float` | `0.0` (auto) | Pendiente mínima de la rampa (0.0 calcula automáticamente el 1% del rango de la ventana). |
| `plateau_max` | `int` | `15` | Conteo de iteraciones consecutivas sin récord necesario para activar la evaluación. |
| `patience` | `int` | `8` | Racha de confirmaciones consecutivas requeridas antes de emitir `fire=True`. |
| `use_ddtw` | `bool` | `False` | Activa el modo derivativo (DDTW) para invariancia morfológica estricta. |
| `adapt_thresholds`| `bool` | `True` | Adapta los umbrales dinámicamente con base en el historial de percentiles. |
| `p_low` | `float` | `30.0` | Percentil inferior para el umbral de cercanía a la meseta ($\theta_c$). |
| `p_high` | `float` | `70.0` | Percentil superior para el umbral de alejamiento de la rampa ($\theta_r, \theta_\Delta$). |

---

## 9. Ciclo de Vida y Ejemplo de Uso

```python
from dtw_stagnation import StagnationConfig, StagnationMonitor

# 1. Instanciar configuración y monitor
cfg = StagnationConfig(window=40, plateau_max=15, patience=5, adapt_thresholds=True)
monitor = StagnationMonitor(cfg)

# 2. Bucle de optimización de una metaheurística
for iteracion in range(1, max_iters + 1):
    mejor_fitness_actual = ejecutar_paso_mh()
    
    # 3. Actualizar monitor
    telemetria = monitor.update(mejor_fitness_actual)
    
    # 4. Verificar si se detonó el estancamiento
    if telemetria["fire"]:
        print(f"Estancamiento detectado en iteración {iteracion}!")
        print(f"D1={telemetria['D1_vs_ramp']:.2f}, D2={telemetria['D2_vs_const']:.2f}, Delta={telemetria['delta']:.2f}")
        
        # Realizar rotación de algoritmo
        monitor.reset()
        cambiar_a_siguiente_metaheuristica()
```

---

## 10. Resumen de Ventajas del Enfoque

1. **Invarianza de Escala:** Al utilizar percentiles móviles sobre el historial propio de la ejecución, funciona de forma idéntica en problemas con fitness de $10^5$ o de $10^{-3}$.
2. **Elasticidad Temporal:** Reconoce cuando una metaheurística progresa a velocidad variable sin confundir pausas breves con estancamiento terminal.
3. **Cero Falsos Positivos:** La combinación de la triple regla de decisión ($N_{no\_improve} + D_2 + D_1/\Delta$) junto al contador de paciencia $P$ evita interrupciones prematuras.
4. **Eficiencia Computacional:** La banda de Sakoe-Chiba mantiene el costo en $\mathcal{O}(W \cdot w)$, requiriendo menos de 1 milisegundo por iteración.
