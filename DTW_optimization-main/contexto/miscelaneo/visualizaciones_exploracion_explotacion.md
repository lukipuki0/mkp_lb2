# Contexto Técnico: Visualizaciones de Exploración vs Explotación

## Objetivo

Crear un sistema de visualizaciones que muestre **cuantitativamente** cómo las metaheurísticas transicionan entre fases de exploración y explotación, usando el monitor DTW como detector de estancamiento y métricas poblacionales como indicadores de fase.

---

## 1. Problema: Multidimensional Knapsack Problem (MKP)

**Definición**: Dado un conjunto de `n` items con beneficios `p` y pesos en `m` dimensiones, seleccionar un subconjunto que maximice el beneficio total sin violar `m` restricciones de capacidad.

**Representación**: Vector binario `x ∈ {0,1}^n` donde `x[i]=1` significa incluir el item `i`.

**Dificultad para el balance exploración-explotación**:
- Múltiples restricciones acopladas → landscape rugoso con muchos óptimos locales
- Dependencia entre variables → dos soluciones similares pueden tener fitness muy diferente
- Efecto meseta → muchas soluciones con mismo fitness pero diferente estructura
- Trampa de factibilidad → la MH gasta iteraciones reparando en vez de mejorar

---

## 2. Metaheurísticas Implementadas

### 2.1 GeneticAlgorithm (GA) — Nativo combinatorio
- **Codificación**: Binaria directa
- **Operadores**: Torneo (k=3), crossover uniforme, mutación bit-flip
- **Parámetros EXPLOIT**: `cx_rate=0.9, mut_rate=0.01`
- **Parámetros EXPLORE**: `cx_rate=0.6, mut_rate=0.15`
- **Mecanismo de adaptación**: Cambio discreto de parámetros

### 2.2 BinaryPSO (PSO) — Adaptado desde continuo
- **Codificación**: Binaria con sigmoid (Kennedy & Eberhart 1997)
- **Velocidad**: `v = w*v + c1*r1*(pbest-x) + c2*r2*(gbest-x)`
- **Binarización**: `prob = 1/(1+exp(-v))`, luego `x = (rand < prob)`
- **Parámetros EXPLOIT**: `w=0.729, c1=1.49, c2=1.49` (Clerc & Kennedy 2002)
- **Parámetros EXPLORE**: `w=0.9, c1=2.5, c2=0.5`
- **Mecanismo de adaptación**: Cambio discreto de w, c1, c2

### 2.3 BinaryGWO (GWO) — Adaptado desde continuo
- **Codificación**: Binaria con sigmoid
- **Jerarquía**: Alpha (mejor), Beta (2do), Delta (3ro), Omega (resto)
- **Movimiento**: Promedio de vectores hacia alpha/beta/delta
- **Parámetro EXPLOIT**: `a=0.5` (lobos convergen)
- **Parámetro EXPLORE**: `a=2.0` (lobos exploran)
- **Mecanismo de adaptación**: Cambio discreto de `a`

### 2.4 ACO (pendiente de implementar) — Nativo combinatorio
- **Codificación**: Construcción secuencial sobre grafo
- **Mecanismo**: Feromonas + heurística (beneficio/peso)
- **Parámetros típicos**: α (peso feromona), β (peso heurística), ρ (evaporación)
- **Ventaja**: Respeta la estructura del problema, no necesita discretización

---

## 3. Monitor DTW (StagnationMonitor)

### 3.1 Funcionamiento

El monitor compara la **ventana de fitness** (últimas `window` iteraciones del mejor fitness) contra dos patrones de referencia:

1. **Rampa ideal**: Progreso constante con pendiente `min_slope`
2. **Meseta**: Estancamiento total (línea constante)

### 3.2 Métricas DTW

- **D1 (vs rampa)**: Distancia DTW entre la ventana y la rampa ideal
  - D1 alto → la curva NO se parece a progreso constante
- **D2 (vs meseta)**: Distancia DTW entre la ventana y la meseta
  - D2 bajo → la curva SE PARECE a estancamiento
- **delta = D1 - D2**: Positivo = más cerca de meseta = estancamiento

### 3.3 Lógica de Fire (3 condiciones simultáneas)

```python
fire = True cuando trigger_streak >= patience Y:
  1. cond_plateau:  no_improve_len >= plateau_max
  2. cond_constant: D2 <= theta_c     (curva se parece a meseta)
  3. cond_ramp:     D1 >= theta_r  OR  delta >= theta_delta
```

### 3.4 Umbrales adaptativos

Con `adapt_thresholds=True`:
- `theta_c = percentile(D2_hist, p_low=30)` → umbral bajo para D2
- `theta_r = percentile(D1_hist, p_high=70)` → umbral alto para D1
- `theta_delta = percentile(delta_hist, p_high=70)` → umbral alto para delta

### 3.5 Configuración actual (fire_binario)

```python
StagnationConfig(
    window=15,        # Ventana de observación
    band=2,           # Banda Sakoe-Chiba
    plateau_max=10,   # Iteraciones sin mejora para considerar plateau
    patience=2,       # Confirmaciones consecutivas para fire
    min_slope=2.0,    # Pendiente mínima de la rampa
    use_ddtw=True,    # Usar Derivative DTW (más robusto)
    adapt_thresholds=True,  # Umbrales adaptativos via percentiles
)
```

---

## 4. Estructura de Datos Disponible

El runner retorna por cada epoch:

```python
{
    "mejor_fitness": float,           # Mejor fitness encontrado
    "mejor_solucion": np.ndarray,     # Vector binario óptimo
    "optimo_conocido": float,         # Óptimo de la literatura
    "historial_fitness": List[float], # Fitness del mejor en cada iteración
    "historial_dtw": List[Dict],      # Métricas DTW por iteración
    "historial_modos": List[str],     # "exploit" o "explore" por iteración
    "fire_count": int,                # Cantidad de transiciones a explore
    "ganancia": float,                # fitness/optimo * 100
}
```

### 4.1 Contenido de historial_dtw (por iteración)

```python
{
    "ready": bool,              # True si ya pasó el warm-up (window)
    "fire": bool,               # True si disparó estancamiento
    "D1_vs_ramp": float,        # Distancia DTW vs rampa
    "D2_vs_const": float,       # Distancia DTW vs meseta
    "delta": float,             # D1 - D2
    "theta_c": float,           # Umbral adaptativo para D2
    "theta_r": float,           # Umbral adaptativo para D1
    "theta_delta": float,       # Umbral adaptativo para delta
    "no_improve_len": int,      # Iteraciones sin mejora
    "trigger_streak": int,      # Confirmaciones consecutivas
    "n": int,                   # Iteración actual
}
```

### 4.2 Datos poblacionales (NO guardados actualmente)

Para las métricas de exploración/explotación necesitamos acceder a la población en cada iteración. **Esto requiere modificar el runner** para guardar:

```python
"historial_poblacion": List[np.ndarray]  # Población completa por iteración
```

**Alternativa**: Calcular las métricas en tiempo real durante la ejecución y guardar solo los valores agregados:

```python
"historial_diversidad": List[float],     # Diversidad poblacional
"historial_entropia": List[float],       # Entropía de bits
"historial_intensificacion": List[float] # Intensificación alrededor del mejor
```

---

## 5. Métricas de Exploración vs Explotación

### 5.1 Diversidad Poblacional (Hamming para binario)

**Qué mide**: Cuán dispersas están las soluciones en la población.

**Fórmula**:
```
D(t) = (1 / N*(N-1)) * Σ_{i<j} d_H(x_i, x_j) / n
```

Donde:
- `N` = tamaño de población
- `n` = dimensión del problema (cantidad de items)
- `d_H(x_i, x_j)` = distancia Hamming entre dos soluciones

**Interpretación**:
- D(t) alto → exploración (soluciones diversas)
- D(t) bajo → explotación (soluciones similares)

**Implementación**:
```python
def hamming_diversity(population: np.ndarray) -> float:
    """
    Args:
        population: Array (N, n) de soluciones binarias
    Returns:
        Diversidad normalizada [0, 1]
    """
    N, n = population.shape
    if N < 2:
        return 0.0
    
    total_dist = 0
    for i in range(N):
        for j in range(i+1, N):
            total_dist += np.sum(population[i] != population[j])
    
    max_dist = N * (N - 1) / 2 * n
    return total_dist / max_dist
```

### 5.2 Entropía de la Población

**Qué mide**: Incertidumbre en cada posición del vector binario.

**Fórmula**:
```
H(t) = -(1/n) * Σ_{j=1}^{n} [p_j * log2(p_j) + (1-p_j) * log2(1-p_j)]
```

Donde `p_j` = proporción de 1s en la posición `j` de la población.

**Interpretación**:
- H(t) alto (cercano a 1) → exploración (bits impredecibles)
- H(t) bajo (cercano a 0) → explotación (bits convergidos)

**Implementación**:
```python
def population_entropy(population: np.ndarray) -> float:
    """
    Args:
        population: Array (N, n) de soluciones binarias
    Returns:
        Entropía normalizada [0, 1]
    """
    N, n = population.shape
    p = np.mean(population, axis=0)  # Proporción de 1s por posición
    
    # Evitar log(0)
    p = np.clip(p, 1e-10, 1 - 1e-10)
    
    entropy = -np.mean(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    return entropy
```

### 5.3 Intensificación (Cercanía al Mejor)

**Qué mide**: Cuán concentrada está la población alrededor de la mejor solución.

**Fórmula**:
```
I(t) = 1 - (1/N) * Σ_{i=1}^{N} d_H(x_i, x_best) / n
```

**Interpretación**:
- I(t) alto (cercano a 1) → explotación (todos cerca del mejor)
- I(t) bajo (cercano a 0) → exploración (dispersos)

**Implementación**:
```python
def intensification(population: np.ndarray, best: np.ndarray) -> float:
    """
    Args:
        population: Array (N, n) de soluciones binarias
        best: Array (n,) mejor solución
    Returns:
        Intensificación [0, 1]
    """
    N, n = population.shape
    distances = np.sum(population != best, axis=1)
    mean_dist = np.mean(distances)
    return 1.0 - (mean_dist / n)
```

### 5.4 Tasa de Cambio Poblacional

**Qué mide**: Cuántos bits cambian entre generaciones.

**Fórmula**:
```
Δ(t) = (1/N) * Σ_{i=1}^{N} d_H(x_i(t), x_i(t-1)) / n
```

**Interpretación**:
- Δ(t) alto → exploración agresiva (muchos cambios)
- Δ(t) bajo → convergencia (pocos cambios)

**Implementación**:
```python
def population_change_rate(pop_t: np.ndarray, pop_t_minus_1: np.ndarray) -> float:
    """
    Args:
        pop_t: Población en iteración t
        pop_t_minus_1: Población en iteración t-1
    Returns:
        Tasa de cambio [0, 1]
    """
    N, n = pop_t.shape
    changes = np.sum(pop_t != pop_t_minus_1)
    return changes / (N * n)
```

---

## 6. Visualizaciones Propuestas

### 6.1 Gráfico Dual: Fitness + Diversidad + Zonas de Fase

**Objetivo**: Mostrar la relación entre progreso (fitness) y comportamiento (exploración/explotación).

**Estructura**:
```
Figura (12x8)
├── Panel superior (60% altura): Fitness
│   ├── Eje Y izquierdo: Fitness (mejor y promedio si está disponible)
│   ├── Eje X: Iteraciones
│   ├── Línea sólida: Modo exploit
│   ├── Línea punteada: Modo explore
│   ├── Zonas sombreadas: Azul=exploración, Rojo=explotación
│   └── Marcadores verticales: Puntos donde DTW disparó fire
│
└── Panel inferior (40% altura): Diversidad
    ├── Eje Y: Diversidad poblacional [0, 1]
    ├── Eje X: Iteraciones (compartido con panel superior)
    ├── Línea: Diversidad (Hamming o entropía)
    └── Zonas sombreadas: Igual que panel superior
```

**Datos necesarios**:
- `historial_fitness` (ya disponible)
- `historial_modos` (ya disponible)
- `historial_dtw` (ya disponible, para detectar fires)
- `historial_diversidad` (requiere modificación del runner)

**Código base**:
```python
def plot_fitness_diversity_dual(
    historial_fitness: List[float],
    historial_diversidad: List[float],
    historial_modos: List[str],
    historial_dtw: List[Dict],
    optimo: float = None,
    title: str = "Fitness y Diversidad",
    save_path: str = None
):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), 
                                     gridspec_kw={'height_ratios': [3, 2]},
                                     sharex=True)
    
    iters = list(range(len(historial_fitness)))
    
    # Detectar zonas de explore/exploit
    explore_zones = []
    in_explore = False
    start = 0
    for i, mode in enumerate(historial_modos):
        if mode == "explore" and not in_explore:
            start = i
            in_explore = True
        elif mode == "exploit" and in_explore:
            explore_zones.append((start, i))
            in_explore = False
    if in_explore:
        explore_zones.append((start, len(historial_modos)))
    
    # Panel 1: Fitness
    ax1.plot(iters, historial_fitness, 'b-', linewidth=1.5, label='Fitness')
    if optimo:
        ax1.axhline(y=optimo, color='r', linestyle='--', alpha=0.7, 
                    label=f'Óptimo ({optimo:.0f})')
    
    # Sombrear zonas
    for start, end in explore_zones:
        ax1.axvspan(start, end, alpha=0.2, color='blue', label='Exploración' if start == explore_zones[0][0] else "")
    
    # Marcar fires
    fire_iters = [i for i, h in enumerate(historial_dtw) if h.get('fire')]
    for fire_iter in fire_iters:
        ax1.axvline(x=fire_iter, color='orange', linestyle=':', alpha=0.5, linewidth=1)
    
    ax1.set_ylabel('Fitness')
    ax1.set_title(title)
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    
    # Panel 2: Diversidad
    ax2.plot(iters, historial_diversidad, 'g-', linewidth=1.5, label='Diversidad')
    
    for start, end in explore_zones:
        ax2.axvspan(start, end, alpha=0.2, color='blue')
    
    ax2.set_xlabel('Iteración')
    ax2.set_ylabel('Diversidad')
    ax2.set_ylim(0, 1)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
```

### 6.2 Heatmap de la Población

**Objetivo**: Mostrar visualmente cómo la población converge y se diversifica.

**Estructura**:
```
Figura con múltiples subplots (snapshots en iteraciones clave)
├── Snapshot 1: Iteración inicial (exploración pura)
├── Snapshot 2: Primera convergencia
├── Snapshot 3: Estancamiento detectado (fire)
├── Snapshot 4: Después del fire (diversificación)
└── Snapshot 5: Final (convergencia final)

Cada snapshot:
├── Matriz: Filas=individuos, Columnas=items
├── Color: 0=blanco, 1=negro (o colormap)
└── Título: "Iter X | Fitness=Y | Diversidad=Z"
```

**Datos necesarios**:
- `historial_poblacion` (requiere modificación del runner)
- O calcular snapshots en tiempo real

**Código base**:
```python
def plot_population_heatmap(
    snapshots: List[Tuple[int, np.ndarray, float, float]],
    # [(iter, population, fitness, diversity), ...]
    save_path: str = None
):
    n_snapshots = len(snapshots)
    fig, axes = plt.subplots(1, n_snapshots, figsize=(4*n_snapshots, 6))
    
    if n_snapshots == 1:
        axes = [axes]
    
    for ax, (iter, pop, fit, div) in zip(axes, snapshots):
        ax.imshow(pop, aspect='auto', cmap='binary', interpolation='nearest')
        ax.set_title(f'Iter {iter}\nFit={fit:.0f}\nDiv={div:.3f}')
        ax.set_xlabel('Items')
        ax.set_ylabel('Individuos')
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
```

### 6.3 Phase Portrait: Exploración vs Explotación

**Objetivo**: Mostrar la trayectoria de la MH en el espacio diversidad-intensificación.

**Estructura**:
```
Figura (8x8)
├── Eje X: Diversidad (exploración) [0, 1]
├── Eje Y: Intensificación (explotación) [0, 1]
├── Trayectoria: Línea con flechas mostrando evolución temporal
├── Color: Gradiente temporal (azul=inicio, rojo=final)
├── Puntos especiales: Marcadores en fires del DTW
└── Zonas ideales:
    - Esquina superior-izquierda: Alta intensificación, baja diversidad (explotación pura)
    - Esquina inferior-derecha: Baja intensificación, alta diversidad (exploración pura)
    - Diagonal: Balance ideal
```

**Datos necesarios**:
- `historial_diversidad`
- `historial_intensificacion`
- `historial_dtw` (para marcar fires)

**Código base**:
```python
def plot_phase_portrait(
    historial_diversidad: List[float],
    historial_intensificacion: List[float],
    historial_dtw: List[Dict],
    title: str = "Phase Portrait: Exploración vs Explotación",
    save_path: str = None
):
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Trayectoria con gradiente de color
    n = len(historial_diversidad)
    colors = plt.cm.viridis(np.linspace(0, 1, n))
    
    for i in range(n-1):
        ax.plot(historial_diversidad[i:i+2], 
                historial_intensificacion[i:i+2],
                color=colors[i], linewidth=2, alpha=0.7)
    
    # Marcar inicio y final
    ax.scatter(historial_diversidad[0], historial_intensificacion[0],
               color='green', s=200, marker='o', label='Inicio', zorder=5)
    ax.scatter(historial_diversidad[-1], historial_intensificacion[-1],
               color='red', s=200, marker='s', label='Final', zorder=5)
    
    # Marcar fires
    fire_iters = [i for i, h in enumerate(historial_dtw) if h.get('fire')]
    for fire_iter in fire_iters:
        ax.scatter(historial_diversidad[fire_iter], 
                   historial_intensificacion[fire_iter],
                   color='orange', s=100, marker='*', zorder=5)
    
    # Zonas de referencia
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.3)
    
    ax.text(0.1, 0.9, 'Explotación\npura', ha='center', va='center',
            fontsize=10, alpha=0.5, style='italic')
    ax.text(0.9, 0.1, 'Exploración\npura', ha='center', va='center',
            fontsize=10, alpha=0.5, style='italic')
    
    ax.set_xlabel('Diversidad (Exploración)', fontsize=12)
    ax.set_ylabel('Intensificación (Explotación)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
```

### 6.4 DTW Distance Over Time

**Objetivo**: Mostrar cómo el DTW detecta el estancamiento.

**Estructura**:
```
Figura (12x6)
├── Eje Y: Distancia DTW
├── Eje X: Iteraciones
├── Línea 1: D1 (vs rampa)
├── Línea 2: D2 (vs meseta)
├── Línea 3: delta = D1 - D2
├── Líneas horizontales: theta_c, theta_r, theta_delta
├── Zonas sombreadas: Donde las 3 condiciones se cumplen
└── Marcadores verticales: Fires disparados
```

**Datos necesarios**:
- `historial_dtw` (ya disponible)

**Código base**:
```python
def plot_dtw_stagnation_detection(
    historial_dtw: List[Dict],
    title: str = "Detección de Estancamiento con DTW",
    save_path: str = None
):
    ready = [h for h in historial_dtw if h.get('ready')]
    if not ready:
        print("Sin datos DTW (no alcanzó warm-up)")
        return
    
    iters = list(range(len(ready)))
    d1s = [h['D1_vs_ramp'] for h in ready]
    d2s = [h['D2_vs_const'] for h in ready]
    deltas = [h['delta'] for h in ready]
    theta_cs = [h['theta_c'] for h in ready]
    theta_rs = [h['theta_r'] for h in ready]
    theta_deltas = [h['theta_delta'] for h in ready]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(iters, d1s, label='D1 (vs rampa)', linewidth=1.5, alpha=0.8)
    ax.plot(iters, d2s, label='D2 (vs meseta)', linewidth=1.5, alpha=0.8)
    ax.plot(iters, deltas, label='delta (D1-D2)', linewidth=2, alpha=0.9)
    
    ax.plot(iters, theta_cs, '--', label='θ_c', alpha=0.5, linewidth=1)
    ax.plot(iters, theta_rs, '--', label='θ_r', alpha=0.5, linewidth=1)
    ax.plot(iters, theta_deltas, '--', label='θ_delta', alpha=0.5, linewidth=1)
    
    # Marcar fires
    fire_iters = [i for i, h in enumerate(ready) if h.get('fire')]
    for fire_iter in fire_iters:
        ax.axvline(x=fire_iter, color='red', linestyle=':', alpha=0.7, linewidth=2)
    
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    
    ax.set_xlabel('Iteración (desde warm-up)')
    ax.set_ylabel('Distancia DTW')
    ax.set_title(title)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
```

### 6.5 Comparación Multi-MH

**Objetivo**: Comparar el comportamiento de las 4 MHs en las mismas métricas.

**Estructura**:
```
Figura (12x10) con 4 paneles (2x2)
├── Panel 1 (superior-izq): Fitness promedio de las 4 MHs
├── Panel 2 (superior-der): Diversidad promedio de las 4 MHs
├── Panel 3 (inferior-izq): Intensificación promedio de las 4 MHs
└── Panel 4 (inferior-der): Tabla resumen con métricas finales
```

**Datos necesarios**:
- Resultados de múltiples epochs de las 4 MHs
- Promediar sobre epochs para suavizar

---

## 7. Modificaciones Requeridas al Runner

### 7.1 Opción A: Guardar población completa (más flexible, más memoria)

```python
# En runner.py, agregar:
historial_poblacion = []

for it in range(num_iteraciones):
    fitness = mh.step()
    
    # Guardar población
    if hasattr(mh, 'poblacion'):
        if isinstance(mh.poblacion, list):
            historial_poblacion.append(np.array(mh.poblacion))
        else:
            historial_poblacion.append(mh.poblacion.copy())
    
    # ... resto del loop

return {
    # ... campos existentes
    "historial_poblacion": historial_poblacion,
}
```

### 7.2 Opción B: Calcular métricas en tiempo real (menos memoria)

```python
# En runner.py, agregar:
historial_diversidad = []
historial_entropia = []
historial_intensificacion = []

for it in range(num_iteraciones):
    fitness = mh.step()
    
    # Calcular métricas poblacionales
    if hasattr(mh, 'poblacion'):
        pop = mh.poblacion if isinstance(mh.poblacion, np.ndarray) else np.array(mh.poblacion)
        best = mh.gbest if hasattr(mh, 'gbest') else pop[np.argmax(mh.fitness_pop)]
        
        historial_diversidad.append(hamming_diversity(pop))
        historial_entropia.append(population_entropy(pop))
        historial_intensificacion.append(intensification(pop, best))
    
    # ... resto del loop

return {
    # ... campos existentes
    "historial_diversidad": historial_diversidad,
    "historial_entropia": historial_entropia,
    "historial_intensificacion": historial_intensificacion,
}
```

**Recomendación**: Usar Opción B para producción (menos memoria, más rápido). Usar Opción A solo si necesitás analizar la población post-mortem.

---

## 8. Consideraciones Técnicas

### 8.1 Rendimiento

- **Cálculo de diversidad**: O(N² * n) por iteración. Con N=20, n=100 → 40,000 operaciones. Aceptable.
- **Cálculo de entropía**: O(N * n) por iteración. Muy rápido.
- **Cálculo de intensificación**: O(N * n) por iteración. Muy rápido.

**Optimización**: Calcular métricas cada K iteraciones (ej: cada 5) en vez de en cada iteración.

### 8.2 Memoria

- **Opción A (guardar población)**: 100 iteraciones * 20 individuos * 100 items * 8 bytes = 1.6 MB por epoch. Con 10 epochs = 16 MB. Aceptable.
- **Opción B (solo métricas)**: 100 iteraciones * 3 métricas * 8 bytes = 2.4 KB por epoch. Despreciable.

### 8.3 Compatibilidad con MHs

- **GA**: `mh.poblacion` es lista de arrays. Convertir a `np.array(mh.poblacion)`.
- **PSO**: `mh.poblacion` es `np.ndarray` directamente.
- **GWO**: `mh.poblacion` es lista de arrays. Convertir a `np.array(mh.poblacion)`.
- **ACO**: No tiene población en el mismo sentido. Las métricas deben adaptarse (ej: diversidad de las soluciones construidas en la iteración actual).

### 8.4 Normalización

Todas las métricas están normalizadas a [0, 1] para facilitar la comparación entre MHs y problemas de diferente tamaño.

### 8.5 Suavizado

Para visualizaciones más limpias, aplicar media móvil (window=5) a las métricas antes de graficar:

```python
def moving_average(x: List[float], window: int = 5) -> List[float]:
    return np.convolve(x, np.ones(window)/window, mode='valid')
```

---

## 9. Plan de Implementación

### Fase 1: Métricas poblacionales (Opción B)

1. Agregar funciones de métricas en `mkp_common/metrics.py`:
   - `hamming_diversity()`
   - `population_entropy()`
   - `intensification()`
   - `population_change_rate()`

2. Modificar `runner.py` para calcular y guardar métricas (Opción B)

3. Actualizar `results.py` para incluir métricas en JSON de salida

### Fase 2: Visualizaciones básicas

4. Implementar `plot_fitness_diversity_dual()` en `mkp_common/visualizations.py`

5. Implementar `plot_dtw_stagnation_detection()` (ya existe parcialmente en `results.py`)

### Fase 3: Visualizaciones avanzadas

6. Implementar `plot_phase_portrait()`

7. Implementar `plot_population_heatmap()` (requiere Opción A o snapshots)

### Fase 4: Comparación multi-MH

8. Implementar `plot_multi_mh_comparison()`

9. Generar tablas resumen con métricas agregadas

### Fase 5: Integración con ACO

10. Adaptar métricas para ACO (o crear métricas específicas)

11. Agregar ACO al sistema de comparación

---

## 10. Ejemplo de Uso

```python
from mkp_common import BinaryPSO, cargar_instancia
from mkp_common.monitor import StagnationConfig
from fire_binario.runner import run_epochs
from mkp_common.visualizations import (
    plot_fitness_diversity_dual,
    plot_phase_portrait,
    plot_dtw_stagnation_detection
)

# Cargar instancia
inst = cargar_instancia("instances/mknapcb4.txt", idx=0)

# Configurar DTW
dtw_cfg = StagnationConfig(window=15, patience=2, use_ddtw=True)

# Ejecutar
resultados = run_epochs(
    mh_class=BinaryPSO,
    inst=inst,
    monitor_cfg=dtw_cfg,
    num_particulas=20,
    num_iteraciones=100,
    epochs=1,
    verbose=False
)

# Tomar el mejor epoch
best_idx = np.argmax([r['mejor_fitness'] for r in resultados])
best_result = resultados[best_idx]

# Visualizar
plot_fitness_diversity_dual(
    historial_fitness=best_result['historial_fitness'],
    historial_diversidad=best_result['historial_diversidad'],
    historial_modos=best_result['historial_modos'],
    historial_dtw=best_result['historial_dtw'],
    optimo=inst['optimo'],
    title="PSO Binario con DTW Auto-Adaptativo",
    save_path="results/pso_fitness_diversity.png"
)

plot_phase_portrait(
    historial_diversidad=best_result['historial_diversidad'],
    historial_intensificacion=best_result['historial_intensificacion'],
    historial_dtw=best_result['historial_dtw'],
    title="Phase Portrait - PSO",
    save_path="results/pso_phase_portrait.png"
)

plot_dtw_stagnation_detection(
    historial_dtw=best_result['historial_dtw'],
    title="Detección de Estancamiento - PSO",
    save_path="results/pso_dtw_detection.png"
)
```

---

## 11. Referencias

- **DTW**: Sakoe, H., & Chiba, S. (1978). Dynamic programming algorithm optimization for spoken word recognition.
- **Binary PSO**: Kennedy, J., & Eberhart, R. C. (1997). A discrete binary version of the particle swarm algorithm.
- **GWO**: Mirjalili, S., et al. (2014). Grey wolf optimizer.
- **MKP**: Chu, P. C., & Beasley, J. E. (1998). A genetic algorithm for the multidimensional knapsack problem.
- **Exploration-Exploitation balance**: Črepinšek, M., et al. (2013). Exploration and exploitation in evolutionary algorithms: A survey.

---

## 12. Gotchas y Notas

1. **Warm-up del DTW**: Las primeras `window` iteraciones no tienen métricas DTW (ready=False). Las visualizaciones deben manejar esto.

2. **Modo inicial**: Todas las MHs inician en modo "exploit". El primer fire puede tardar en llegar.

3. **Reparación greedy**: La función `reparar()` puede cambiar significativamente la población, afectando las métricas de diversidad. Considerar calcular métricas ANTES de reparar (si es posible).

4. **Elitismo en GA**: El GA mantiene los 2 mejores individuos sin cambios. Esto reduce artificialmente la diversidad. Considerar excluirlos del cálculo.

5. **PSO con velocidades**: En PSO, las velocidades también son un indicador de exploración (velocidades altas = exploración). Podría agregarse como métrica adicional.

6. **GWO con líderes**: En GWO, la distancia entre alpha/beta/delta es un indicador de convergencia. Si los 3 líderes son muy similares → explotación.

7. **Escalado de ejes**: Para comparar MHs, usar los mismos límites de ejes en todas las visualizaciones.

8. **Colores consistentes**: Usar los mismos colores para cada MH en todas las visualizaciones (ya definido en `MH_COLORS`).

9. **Guardado de plots**: Usar `dpi=150` para buena calidad sin archivos gigantes. Formato PNG por defecto, considerar SVG para papers.

10. **Reproducibilidad**: Guardar la semilla usada en cada epoch para poder reproducir visualizaciones específicas.
