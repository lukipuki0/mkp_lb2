# Informe Técnico: Metaheurísticas en el Pipeline Híbrido de Rotación

Este documento describe el rol y el comportamiento de cada metaheurística (MH) dentro del **Pipeline Híbrido de Rotación guiado por DTW**. En este marco, los algoritmos no actúan en forma independiente: el `StagnationMonitor` (basado en DTW) detecta cuándo cada uno se estanca y le entrega el control al siguiente, pasando la mejor solución encontrada como punto de partida.

El pipeline alterna dos tipos de algoritmos:
- 🌍 **Poblacionales** (Exploración): GA, PSO, GWO — buscan zonas prometedoras en el espacio de soluciones.
- 🎯 **Trayectoria** (Explotación): SA, TS — profundizan y refinan la mejor solución encontrada por el algoritmo anterior.

---

## Arquitectura del Pipeline

```
[INICIO]
    │
    ▼
 🌍 Poblacional (GA | PSO | GWO) elegida al azar
    │  ← recibe sol_inyectada (si existe una solución global previa)
    │  ← corre hasta que DTW detecta estancamiento (fire = True)
    │  → entrega su mejor_solucion al orquestador
    ▼
 🎯 Trayectoria (SA | TS) elegida al azar
    │  ← recibe sol_inicial = mejor_solucion del paso anterior
    │  ← refina intensamente esa solución
    │  ← corre hasta que DTW detecta estancamiento (fire = True)
    │  → entrega su mejor_solucion al orquestador
    ▼
 ¿Tiempo agotado?
    NO → vuelve a Poblacional con la solución refinada
    SÍ → [FIN] → mejor solución global
```

El orquestador en `hybrid_mkp/orchestrator.py` mantiene en todo momento la mejor solución global (`solucion_global`) y la actualiza después de cada turno.

---

## Mecanismo de Traspaso: ¿Cómo pasan las soluciones?

### De Poblacional a Trayectoria
Cuando una MH poblacional termina su turno, el orquestador extrae su `mejor_solucion` (el mejor individuo/partícula/lobo encontrado). La MH de trayectoria **no arranca desde cero**: recibe directamente esa solución como punto de partida (`sol_inicial`), y la repara si es necesario para garantizar factibilidad.

### De Trayectoria a Poblacional
Cuando una MH de trayectoria termina su turno con una solución refinada, la siguiente MH poblacional la recibe como `sol_inyectada`. Esta solución **reemplaza al peor individuo** de la nueva población generada aleatoriamente. El resto de la población es aleatoria para mantener diversidad genética/de enjambre, pero el mejor individuo conocido ya está sembrado desde el inicio.

---

## 1. Algoritmo Genético (GA) — 🌍 Explorador

**Rol en el pipeline:** Exploración global. Busca en múltiples zonas del espacio de soluciones simultáneamente gracias a su población de individuos.

**¿Cómo recibe la solución previa?**
Al inicializar la población aleatoriamente, el peor individuo generado es reemplazado por la `sol_inyectada`. Esto garantiza que el conocimiento previo no se pierde, mientras el resto de la población explora libremente.

**¿Cómo termina su turno?**
Cuando el `StagnationMonitor` detecta que la curva del mejor valor de la generación se ha aplanado (se parece más a una línea constante que a una rampa ascendente), dispara `fire = True` y el GA ejecuta un `break` en su bucle de generaciones, entregando el control al orquestador.

**Parámetros en el pipeline:**
- `pop_size = 50`, `generations = 500`
- `stag_strategy = "abort"`, `stag_max_fires = 0` (sin límite: termina al primer disparo)
- Operadores: Cruce uniforme + Mutación bitflip (por defecto)

**Fortaleza en este contexto:** Gracias a su población, el GA puede escapar de óptimos locales y descubrir regiones del espacio de búsqueda que la trayectoria nunca exploraría por sí sola. Es el "cazador de territorios" del pipeline.

---

## 2. Particle Swarm Optimization (PSO) — 🌍 Explorador

**Rol en el pipeline:** Exploración guiada por inteligencia colectiva. El enjambre de partículas converge hacia zonas prometedoras atraído por la mejor solución personal de cada partícula y por el mejor global del enjambre.

**¿Cómo recibe la solución previa?**
La solución inyectada (`sol_inyectada`) reemplaza a la peor partícula del enjambre inicial. Adicionalmente, si el valor de la solución inyectada supera al mejor global del enjambre, se actualiza el `mejor_solucion_global` inmediatamente. Esto hace que todo el enjambre se oriente desde el inicio hacia esa zona.

**¿Cómo termina su turno?**
El monitor DTW observa el `mejor_valor_global` del enjambre en cada iteración. Al detectar la meseta (`fire = True`), el PSO hace un `break` en su bucle de iteraciones.

**Parámetros en el pipeline:**
- `pop_size = 30`, `iterations = 300`
- `stag_strategy = "abort"`, `stag_max_fires = 0`
- Binarización LB2 con parámetros G adaptativos

**Fortaleza en este contexto:** El PSO tiene una memoria colectiva: cada partícula recuerda su propia mejor posición (`mejor_solucion_personal`). Esto le permite explorar múltiples valles del espacio al mismo tiempo, siendo especialmente útil cuando la solución inyectada está en un óptimo local y el enjambre puede encontrar mejores vecindarios.

---

## 3. Grey Wolf Optimizer (GWO) — 🌍 Explorador

**Rol en el pipeline:** Exploración con convergencia jerárquica. La manada de lobos converge progresivamente desde exploración global hacia explotación local, guiada por los tres mejores individuos (Alpha, Beta, Delta).

**¿Cómo recibe la solución previa?**
La `sol_inyectada` reemplaza al peor lobo (Omega) de la manada inicial. La jerarquía (Alpha, Beta, Delta) se recalcula después de la inyección, de modo que si la solución inyectada es la mejor, automáticamente se convierte en Alpha y guía a toda la manada desde el primer paso.

**¿Cómo termina su turno?**
Igual que el PSO: el monitor DTW observa el `mejor_val` (valor del Alpha) en cada iteración. Al detectar estancamiento, el GWO hace un `break` en su bucle.

**Parámetros en el pipeline:**
- `pop_size = 30`, `iterations = 300`
- `stag_strategy = "abort"`, `stag_max_fires = 0`
- Binarización LB2 con parámetros G adaptativos vía DTW

**Fortaleza en este contexto:** El GWO tiene una transición natural de exploración a explotación controlada por el coeficiente `a` (decrece de 2 a 0 a lo largo de las iteraciones). Esto lo hace especialmente versátil: comienza explorando ampliamente y termina explotando. En el pipeline, cuando recibe una buena `sol_inyectada`, el Alpha la hereda y guía a toda la manada hacia esa zona desde el primer paso.

---

## 4. Simulated Annealing (SA) — 🎯 Explotador

**Rol en el pipeline:** Explotación probabilística. Refina intensamente la solución recibida mediante perturbaciones pequeñas (flip de bits), aceptando soluciones peores con probabilidad controlada por la temperatura (criterio de Metrópolis).

**¿Cómo recibe la solución previa?**
El SA recibe `sol_inicial` directamente. En vez de generar una solución aleatoria, arranca desde esa solución (reparándola si es necesario). La temperatura inicial es alta (`T_inicial = 5000`), lo que permite explorar alrededor de la zona sin quedarse atrapado inmediatamente.

**¿Cómo termina su turno?**
El monitor DTW observa el `mejor_val` del epoch en cada nivel de temperatura. Al detectar el estancamiento sostenido (`fire = True`), el SA ejecuta un `break` en su bucle `while T > T_final`, devolviendo la mejor solución encontrada hasta ese momento.

**Parámetros en el pipeline:**
- `T_inicial = 5000`, `T_final = 1.0`, `alpha = 0.97`, `iter_por_T = 50`
- `stag_strategy = "abort"`, `stag_max_fires = 0`

**Fortaleza en este contexto:** El SA es el refinador más clásico y robusto. Gracias al criterio de Metrópolis, no cae directamente en el óptimo local más cercano: puede saltar sobre barreras de energía. Al arrancar desde la mejor solución del GA/PSO/GWO, tiene una ventaja inicial enorme y puede pulir esa zona con mayor profundidad que un algoritmo poblacional.

---

## 5. Búsqueda Tabú (TS) — 🎯 Explotador

**Rol en el pipeline:** Explotación determinista e intensiva. En cada iteración, evalúa sistemáticamente múltiples vecinos y siempre elige el mejor, aunque sea peor que la solución actual. La lista tabú le impide volver sobre sus pasos recientes.

**¿Cómo recibe la solución previa?**
El TS recibe `sol_inicial` directamente y arranca desde ella (con reparación si aplica). La lista tabú comienza vacía: no hay restricciones en el primer paso.

**¿Cómo termina su turno?**
El monitor DTW observa el `mejor_val` en cada iteración del bucle principal. Al detectar estancamiento, el TS hace un `break` inmediato en su bucle de iteraciones.

**Parámetros en el pipeline:**
- `iterations = 2000`, `tabu_tenure = 10`, `neighborhood_sz = 30`
- `stag_strategy = "abort"`, `stag_max_fires = 0`

**Fortaleza en este contexto:** El TS es el explotador más agresivo del pipeline. A diferencia del SA (que acepta soluciones peores probabilísticamente), el TS *siempre* elige el mejor vecino disponible, moviéndose incluso en descensos para escapar de óptimos locales locales. Al recibir la solución del algoritmo poblacional anterior, puede rascar hasta el fondo de ese valle con mucha precisión. Su memoria tabú evita que se quede dando vueltas en el mismo punto.

---

## Resumen Comparativo

| MH  | Tipo         | Recibe            | Entrega           | Rol principal     | Fortaleza clave                     |
|-----|--------------|-------------------|-------------------|-------------------|-------------------------------------|
| GA  | 🌍 Poblacional | `sol_inyectada`  | `mejor_solucion`  | Explorar regiones | Diversidad genética, múltiples zonas|
| PSO | 🌍 Poblacional | `sol_inyectada`  | `mejor_solucion`  | Explorar guiado   | Memoria personal + global colectiva |
| GWO | 🌍 Poblacional | `sol_inyectada`  | `mejor_solucion`  | Explorar jerárquico | Alpha guía a la manada entera     |
| SA  | 🎯 Trayectoria | `sol_inicial`    | `mejor_solucion`  | Refinar amplio    | Acepta peores (Metrópolis), robusto |
| TS  | 🎯 Trayectoria | `sol_inicial`    | `mejor_solucion`  | Refinar profundo  | Determinista, memoria tabú precisa  |

---

## Condición de Parada Global

El pipeline corre en un bucle continuo de alternancia `Poblacional ↔ Trayectoria` hasta que se agota el **tiempo máximo** configurado en `rotating_benchmark.py` (`TIEMPO_MAX = 120` segundos por defecto). El número de switches es una consecuencia emergente de qué tan rápido se estanca cada algoritmo en cada turno: no está fijado de antemano.

---

## Algoritmos Pendientes (`pending_mh/`)

Para completar 4 MH de cada tipo, están planificados como trabajo futuro:
- **Differential Evolution (DE)** — Cuarto algoritmo poblacional: mutación diferencial entre individuos.
- **Hill Climbing (HC)** — Tercer algoritmo de trayectoria: explotación pura 1-flip hasta convergencia.
- **Iterated Local Search (ILS)** — Cuarto algoritmo de trayectoria: HC + perturbación aleatoria reiterada.
