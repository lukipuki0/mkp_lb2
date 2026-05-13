# Informe de Tareas Pendientes y Propuestas de Mejora (DTW y Orquestador)

Este documento recopila las tareas pendientes y las 4 propuestas clave para sacarle el máximo provecho al sistema de detección de estancamiento (DTW) en el orquestador híbrido del MKP.

## 1. Perfiles de DTW Específicos por Algoritmo
**Estado:** Pendiente
**Descripción:** Actualmente todos los algoritmos usan la misma configuración global (`STAG_WINDOW`, `STAG_PATIENCE`). Sin embargo, su comportamiento es distinto:
*   **Poblacionales (GA, PSO, GWO):** Dan saltos grandes iniciales y luego se estabilizan lentamente. Requieren una ventana más grande (ej. `window = 40`) y mayor paciencia (`patience = 4`).
*   **Trayectoria (SA, TS):** Dan pasos locales muy pequeños. Requieren una ventana más corta (ej. `window = 20`) y menos paciencia (`patience = 2`) para rotar rápido si caen en un óptimo local.
**Acción:** Modificar `rotating_benchmark.py` para instanciar `StagnationConfig` con valores específicos dependiendo de si el algoritmo en turno es poblacional o de trayectoria.

## 2. Activar el DDTW (Derivadas)
**Estado:** Pendiente
**Descripción:** En `rotating_benchmark.py`, la variable `STAG_USE_DDTW` está en `False`. El DTW normal puede ser engañado por la escala de los valores de fitness (que pueden ser muy grandes en MKP).
**Acción:** Cambiar `STAG_USE_DDTW = True` para aislar la detección y enfocarla puramente en la **tendencia de mejora** (las derivadas), haciendo la detección más precisa e ignorando la "altura" absoluta del fitness.

## 3. Graficar el `Delta` del DTW en los Resultados
**Estado:** ✅ Implementado
**Descripción:** El método `update()` del DTW devuelve las distancias `D1`, `D2` y el `delta`. Esta es información matemática valiosísima para el informe final o tesis.
**Implementación realizada:**
*   Se agregó el campo `dtw_deltas` a los `EpochResult` de las 5 metaheurísticas (PSO, GA, GWO, SA, TS).
*   Se acumula el historial de deltas en el orquestador (`dtw_deltas_global` en `PipelineResult`).
*   Se exporta un archivo CSV (`historial_dtw.csv`) con columnas: `iteracion`, `fitness`, `dtw_delta`.
*   Se genera un gráfico de doble panel: Arriba la curva de Fitness y abajo la curva del Delta DTW, ambos coloreados por algoritmo.

## 4. Inyecciones de Población Inteligentes basadas en `Delta`
**Estado:** Pendiente
**Descripción:** El orquestador actualmente inyecta poblaciones al rotar (ej. Mutada, Random, Mixta). Esta decisión se puede tomar dinámicamente según la **gravedad del estancamiento**.
**Acción:** Leer el valor final de `delta` cuando el DTW dispara `fire = True`:
*   Si `delta` es moderado (estancamiento leve): Usar inyección **Mixta** o **Conservadora** para no perder las buenas soluciones actuales.
*   Si `delta` es extremadamente alto (estancamiento severo): Usar inyección **Mutada agresiva** o completamente **Random** para forzar la exploración lejos del óptimo local actual.

---

## 5. Registro Detallado de Resultados por Ejecución (Propuesta)
**Estado:** Pendiente
**Descripción:** Actualmente el `resumen_pipeline.txt` solo guarda la tabla de switches. Se propone enriquecer los resultados exportados para que cada ejecución quede completamente documentada y reproducible.
**Datos propuestos a registrar:**
*   **Metaheurísticas utilizadas:** Qué algoritmos se ejecutaron, en qué orden, y cuántas veces cada uno.
*   **Parámetros de cada MH:** Pop_size, iterations, crossover_rate, mutation_rate, T_inicial, alpha, tabu_tenure, etc. (según el tipo de MH).
*   **Parámetros DTW:** El `StagnationConfig` usado (window, patience, use_ddtw, etc.).
*   **Modo de inyección:** "random", "mutated" o "mixed".
*   **Resumen estadístico:** Fitness inicial vs final, mejora absoluta, mejora porcentual, número de fires DTW por MH.
*   **Formato sugerido:** Archivo JSON (`config_y_resultados.json`) que contenga toda la configuración y resultados en un solo archivo parseable, ideal para análisis automático o comparación entre ejecuciones.

