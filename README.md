# Optimization Metaheuristics for the Multidimensional Knapsack Problem (MKP)

Este repositorio contiene la implementación y evaluación de diversas metaheurísticas aplicadas al **Problema de la Mochila Multidimensional (MKP)**. El enfoque principal del proyecto es el uso de técnicas de monitoreo de estancamiento basadas en **Dynamic Time Warping (DTW)** y la evaluación de múltiples estrategias (variantes) para escapar de óptimos locales.

## Estructura del Proyecto

El código está modularizado para facilitar la experimentación y la adición de nuevos algoritmos:

- **`mkp_core/`**: Módulo base compartido por todas las metaheurísticas.
  - `data_loader.py`: Funciones para cargar y parsear las instancias del MKP (ej. archivos del OR-Library).
  - `problem.py`: Definición de la estructura de datos del problema y evaluación de la función objetivo.
  - `repair.py`: Algoritmos de reparación para mantener la factibilidad de las soluciones generadas.

- **`dtw_stagnation.py`**: Implementación del monitor de estancamiento. Utiliza DTW para analizar la serie de tiempo de los valores objetivos recientes y detectar mesetas de estancamiento, disparando estrategias de rescate de forma dinámica.

- **`lb2/`**: Framework de binarización compartida (LB2). Implementa las funciones de transferencia L1/L2 vectorizadas, esenciales para convertir posiciones continuas a binarias en algoritmos como GWO y PSO, garantizando una equivalencia estricta con el diseño original.

- **`mh/`**: Módulo unificado con las implementaciones de las metaheurísticas:
  - `pso.py`: Particle Swarm Optimization (PSO).
  - `gwo.py`: Grey Wolf Optimizer (GWO).
  - `ga.py` y `ga_operators.py`: Algoritmo Genético (Genetic Algorithm).
  - `sa.py` y `sa_neighborhood.py`: Recocido Simulado (Simulated Annealing).
  - `ts.py` y `ts_neighborhood.py`: Búsqueda Tabú (Tabu Search).

- **`hybrid_mkp/`**: Lógica de orquestación para metaheurísticas híbridas y ejecución secuencial rotativa.

- **`plots/`**: Scripts de visualización para generar gráficas de convergencia, métricas instantáneas y seguimiento del DTW.

## Estrategias de Escape de Estancamiento (Variantes)

El proyecto evalúa estrategias avanzadas (y sus versiones originales) para recuperar la diversidad y escapar de óptimos locales cuando el `StagnationMonitor` detecta un estancamiento. Estas rutinas se activan dinámicamente según el análisis de la serie de tiempo de la convergencia.

## Uso y Ejecución

Para evaluar el rendimiento de los algoritmos y visualizar cómo alternan o convergen, puedes ejecutar el script principal unificado desde la raíz del proyecto.

```bash
python rotating_benchmark.py
```

Este script se encarga de probar las metaheurísticas y registrar los resultados en la carpeta `resultados/`, donde se guardarán tanto los `.csv` con las métricas como las visualizaciones en PDF/PNG.

## Archivos Ignorados (`.gitignore`)
Por defecto, los entornos virtuales y las carpetas de resultados están ignorados.
