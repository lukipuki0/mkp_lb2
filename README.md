# Optimization Metaheuristics for the Multidimensional Knapsack Problem (MKP)

Este repositorio contiene la implementación y evaluación de diversas metaheurísticas aplicadas al **Problema de la Mochila Multidimensional (MKP)**. El enfoque principal del proyecto es el uso de técnicas de monitoreo de estancamiento basadas en **Dynamic Time Warping (DTW)** y la evaluación de múltiples estrategias (variantes) para escapar de óptimos locales.

## Estructura del Proyecto

El código está modularizado para facilitar la experimentación y la adición de nuevos algoritmos:

- **`mkp_core/`**: Módulo base compartido por todas las metaheurísticas.
  - `data_loader.py`: Funciones para cargar y parsear las instancias del MKP (ej. archivos del OR-Library).
  - `problem.py`: Definición de la estructura de datos del problema y evaluación de la función objetivo.
  - `repair.py`: Algoritmos de reparación para mantener la factibilidad de las soluciones generadas.

- **`dtw_stagnation.py`**: Implementación del monitor de estancamiento. Utiliza DTW para analizar la serie de tiempo de los valores objetivos recientes y detectar mesetas de estancamiento, disparando estrategias de rescate de forma dinámica.

- **Implementaciones de Metaheurísticas**:
  - **`sa_mkp/`** y `sa_benchmark_variants.py`: Recocido Simulado (Simulated Annealing).
  - **`ga_mkp/`** y `ga_benchmark_variants.py`: Algoritmo Genético (Genetic Algorithm).
  - **`ts_mkp/`** y `ts_benchmark_variants.py`: Búsqueda Tabú (Tabu Search).
  - *(Otras metaheurísticas como PSO están contempladas en el flujo de trabajo).*

## Estrategias de Escape de Estancamiento (Variantes)

El proyecto evalúa 8 estrategias avanzadas (y sus versiones originales) para recuperar la diversidad y escapar de óptimos locales cuando el `StagnationMonitor` detecta estancamiento:

- **Original**: Estrategia de rescate clásica (ej. *reheat* en SA).
- **V1 (Exploit)**: Intensificación enfocada.
- **V2 (Cycle)**: Enfoque basado en ciclos.
- **V3 (Explore)**: Aumento agresivo de la exploración.
- **V4 (Nonlinear)**: Ajustes no lineales de los parámetros.
- **V5 (Heuristic)**: Reparación y búsqueda guiada por heurísticas.
- **V6 (LP)**: Relajación de Programación Lineal (Linear Programming).
- **V7 (Tabu LP)**: Integración de búsqueda tabú con optimización lineal.
- **V8 (Ruin & Recreate)**: Destrucción parcial de la solución y posterior reconstrucción.

## Uso y Ejecución

Para correr los benchmarks de cada algoritmo, puedes ejecutar los scripts correspondientes desde la raíz del proyecto. Estos scripts evaluarán las diferentes variantes y guardarán/mostrarán los resultados del rendimiento.

Ejemplo para correr el benchmark de Simulated Annealing:
```bash
python sa_benchmark_variants.py
```

Ejemplo para Algoritmo Genético:
```bash
python ga_benchmark_variants.py
```

## Archivos Ignorados (`.gitignore`)
Por defecto, la carpeta `resultados/`
