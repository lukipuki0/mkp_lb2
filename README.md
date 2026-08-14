# Optimization Metaheuristics & Hybrid Suite (MKP & CEC2022 Continuous Benchmark)

Este repositorio contiene un framework modular para la implementación, evaluación e hibridación de **metaheurísticas poblacionales y de trayectoria**, aplicadas tanto al **Problema de la Mochila Multidimensional (MKP)** en el dominio discreto como a la suite oficial de **Funciones de Benchmark Continuo CEC2022 (F1 a F12)**.

El proyecto incorpora un monitor de estancamiento dinámico basado en **Dynamic Time Warping (DTW)** y un espacio dedicado a la **hibridación de metaheurísticas (`mezclas_mh/`)**, incluyendo la suite híbrida **WOA-ABC (MDG-WABC)**.

---

## 🏛️ Arquitectura del Proyecto

```
mkp_lb2/
├── mkp_core/                 # Módulo base del problema discreto MKP
│   ├── data_loader.py        # Cargador y parser de instancias OR-Library
│   ├── problem.py            # Definición de la estructura del problema MKP
│   └── repair.py             # Algoritmo de reparación greedy y factibilidad
│
├── lb2/                      # Framework de binarización compartida (LB2)
│   ├── transfer.py           # Funciones de transferencia (V1-V4, S1-S4)
│   └── binarization.py       # Mapeo continuo-discreto vectorizado (L1/L2)
│
├── dtw_stagnation.py         # Monitor de estancamiento dinámico por DTW
│
├── mh/                       # Metaheurísticas individuales (Dominio Discreto MKP)
│   ├── pso.py                # Particle Swarm Optimization
│   ├── gwo.py                # Grey Wolf Optimizer
│   ├── ga.py                 # Algoritmo Genético
│   ├── sa.py                 # Recocido Simulado
│   ├── ts.py                 # Búsqueda Tabú
│   ├── aco.py                # Ant Colony Optimization
│   ├── ils.py                # Iterated Local Search
│   ├── eho.py                # Elk Herd Optimizer
│   ├── vns.py                # Variable Neighborhood Search
│   ├── woa.py                # Whale Optimization Algorithm
│   └── abc.py                # Artificial Bee Colony
│
├── continuous_benchmark/     # Benchmark de Optimización Continua (CEC2022)
│   ├── funciones_cec2022.py  # 12 Funciones CEC2022 (Unimodales, Multimodales, Híbridas, Composición)
│   ├── benchmark_continuo.py # Ejecutor por lotes del pipeline continuo
│   └── mh/                   # Adaptación de metaheurísticas al dominio continuo
│
├── mezclas_mh/               # 🔀 Mezclas e Hibridaciones de Metaheurísticas
│   └── woa_abc/              # Suite de Hibridación WOA-ABC
│       ├── variante_a.py     # Switching por parámetro dinámico |A| / a(t)
│       ├── variante_b.py     # Momentum-Guided + Paso adaptativo exponencial
│       ├── variante_c.py     # Switching dinámico por Diversidad Poblacional
│       ├── mdg_wabc.py       # MDG-WABC: Combinación de Momentum y Diversidad (B + C)
│       ├── run_experiment.py # Script de experimentos comparativos rápidos
│       └── benchmark_woa_abc.py # Benchmark unificado completo (Generador de resultados MKP + CEC2022)
│
├── hybrid_mkp/               # Orquestación de pipelines híbridos y rotación secuencial
├── plots/                    # Utilidades modulares de visualización de métricas
└── resultados/               # Carpeta de salida con gráficos, CSVs e informes TXT/MD
```

---

## 🔀 Suite de Mezclas de Metaheurísticas (`mezclas_mh/`)

La carpeta `mezclas_mh/` está diseñada para estudiar la interacción de pares o grupos de metaheurísticas en ambos dominios (discreto y continuo). 

Actualmente incluye la suite **WOA-ABC**:

1. **Variante A (Switching por $|A|$)**: Utiliza el parámetro dinámico $a(t)$ de WOA para alternar entre fases de exploración (WOA) y explotación (ABC).
2. **Variante B (Momentum-Guided)**: Incorpora un vector de impulso histórico y paso adaptativo exponencial $S(t) = S_0 \cdot e^{-\lambda t}$ para guiar la búsqueda en vecindarios.
3. **Variante C (Control por Diversidad)**: Calcula la diversidad poblacional normalizada respecto al centroide e intercambia las fases según un umbral dinámico.
4. **MDG-WABC (Variante B + C)**: Metaheurística híbrida combinada que integra guiado por momentum con control adaptativo de diversidad.

---

## 📊 Benchmark Continuo (CEC2022)

Soporta la evaluación completa sobre las **12 funciones oficiales de CEC2022**:
- **F1**: Zakharov (Unimodal) - Bias: 300
- **F2-F5**: Rosenbrock, Schaffer F6, Rastrigin, Levy (Multimodales Básicas)
- **F6-F8**: Funciones Híbridas 1, 2 y 3 (Subfunciones agrupadas)
- **F9-F12**: Funciones de Composición 1, 2, 3 y 4 (Mezclas gaussianas)

---

## 🚀 Guía de Ejecución

### 1. Ejecución del Benchmark Unificado de Mezclas (`woa_abc`)
Para ejecutar la suite completa de mezclas **WOA-ABC** evaluando tanto instancias **MKP** como **CEC2022**, generando reportes y gráficos por subcarpeta:

```bash
python -m mezclas_mh.woa_abc.benchmark_woa_abc
```

Genera la siguiente estructura dentro de `resultados/mezclas_mh/woa_abc/run_<TIMESTAMP>/`:
- **`mkp/`**: Subcarpetas para cada instancia MKP (`Instancia_01_...`), con gráficos de convergencia, instantáneo, CSV e informe TXT/MD.
- **`cec2022/`**: Subcarpetas para las 12 funciones continuas (`F1_...` a `F12_...`), con sus gráficos de convergencia, instantáneo, CSV e informe TXT/MD.
- **`resumen_general.md`**: Informe unificado global.

### 2. Ejecución Directa de una Variante Específica
Puedes probar individualmente cualquier variante de la mezcla directamente desde la terminal:

```bash
# Probar la variante combinada MDG-WABC
python mezclas_mh/woa_abc/mdg_wabc.py

# Probar la variante basada en diversidad
python mezclas_mh/woa_abc/variante_c.py
```

### 3. Ejecución del Benchmark Continuo General
Para evaluar el pipeline con monitor DTW sobre las 12 funciones del CEC2022:

```bash
python -m continuous_benchmark.benchmark_continuo
```

---

## 📦 Estructura de Resultados
Todos los scripts de benchmark generan automáticamente:
- **Gráficos PNG**: Curvas de convergencia histórica y fitness instantáneo por iteración.
- **Archivos CSV**: Series de tiempo de fitness y diferencias DTW.
- **Informes TXT / Markdown**: Tablas comparativas con óptimos conocidos y brechas relativas (Gap %).
