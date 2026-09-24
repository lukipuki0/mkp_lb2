# Continuous Metaheuristics & Adaptive WOA--ABC

Este repositorio contiene un framework modular para la implementación y
evaluación de metaheurísticas. Para la implementación activa de CEC2022 y
HRES2 se usará exclusivamente la MH poblacional WOA--ABC y sus siete variantes
adaptativas; las MH de trayectoria y los pipelines rotativos quedan fuera de
este experimento.

El proyecto incorpora un monitor de estancamiento dinámico basado en **Dynamic Time Warping (DTW)** y un plan de variantes adaptativas para WOA--ABC continuo.

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
├── woa_abc/                  # Wrappers WOA + ABC para funciones continuas
│   ├── adaptive.py           # Siete políticas/perfiles de parámetros DTW
│   ├── cooperativo_cec_dtw.py # Motor WOA--ABC exclusivo CEC
│   ├── cooperativo_hres2_dtw.py # Motor WOA--ABC exclusivo HRES2
│   ├── run_cec_variants.py   # Ejecutor CEC2022
│   ├── run_hres2_variants.py # Ejecutor HRES2
│   └── resultados/            # Salidas CEC/HRES2
│
├── hybrid_mkp/               # Orquestación de pipelines híbridos y rotación secuencial
├── plots/                    # Utilidades modulares de visualización de métricas
└── resultados/               # Carpeta de salida con gráficos, CSVs e informes TXT/MD
```

---

## Archivo MCDP

El código, datos, resultados y documentación MCDP se conservaron para una
etapa posterior en [`planes/futuro_mcdp`](planes/futuro_mcdp/). No forma parte
de la implementación continua activa.

---

## 📊 Benchmark Continuo (CEC2022)

Soporta la evaluación completa sobre las **12 funciones oficiales de CEC2022**:
- **F1**: Zakharov (Unimodal) - Bias: 300
- **F2-F5**: Rosenbrock, Schaffer F6, Rastrigin, Levy (Multimodales Básicas)
- **F6-F8**: Funciones Híbridas 1, 2 y 3 (Subfunciones agrupadas)
- **F9-F12**: Funciones de Composición 1, 2, 3 y 4 (Mezclas gaussianas)

---

## 🚀 Guía de Ejecución

### Variantes WOA--ABC sobre CEC2022

```bash
python -m woa_abc.run_cec_variants --variants all --runs 31 --iterations 1000 --ddtw
```

### Variantes WOA--ABC sobre HRES2

```bash
python -m woa_abc.run_hres2_variants --variants all --runs 31 --iterations 1000 --ddtw
```

---

## 📦 Estructura de Resultados

Los nuevos ejecutores guardan configuraciones, resultados por seed, curvas de
convergencia, estados DTW y perfiles de parámetros en
`woa_abc/resultados/cec/` y `woa_abc/resultados/hres2/`. El análisis estadístico
inferencial se añadirá en una etapa posterior.
