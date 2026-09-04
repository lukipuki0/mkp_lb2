# Optimization Metaheuristics & Cooperative WOA--ABC for MCDP

Este repositorio contiene un framework modular para la implementación, evaluación e hibridación de **metaheurísticas poblacionales y de trayectoria**, aplicadas al **Machine Cell Design Problem (MCDP)** y a otros dominios del proyecto.

El proyecto incorpora un monitor de estancamiento dinámico basado en **Dynamic Time Warping (DTW)** y una mezcla cooperativa de **WOA + ABC** para MCDP.

---

## 🏛️ Arquitectura del Proyecto

```
mkp_lb2/
├── mkp_core/                 # Módulo base del problema discreto MKP
│   ├── data_loader.py        # Cargador y parser de instancias OR-Library
│   ├── problem.py            # Definición de la estructura del problema MKP
│   └── repair.py             # Algoritmo de reparación greedy y factibilidad
│
├── mcdp_core/                # Modelo del Machine Cell Design Problem (MCDP)
│   ├── data.py               # Carga de matrices y generación de instancias
│   ├── environment.py        # Evaluación, factibilidad y vecinos MCDP
│   └── results.py            # Persistencia de resultados MCDP
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
│   └── woa_abc/              # Única mezcla activa para MCDP
│       ├── cooperativo_mcdp_dtw.py # WOA + ABC con comunicación y DTW
│       ├── run_cooperative_mcdp.py # Ejecutor para MCDP
│       └── README.md               # Documentación de la mezcla
│
├── hybrid_mkp/               # Orquestación de pipelines híbridos y rotación secuencial
├── plots/                    # Utilidades modulares de visualización de métricas
└── resultados/               # Carpeta de salida con gráficos, CSVs e informes TXT/MD
```

---

## 🔀 Mezcla cooperativa WOA--ABC para MCDP

La implementación activa es
[`cooperativo_mcdp_dtw.py`](mezclas_mh/woa_abc/cooperativo_mcdp_dtw.py).
WOA y ABC trabajan sobre una población común: WOA explora, ABC refina la misma
población y ambas fases comparten inmediatamente el mejor global. Se incorpora
momentum entre mejores consecutivos y DTW reajusta los parámetros cuando
detecta estancamiento sostenido.

Para el MCDP, la variante cooperativa usa directamente asignaciones de
máquina a celda (no LB2, porque la variable tiene más de dos categorías):

```bash
python -m mezclas_mh.woa_abc.run_cooperative_mcdp --iterations 300
```

---

## 📊 Benchmark Continuo (CEC2022)

Soporta la evaluación completa sobre las **12 funciones oficiales de CEC2022**:
- **F1**: Zakharov (Unimodal) - Bias: 300
- **F2-F5**: Rosenbrock, Schaffer F6, Rastrigin, Levy (Multimodales Básicas)
- **F6-F8**: Funciones Híbridas 1, 2 y 3 (Subfunciones agrupadas)
- **F9-F12**: Funciones de Composición 1, 2, 3 y 4 (Mezclas gaussianas)

---

## 🚀 Guía de Ejecución

### 1. Ejecución del solver cooperativo MCDP

```bash
python -m mezclas_mh.woa_abc.run_cooperative_mcdp --iterations 300
```

El ejecutor trabaja sobre una instancia MCDP y muestra el costo, factibilidad,
comunicaciones entre WOA y ABC y activaciones del DTW. Se pueden ajustar sus
parámetros, por ejemplo:

```bash
python -m mezclas_mh.woa_abc.run_cooperative_mcdp \
  --instance 1 --cells 3 --capacity 6 --iterations 300 \
  --pop-size 20 --seed 42 --ddtw
```

### 2. Ejecución del Benchmark Continuo General
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
