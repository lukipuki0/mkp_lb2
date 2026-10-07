# DTW/DDTW Trajectory-Driven Collaborative Framework — MKP

A hybrid metaheuristic rotation framework with stagnation detection based on **Dynamic Time Warping (DTW/DDTW)** for the **Multidimensional Knapsack Problem (MKP)**.

The orchestrator dynamically alternates between a **population-based** solver pool (exploration) and a **trajectory-based** solver pool (exploitation). The DTW/DDTW monitor analyzes the best-so-far fitness curve against synthetic progress and stagnation reference patterns under Sakoe–Chiba band constraints, deciding when to switch algorithms.

## Metaheuristic Pools (10)

| Type | MH | Description |
|---|---|---|
| **Population** | PSO | Particle Swarm Optimization + LB2 |
| **Population** | GA | Genetic Algorithm |
| **Population** | GWO | Grey Wolf Optimizer + LB2 |
| **Population** | EHO | Elk Herd Optimizer + LB2 |
| **Population** | ACO | Ant Colony Optimization |
| **Trajectory** | SA | Simulated Annealing |
| **Trajectory** | TS | Tabu Search |
| **Trajectory** | ILS | Iterated Local Search |
| **Trajectory** | VNS | Variable Neighborhood Search |
| **Trajectory** | ABC | Artificial Bee Colony |

> **Pipeline flow**: Population-based (stagnated by DTW) → Trajectory-based → Population-based → ... until `MAX_ITERS` is exhausted.
> The best feasible solution (`x_best`) is injected as elite memory at each transition.

## Benchmark Instances

The **9 Chu & Beasley families** from the OR-Library (`mknapcb1` to `mknapcb9`) are used, with 30 instances per file (270 total instances):

| File | n (items) | m (constraints) |
|---|---|---|
| `mknapcb1` – `mknapcb3` | 100 | 5, 10, 30 |
| `mknapcb4` – `mknapcb6` | 250 | 5, 10, 30 |
| `mknapcb7` – `mknapcb9` | 500 | 5, 10, 30 |

## Project Structure

```text
mkp_lb2/
├── hybrid_mkp/               # Main hybrid pipeline framework for MKP
│   ├── orchestrator.py       # Orchestrator: DTW-driven rotation between population/trajectory pools
│   ├── batch_benchmark.py    # Sequential batch execution (31 runs × N instances)
│   ├── parallel_mkp_first_inst.py  # Parallel HPC execution (9 families concurrently)
│   ├── analisis_estadistico.py     # Shapiro-Wilk, Wilcoxon, Mann-Whitney U, Friedman, 95% CI
│   ├── resumen_estadistico_global.py # Consolidated multi-instance summary
│   ├── mh/                   # Metaheuristics for MKP
│   │   ├── pso.py            # Particle Swarm Optimization + LB2
│   │   ├── ga.py             # Genetic Algorithm
│   │   ├── gwo.py            # Grey Wolf Optimizer + LB2
│   │   ├── eho.py            # Elk Herd Optimizer + LB2
│   │   ├── aco.py            # Ant Colony Optimization
│   │   ├── abc.py            # Artificial Bee Colony
│   │   ├── sa.py             # Simulated Annealing
│   │   ├── ts.py             # Tabu Search
│   │   ├── ils.py            # Iterated Local Search
│   │   └── vns.py            # Variable Neighborhood Search
│   ├── mkp_core/             # MKP problem core
│   │   ├── data_loader.py    # OR-Library instance loader and parser
│   │   ├── problem.py        # MKPInstance structure definition
│   │   └── repair.py         # Greedy repair algorithm and feasibility check
│   └── plots/                # Modular metric visualization utilities
│       ├── convergencia.py   # Convergence curves
│       ├── dtw_delta.py      # DTW/DDTW distances over iterations
│       ├── instantaneo.py    # Instantaneous state snapshot
│       ├── solo_instantaneo.py # Simplified snapshot
│       └── switches_gantt.py # MH switch Gantt chart
│
├── mkp_core/                 # Shared base module for the MKP problem
│   ├── data_loader.py        # OR-Library instance loader
│   ├── problem.py            # MKP problem structure
│   └── repair.py             # Greedy repair
│
├── lb2/                      # Shared binarization framework (LB2)
│   ├── binarization.py       # Vectorized continuous-to-discrete mapping (L1/L2)
│   └── __init__.py           # Transfer functions (V1-V4, S1-S4)
│
├── mh/                       # Individual metaheuristics (discrete MKP domain)
│   ├── pso.py, ga.py, gwo.py, eho.py, aco.py, abc.py
│   ├── sa.py, ts.py, ils.py, vns.py
│   └── ga_operators.py, sa_neighborhood.py, ts_neighborhood.py
│
├── dtw_stagnation.py         # DTW/DDTW stagnation monitor (Sakoe-Chiba)
│
├── instancias/               # Chu & Beasley instances: mknapcb1..9
├── resultados/               # Output folder with results
│   ├── batch_mkp/            # Sequential batch results
│   └── parallel_hpc_first_inst/  # Parallel HPC results
│
├── sensitivity/              # DTW hyperparameter sensitivity analysis
│   ├── design.py             # Experimental design (Latin Hypercube)
│   ├── run.py                # Sensitivity analysis runner
│   └── analysis.py           # Results analysis
│
├── paper/                    # Scientific paper (LaTeX)
│   └── paper_final.tex       # Main manuscript
│
├── run_batch.sh              # SLURM script: sequential batch execution
├── run_parallel_hpc.sh       # SLURM script: parallel execution (9 workers)
└── ALGORITMO_CENTRAL.md      # Core algorithm documentation
```

## Requirements

```bash
pip install numpy scipy matplotlib tqdm
```

> On the Océano HPC cluster, a dedicated Conda environment named `mkp_env` is used.

## Central Configuration

### Batch benchmark parameters

Edit the constants in `hybrid_mkp/batch_benchmark.py`:

```python
MKNAPCB_SELECCION = None     # None = all 9 families (270 instances), or 1..9
INSTANCIAS_POR_FAMILIA = 30  # 30 instances per file (0 to 29)
N_RUNS       = 31            # Independent repetitions per instance
MAX_ITERS    = 1000          # Maximum iterations per run
RANDOM_SEED  = 42            # Global seed for reproducibility
```

### Parallel HPC benchmark parameters

Edit the constants in `hybrid_mkp/parallel_mkp_first_inst.py`:

```python
N_RUNS       = 31            # Independent repetitions
MAX_ITERS    = 3000          # Maximum iterations per run
RANDOM_SEED  = 42            # Global seed
MAX_WORKERS  = 9             # Concurrent workers (auto-detected from SLURM)
```

### DTW/DDTW parameters

```python
STAG_WINDOW      = 75        # Observation window (last N iterations)
STAG_BAND        = 0         # Sakoe-Chiba band (0 = no band constraint)
STAG_MIN_SLOPE   = 0.1       # Minimum slope for progress ramp
STAG_PLATEAU_MAX = 15        # Maximum plateau window for detection
STAG_PATIENCE    = 25        # Patience: consecutive confirmations before firing
```

## Basic Commands

### 1. Run batch benchmark (sequential, all instances)

```bash
python -m hybrid_mkp.batch_benchmark
```

Runs 31 independent runs per instance across all 270 MKP instances. Generates individual reports, boxplots, convergence curves, and statistical analysis.

### 2. Run parallel HPC benchmark (first instance of each family)

```bash
python -m hybrid_mkp.parallel_mkp_first_inst
```

Runs the 9 families (`mknapcb1_inst0` to `mknapcb9_inst0`) in parallel using `ProcessPoolExecutor`. Ideal for HPC with multiple cores.

### 3. Submit to SLURM (Océano HPC)

**Sequential batch:**

```bash
sbatch run_batch.sh
```

Example `run_batch.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=hybrid_mkp
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --output=./.hpc/logs/hybrid_mkp_%j.log
#SBATCH --error=./.hpc/errors/hybrid_mkp_%j.error
#SBATCH --qos=normal

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mkp_env

cd /work/lucas.erazo/mkp_lb2
export PYTHONPATH="/work/lucas.erazo/mkp_lb2:${PYTHONPATH}"

python3 -u -m hybrid_mkp.batch_benchmark
```

**Parallel HPC (9 workers):**

```bash
sbatch run_parallel_hpc.sh
```

Example `run_parallel_hpc.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=ddtw_2k
#SBATCH --partition=CPU
#SBATCH --cpus-per-task=9
#SBATCH --threads-per-core=1
#SBATCH --mem=32G
#SBATCH --output=./.hpc/logs/mkp_par9_%j.log
#SBATCH --error=./.hpc/errors/mkp_par9_%j.error
#SBATCH --qos=long

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate mkp_env

cd /work/lucas.erazo/mkp_lb2
export PYTHONPATH="/work/lucas.erazo/mkp_lb2:${PYTHONPATH}"

python3 -u -m hybrid_mkp.parallel_mkp_first_inst
```

## Statistical Analysis

The `hybrid_mkp/analisis_estadistico.py` module runs automatically upon completion of each instance:

- **Shapiro-Wilk**: Normality test per algorithm
- **Wilcoxon Signed-Rank**: Paired comparison vs. reference algorithm
- **Mann-Whitney U**: Independent comparison vs. reference
- **Friedman**: Global non-parametric ranking across all algorithms
- **95% CI**: Confidence interval for each algorithm's mean

Per-instance outputs:

```text
resultados/{batch_mkp,parallel_hpc_first_inst}/run_TIMESTAMP/
    └── mknapcbX_instY/
        ├── boxplot_comparativo.png
        ├── analisis_estadistico_pvalues.csv
        ├── analisis_estadistico_pvalues.md
        ├── runs_resultados.csv
        ├── resumen_pipeline.txt
        └── convergence, DTW delta, and switch Gantt charts...
```

## Results Structure

```text
resultados/
├── batch_mkp/
│   └── run_TIMESTAMP/
│       ├── mknapcb1_inst0/
│       │   ├── resumen_pipeline.txt
│       │   ├── resumen_pipeline_run1.txt
│       │   ├── historial_dtw_run1.csv
│       │   ├── runs_resultados.csv
│       │   ├── boxplot_comparativo.png
│       │   ├── boxplot_runs.png
│       │   ├── analisis_estadistico_pvalues.csv
│       │   └── convergencia_runN.png, dtw_delta_runN.png, switches_runN.png
│       ├── mknapcb1_inst1/ ... mknapcb9_inst29/
│       ├── resumen_batch.txt
│       ├── resumen_batch.csv
│       └── resumen_batch.md
│
└── parallel_hpc_first_inst/
    └── run_TIMESTAMP/
        ├── mknapcb1_inst0/ ... mknapcb9_inst0/
        ├── resumen_batch.txt
        ├── resumen_batch.csv
        └── resumen_batch.md
```

Each instance report contains:

- **runs_resultados.csv**: Final fitness of each run (31 values)
- **resumen_pipeline.txt**: Configuration, best fitness, gap to known optimum, switches performed
- **historial_dtw_runN.csv**: DTW/DDTW values per iteration
- **boxplot_comparativo.png**: Boxplot comparing Pipeline vs. Standalone baselines
- **Convergence plots**: Best-so-far curve with MH switch annotations
- **Gantt chart**: Temporal visualization of which MH was active in each segment

## Usage Notes

- All experiments use **31 independent runs** with a fixed global seed (`RANDOM_SEED = 42`) for reproducibility. Each run generates its own sub-seed.
- The pipeline automatically compares the **Hybrid DTW Pipeline** against each MH executed **standalone** (same iterations and runs).
- The green horizontal line in boxplots represents the **known optimum** of each Chu & Beasley instance.
- Environment variables `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, etc. are critical on HPC to avoid destructive CPU contention between workers.
- Statistical analysis pairs results by **seed/run** (same order), so all algorithms must use the same number of runs.
