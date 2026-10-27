# DTW Optimization — MKP

Comparación de adaptaciones DTW (Dynamic Time Warping) para la optimización del **Multidimensional Knapsack Problem (MKP)** usando 4 metaheurísticas binarias (PSO, GA, GWO, DE).

## Versiones del estudio (6)

| # | Versión | Tipo | Estrategia | Descripción |
|---|---|---|---|---|
| 1 | **Vanilla-Explotación** | Baseline | — | MHs forzadas a modo exploit puro durante toda la ejecución |
| 2 | **Vanilla-Exploración** | Baseline | — | MHs forzadas a modo explore puro durante toda la ejecución |
| 3 | **Binary-Simple** | DTW Binario | A3 — Fire D₂ | Decisión booleana: `fire = D₂ ≤ θ_c`. La pregunta más directa posible. |
| 4 | **Binary-Complex** | DTW Binario | A4 — 3 condiciones + patience | Baseline DTW: plateau + D₂ + D₁/Δ + confirmación temporal. Máxima robustez. |
| 5 | **Continuous-Simple** | DTW Continuo | B3 — D₂ directo | Intensidad continua: `intensity = 1 − clip(D₂/(θ_c × scale))`. El regulador más simple. |
| 6 | **Continuous-Complex** | DTW Continuo | B1 — Sigmoid Δ | Intensidad sigmoidal sobre delta normalizado. Respuesta no lineal con zona muerta. |

> **Documentación conceptual**: `contexto/oficial/` contiene documentos detallados de cada estrategia, cada metaheurística, y los fundamentos del DTW.

## Estructura del proyecto

```text
DTW_optimization/
├── mkp_common/              # Código compartido: MHs, DTW, runner, estadísticas
│   ├── mh/                  # Metaheurísticas: BinaryPSO, GA, BinaryGWO, BinaryDE
│   ├── config.py            # Configuración central (población, iteraciones, epochs, DTW)
│   ├── base.py              # Interfaz BaseMH + adapt_continuous
│   ├── monitor.py           # StagnationMonitor: DTW + 3 condiciones + umbrales adaptativos
│   ├── problem.py           # Carga de instancias OR-Library + reparación greedy
│   ├── runner.py            # Loop genérico con fire_fn inyectable (Strategy pattern)
│   ├── stats.py             # Wilcoxon + Holm-Bonferroni + tablas
│   └── results.py           # Guardado/carga de resultados JSON
├── vanilla_explotacion/     # Versión 1: Baseline — exploit puro
├── vanilla_exploracion/     # Versión 2: Baseline — explore puro
├── binary_simple/           # Versión 3: A3 — Fire D₂
├── binary_complex/          # Versión 4: A4 — 3 condiciones + patience
├── continuous_simple/       # Versión 5: B3 — D₂ directo continuo
├── continuous_complex/      # Versión 6: B1 — Sigmoid Δ
├── contexto/                # Documentación y referencia
│   ├── oficial/             # Documentos conceptuales (paper)
│   │   ├── 01_binary_simple_fire_d2.md
│   │   ├── 02_binary_complex_fire_binario.md
│   │   ├── 03_continuous_simple_b3_d2.md
│   │   ├── 04_continuous_complex_b1_sigmoid.md
│   │   ├── 09_dtw_fundamentos.md
│   │   └── mhs/             # Documentos de metaheurísticas
│   │       ├── 05_mh_pso.md
│   │       ├── 06_mh_ga.md
│   │       ├── 07_mh_gwo.md
│   │       └── 08_mh_de.md
│   ├── info_dtw/            # Explicaciones del DTW como monitor
│   ├── estrategias_dtw/     # Análisis de estrategias de adaptación
│   ├── params_instancias/   # Parámetros de referencia e instancias MKP
│   └── miscelaneo/          # Visualizaciones, DE, enfoque general
├── instances/               # Instancias Chu & Beasley (OR-Library): mknapcb1..9
├── analisis/                # Análisis estadístico y boxplots
├── results/                 # Resultados por estrategia e instancia
├── run_all.py               # Ejecución secuencial de todas las estrategias
├── run_all_hpc.py           # Ejecución paralela para HPC/SLURM
└── run_dtw.sh               # Script de envío a SLURM
```

## Requisitos

```bash
pip install numpy scipy matplotlib tqdm
```

> En el HPC Océano se usa un entorno Conda llamado `DTW_optimization`.

## Configuración central

Edita `mkp_common/config.py` para cambiar la instancia, población, iteraciones y epochs:

```python
RUTA_INSTANCIA = "instances/mknapcb4.txt"   # instancia por defecto
INDICE_INSTANCIA = 0                         # índice dentro del archivo
NUM_PARTICULAS = 20
NUM_ITERACIONES = 200
EPOCHS = 20
```

También puedes sobreescribirlo por variable de entorno antes de ejecutar:

```bash
export MKP_INSTANCIA=instances/mknapcb1.txt
export MKP_INDICE=2
```

## Comandos básicos

### 1. Ejecutar una sola estrategia

```bash
# Líneas base vanilla (sin DTW)
python -m vanilla_explotacion.resultados
python -m vanilla_exploracion.resultados

# Estrategias DTW — Binary
python -m binary_simple.resultados
python -m binary_complex.resultados

# Estrategias DTW — Continuous
python -m continuous_simple.resultados
python -m continuous_complex.resultados
```

Cada comando guarda resultados en `results/{strategy}/todos/{instancia}_{indice}/comparacion_mhs_{timestamp}/`.

### 2. Ejecutar todas las estrategias secuencialmente

```bash
python run_all.py
python run_all.py --instancia instances/mknapcb1.txt
python run_all.py --instancia instances/mknapcb1.txt --indice 2
```

### 3. Ejecutar en HPC con paralelismo total (estrategia × MH × epoch)

```bash
python run_all_hpc.py
python run_all_hpc.py --instancia instances/mknapcb1.txt --indice 0
python run_all_hpc.py --cpus 40 --epochs 30
```

### 4. Enviar a SLURM (HPC Océano)

```bash
sbatch run_dtw.sh
```

Ejemplo de `run_dtw.sh`:

```bash
#!/bin/bash
#SBATCH --job-name=dtw_mkp
#SBATCH --partition=CPU
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --time=24:00:00
#SBATCH --output=dtw_%j.out

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate DTW_optimization

cd /work/jose.villamayor/DTW_optimization
python run_all_hpc.py --cpus $SLURM_CPUS_PER_TASK
```

## Análisis estadístico

Una vez generados los resultados, compara todas las versiones contra Vanilla-Explotación con Wilcoxon + Holm-Bonferroni:

```bash
# Usa los resultados más recientes de cualquier instancia
python -m analisis.estadistico

# Filtra por instancia específica
python -m analisis.estadistico --instancia instances/mknapcb4.txt
python -m analisis.estadistico --instancia instances/mknapcb4.txt --indice 0

# Test one-sided: versión > vanilla
python -m analisis.estadistico --one-sided
```

Salidas en `results/estadistico/{instancia}_{indice}/`:

- `tabla_estadistica_{timestamp}.txt`
- `tabla_matematica_{timestamp}.txt`
- `comparacion_{timestamp}.png`  ← boxplot con línea verde punteada del óptimo conocido

## Estructura de resultados

```text
results/
├── vanilla_explotacion/
├── vanilla_exploracion/
├── binary_simple/
├── binary_complex/
├── continuous_simple/
├── continuous_complex/
└── estadistico/
    └── mknapcb4_0/
        ├── tabla_estadistica_*.txt
        ├── tabla_matematica_*.txt
        └── comparacion_*.png
```

Cada JSON contiene:

- `fitness`: lista de fitness por epoch
- `fire_counts`: cantidad de fires DTW por epoch (0 en vanilla)
- `tiempos`: tiempo de ejecución por epoch
- `optimo_conocido`: valor óptimo teórico de la instancia
- `stats`: mejor, promedio, peor, std y gap al óptimo
- `info`: metadatos del experimento

## Notas de uso

- El análisis estadístico empareja resultados por **semilla/epoch** (mismo orden), por lo que todos los experimentos deben usar el mismo número de epochs.
- La línea horizontal verde en el boxplot representa el **óptimo conocido** de la instancia.
- Si una metaheurística repite exactamente el mismo fitness en muchas epochs (por ejemplo GA + `binary_complex` en instancias difíciles), eso indica **convergencia prematura**: la población perdió diversidad y el operador de reparación determinístico `reparar()` no genera suficiente variación. Esto no es un bug del código, sino un comportamiento conocido del GA binario con reparación greedy sobre MKP. Para mitigarlo se puede aumentar `NUM_PARTICULAS`, aumentar la tasa de mutación en modo explore, o probar una inicialización más diversa.
