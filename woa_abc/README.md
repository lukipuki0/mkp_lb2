# WOA--ABC continuo

Esta carpeta contiene la implementación activa del híbrido WOA--ABC para
CEC2022 y HRES2. El diseño y la fase estadística posterior están documentados
en:

[PLAN_IMPLEMENTACION_VARIANTES_DTW_CEC_HRES2.md](PLAN_IMPLEMENTACION_VARIANTES_DTW_CEC_HRES2.md)

El material MCDP fue archivado en:

```text
planes/futuro_mcdp/
```

Las siete variantes implementadas son:

```text
M0_no_dtw
M1_legacy_fire_full
M2_fire_params_only
M4_hysteresis_woa
M5_hysteresis_abc
M6_hysteresis_woa_abc
M8_four_state_woa_abc
```

La carpeta `woa_abc/` es el módulo activo y concentra el trabajo:

```text
woa_abc/
├── adaptive.py              # políticas y perfiles de las siete variantes
├── plots/                   # gráficos DTW con el estilo de DTW_optimization
│   └── dtw.py
├── config_experimentos.py   # configuración manual DTW exclusiva de WOA--ABC
├── cooperativo_cec_dtw.py   # motor WOA--ABC exclusivo CEC
├── cooperativo_hres2_dtw.py # motor WOA--ABC exclusivo HRES2
├── run_cec_variants.py      # ejecutor CEC2022
├── cec_reporting.py         # gráficos y reportes CEC por variante/función
├── run_hres2_variants.py    # ejecutor HRES2
├── result_io.py             # persistencia de resultados
└── resultados/
    ├── cec/
    └── hres2/
```

Las funciones CEC y el modelo HRES2 se importan desde sus carpetas de dominio,
pero cada motor mantiene su propia evaluación y sus propios límites. Los
ejecutores y todas las salidas pertenecen a este módulo.

## Configuración manual DTW

La ventana y los umbrales de estos experimentos se editan únicamente en
`woa_abc/config_experimentos.py`:

```python
CEC_DTW = DTWManualConfig(
    window=20, band=2, min_slope=2.0,
    plateau_max=4, patience=2, improvement_tol=1e-6,
)
HRES2_DTW = DTWManualConfig(
    window=20, band=2, min_slope=2.0,
    plateau_max=4, patience=2, improvement_tol=1e-6,
)

# Base WOA--ABC (independiente para CEC y HRES2)
CEC_MH = WOAABCManualConfig(
    woa_exploration=1.20, exploration_probability=0.65,
    step_initial=2.0, step_final=0.10, abc_limit_divisor=4,
)
```

Estas configuraciones solo afectan a los ejecutores `woa_abc`; no modifican el
benchmark anterior de `continuous_benchmark` ni los experimentos de MCDP.
Los argumentos de línea de comandos pueden sobreescribir un valor para una
prueba puntual.

`improvement_tol` evita que mejoras numéricas despreciables reinicien el
contador de estancamiento. Un `fire` aceptado se aplica una sola vez mientras
la misma condición de estancamiento permanezca activa; vuelve a estar
disponible cuando el monitor sale de esa condición.

## Ejecución

Todas las variantes sobre CEC2022:

```bash
python -m woa_abc.run_cec_variants \
  --functions all --dimension 10 --variants all \
  --runs 1 --iterations 1000 --ddtw
```

Todas las variantes sobre HRES2:

```bash
python -m woa_abc.run_hres2_variants \
  --variants all --runs 1 --iterations 1000 --ddtw
```

También hay un script `.sh` independiente para cada ejecutor:

```bash
sbatch run_woa_abc_cec.sh
sbatch run_woa_abc_hres2.sh
```

Cada script ejecuta las siete variantes en todos sus `runs`. Para una prueba
rápida se pueden cambiar los valores sin editar el archivo:

```bash
RUNS=1 ITERATIONS=10 sbatch run_woa_abc_cec.sh
```

El directorio de salida puede cambiarse con `OUTPUT_DIR`. Las salidas por
dominio permanecen separadas en `woa_abc/resultados/cec/` y
`woa_abc/resultados/hres2/`.

Los gráficos DTW se generan desde `woa_abc/plots/dtw.py` con el estilo del ZIP
de referencia: dos paneles, convergencia con segmentos de
explotación/exploración y señal `Delta` con `theta_delta`. Cada corrida guarda
`dtw_delta.png` y `dtw_delta.pdf`; cada carpeta de función guarda además
`dtw_fire.png` y `dtw_fire.pdf`.

Los triángulos rojos marcan un `fire` o cambio de modo, igual que en el ZIP.
Los círculos naranjos marcan una actualización de parámetros sin cambio de
modo; el punto se dibuja en la iteración donde el nuevo perfil comienza a
aplicarse.

En CEC se crea una subcarpeta `run_D10_dtw_<marca_de_tiempo>`,
`run_D10_ddtw_<marca_de_tiempo>`, `run_D20_dtw_<marca_de_tiempo>` o
`run_D20_ddtw_<marca_de_tiempo>`. Así la dimensión y el tipo de DTW quedan
visibles sin abrir ningún archivo.

Se puede reemplazar `all` por uno o varios nombres completos de variante. Los
resultados se guardan en `woa_abc/resultados/cec/` y
`woa_abc/resultados/hres2/`.

En CEC el orden de ejecución es `F1, F2, ..., F12`; dentro de cada función se
recorren las siete variantes y los `runs`. Cada experimento utiliza una sola
dimensión: `D=10` por defecto. Para ejecutar `D=20` como otro experimento:

```bash
DIMENSION=20 sbatch run_woa_abc_cec.sh
```

La semilla
(`seed`) controla la inicialización aleatoria de la población. Para una misma
función y corrida se reutiliza la misma semilla en todas las variantes, de
modo que la comparación sea justa. La primera corrida usa directamente la
semilla base (por defecto `42`); las siguientes, cuando se habiliten, usan
`base_seed + 1`, `base_seed + 2`, etc.

La API pública continua se mantiene compatible:

```python
from continuous_benchmark.funciones_cec2022 import get_test_functions
from woa_abc import CooperativeCECParams, ejecutar_cec_cooperativo

funcion = get_test_functions(10)[0]
resultado = ejecutar_cec_cooperativo(
    funcion,
    CooperativeCECParams(pop_size=30, iterations=100, seed=42),
    verbose=False,
)
```

El modelo HRES2 tendrá un motor separado con límites vectoriales; no se
modifica el modelo físico ni su función de decodificación.

El alcance es exclusivamente esta MH poblacional WOA--ABC. No se incorporan
algoritmos de trayectoria ni pipelines rotativos. En las siete variantes, DTW
solo selecciona perfiles de parámetros; no aborta iteraciones ni modifica,
rescata o resembra la población.

El análisis estadístico inferencial se realizará después usando los CSV por
seed que generan ambos ejecutores.

La salida CEC de una ejecución tiene esta forma:

```text
run_D10_ddtw_<fecha>/
├── configuracion.txt
├── configuracion.json
├── CEC_01_F1_.../
│   ├── M0_no_dtw/run_01/...
│   ├── M1_legacy_fire_full/run_01/...
│   ├── ...
│   ├── resultados_variantes.csv
│   ├── resumen_variantes.txt
│   ├── resumen_variantes.md
│   └── dtw_fire.png / dtw_fire.pdf
├── ... CEC_02 a CEC_12 ...
├── todos_los_runs.csv
├── resultados_variantes.csv
├── resumen_global.csv
├── resumen_global.txt
├── resumen_global.md
└── comparacion_global.png
```

Cada `run_01/` contiene directamente los historiales CSV y los gráficos de
convergencia, delta DTW y parámetros adaptativos. El módulo `woa_abc` trabaja
con una sola ejecución continua por variante y no usa epochs.

Cada carpeta de función contiene `dtw_fire.png` y `dtw_fire.pdf`, siguiendo el
formato de `DTW_optimization`: panel superior de convergencia con estilos de
exploración/explotación y panel inferior con `Δ`, `θΔ` y marcadores de
adaptación DTW. No se genera un gráfico de barras `comparacion_variantes`.

HRES2 usa la misma estructura de experimento que CEC, pero con una sola
carpeta de problema:

```text
run_HRES2_ddtw_<fecha>/
├── HRES2_H2_WPEB/
│   ├── M0_no_dtw/run_01/...
│   ├── ...
│   ├── resultados_variantes.csv
│   ├── resumen_variantes.txt
│   ├── resumen_variantes.md
│   ├── dtw_fire.png / dtw_fire.pdf
│   └── factibilidad_variantes.png
├── todos_los_runs.csv
├── resultados_variantes.csv
├── resumen_global.csv
├── resumen_global.txt
├── resumen_global.md
├── comparacion_global.png
└── configuracion.json
```

Además de la convergencia y el DTW, cada corrida HRES2 conserva la solución
decodificada, las métricas físicas (LCOE, LCOH, AGSR, H₂) y la factibilidad.
