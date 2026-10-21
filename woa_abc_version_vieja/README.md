
borr

# New WOA--ABC con adaptación DTW/DDTW

Esta carpeta contiene la implementación nueva e independiente de WOA--ABC.
La arquitectura y las ecuaciones del sensor se basan en
`DTW_optimization-main`, pero todo el código necesario del monitor quedó
copiado y adaptado dentro de `woa_abc_version_vieja/dtw/`. Por eso, borrar después la
carpeta de referencia no rompe esta implementación.

No se importa código de la implementación anterior `woa_abc/`, del monitor
global `dtw_stagnation.py` ni de MCDP. CEC2022, HRES2 y MKP tienen adaptadores,
motores cuando corresponde y ejecutores separados.

## Qué hace el algoritmo

CEC/HRES2 usan una única población continua. MKP mantiene sincronizadas una
posición latente continua y una solución binaria factible por individuo:

1. WOA propone y acepta mejoras.
2. ABC ejecuta las fases empleada, observadora y scout.
3. Se registra el mejor valor global (costo o beneficio).
4. DTW/DDTW observa la ventana reciente de ese mejor valor.
5. La estrategia calcula el perfil que se aplicará en la iteración siguiente.

En MKP, cada propuesta se convierte a bits con las dos funciones LB2, se repara
por densidad normalizada y se conserva la mejor candidata. Un pulido acotado
de intercambios hasta `3-out/3-in` se aplica al élite. Sigue siendo una sola
WOA--ABC: no se cambia a otra MH ni se construye una trayectoria.

DTW no mueve soluciones, no inyecta individuos, no reinicia la población y no
altera la función objetivo. Solo modifica parámetros nativos de WOA--ABC:

- WOA: parámetro `a`;
- ABC: amplitud `phi`, guía al mejor, vecindad vectorial y límite scout;
- cooperación: paso decreciente y momentum entre mejoras globales.

El gráfico de parámetros separa siempre el calendario base, el factor
adaptativo y el valor efectivo. Así no se confunde la caída natural de `a` con
un cambio causado por DTW.

## Las siete variantes

| Variante                  | Base                  | Decisión                                      |
| ------------------------- | --------------------- | ---------------------------------------------- |
| `M0_no_dtw`             | Vanilla               | parámetros base; el monitor no se crea        |
| `M1_fire_d2`            | A3 Binary-Simple      | mismo fire confirmado que M2; el nombre D2 es histórico |
| `M2_fire_3cond`         | A4 Binary-Complex     | fire confirmado: meseta, distancias y patience |
| `M4_d2_continuous`      | B3 Continuous-Simple  | intensidad continua basada en D2; no es un fire |
| `M5_sigmoid_delta`      | B1 Continuous-Complex | sigmoid de `delta/theta_delta`; `theta_delta` nunca es negativo |
| `M6_hysteresis_woa_abc` | extensión            | histéresis aplicada conjuntamente a WOA y ABC |
| `M8_four_state_woa_abc` | extensión            | cuatro niveles coordinados de parámetros      |

M1, M2, M4 y M5 conservan los identificadores y alias CLI de las cuatro
familias de referencia. M6 y M8 siguen siendo extensiones. El fire discreto
de M1 ya no es `D2 < theta_c`: ese gate solo de D2 se sustituye a propósito
por el mismo evento confirmado de M2. El solapamiento M1/M2 es intencional.

## Fire confirmado y rearme

Un fire es un evento, no una condición repetida en cada iteración.

1. La ventana está llena.
2. Se cumplen juntas la meseta (`no_improve_len >= plateau_max`), la cercanía
   a la constante (`D2 <= theta_c`) y la separación de la rampa
   (`D1 >= theta_r` o `delta >= theta_delta`).
3. Esa tripleta se sostiene `patience` observaciones listas.

`theta_c` empieza en `0.1 * window`; con adaptación y al menos diez ventanas
listas usa el percentil `p_low` del historial de D2. No se eleva al D2 de la
ventana actual: una distancia grande debe poder bloquear el fire aunque haya
meseta y se cumpla la condición de rampa.

Las tres condiciones más patience confirman un solo evento. El monitor guarda
ese diagnóstico y después vacía la ventana, `no_improve_len`, la racha y los
historiales D1/D2/delta. La observación siguiente queda en `n=1`, sin fire.
No hay otro fire hasta llenar una ventana nueva y volver a cumplir meseta y
patience. Con ventana 5, meseta 2 y patience 2 sobre una serie plana, los
índices de detección son 5, 11, 17, …

`raw_fire` es el alias de esquema de ese evento. No significa “D2 bajo” ni
“intensidad alta”. M0 no crea monitor y no escribe distancias ni fires.
M4, M5, M6 y M8 conservan la fórmula de intensidad; marcan `raw_fire` solo
en el evento común. Durante el rearme el monitor no está listo, así que esas
variantes vuelven al perfil base hasta llenar la ventana nueva. No es otro
gate: la entrada del sensor se vació. El contador `fire_count` suma cada evento una vez, aunque el modo no
cambie. Un fire sin cambio de parámetros queda en `eventos_control` y no
incrementa `parameter_update_count`. Si el evento cae en la última iteración,
o la parada temprana no deja otro ciclo, `applied_iteration` es nulo: no se
simula una aplicación que no ocurrió.

El rearme no reinicia población, trials ABC, archivo, iteración global,
calendario LB2 ni el reloj base de parámetros. Solo cambia el estado del
monitor. La decisión detectada se aplica en el ciclo siguiente, como antes.

En `dtw_adaptation` el dibujo es el cuerpo de `HRES2-H2/plots/dtw_delta.py`,
no un restyle. La curva es el delta real `D1 - D2`. Las líneas verticales son
los cortes de segmento de esa fuente, no cambios de metaheurística.

La frontera en la iteración 0 abre el primer segmento. No es un fire y no
suma `fire_count`. Los cortes siguientes son los índices exactos de
`raw_fire`. Con 50 segmentos o más la fuente no dibuja esas líneas; esta
copia conserva ese límite. Una intensidad o una condición pegada no es un
fire y no abre un segmento.

## Configuración DTW manual

Los valores predeterminados de CEC son los del repositorio de referencia:

```text
window=20, band=2, min_slope=2.0, plateau_max=4, patience=2,
DDTW=True, adaptive_thresholds=True, p_low=30, p_high=70
```

Se pueden cambiar por ejecución. Por ejemplo:

```bash
python3 -m woa_abc_version_vieja.runners.run_cec \
  --functions F1 --dimension 10 --variants all --runs 1 \
  --window 30 --band 3 --plateau-max 6 --patience 3
```

Use `--no-ddtw` para DTW normal y `--no-adaptive-thresholds` para umbrales
fijos. La carpeta de resultados incluye `ddtw` o `dtw` en su nombre y guarda
la configuración completa en JSON y TXT.

HRES2 usa por defecto `min_slope=0`, que activa una rampa automática acorde a
la escala de LCOE. HRES2 no tiene un óptimo analítico certificado: se reporta
el mejor LCOE factible encontrado y no se dibuja una línea de “óptimo”. Los
demás campos se mantienen iguales y también pueden configurarse por CLI.

MKP usa los valores de `hybrid_mkp/batch_benchmark.py`, salvo la ventana
ampliada a `window=70`: `band=0` (banda automática efectiva 7),
`min_slope=0.1`, `plateau_max=15`,
`patience=8`, DDTW y umbrales adaptativos activados, percentiles 30/70.
Se pueden sobreescribir por CLI. CEC/HRES2 conservan sus valores anteriores.
Este cambio solo afecta ejecuciones nuevas: no modifica resultados guardados
ni la regla de fire confirmado y rearme.

MKP entrega `best_profit` directamente al monitor porque ambos son de
maximización. CEC/HRES2 continúan entregando `-best_cost`. El calendario LB2
es independiente de DTW; las siete variantes solo adaptan WOA--ABC.

## Ejecución predeterminada: 31 seeds y análisis estadístico

CEC completo, doce funciones, dimensión 10, siete variantes y 31 seeds:

```bash
./woa_abc_version_vieja/run_cec.sh
```

HRES2, siete variantes y 31 seeds:

```bash
./woa_abc_version_vieja/run_hres2.sh
```

CEC y HRES2 distribuyen cada corrida independiente entre los CPU asignados
por Slurm. Por defecto CEC solicita 12 workers y HRES2 7; ambos activan
explícitamente `mkp_env` y limitan NumPy/OpenBLAS/MKL a un hilo por worker.
En ejecución local se puede reducir el paralelismo con `WORKERS=4`.

MKP: las tres primeras instancias de cada uno de los archivos `mknapcb1` a
`mknapcb9`, 31 seeds y siete variantes. `mknapcb1/4/7` corresponden a 100
ítems, `mknapcb2/5/8` a 250 y `mknapcb3/6/9` a 500. La selección
predeterminada usa índices 0–2: 27 problemas (9 por grupo).

```bash
./woa_abc_version_vieja/run_mkp.sh
```

El runner MKP sigue el esquema de `run_parallel_hpc.sh`: solicita nueve CPU,
32 GB, QoS `long`, activa `mkp_env`, fija un hilo BLAS/NumPy por proceso y
ejecuta una instancia completa por worker. Los resultados se separan en
`grupo_100_items`, `grupo_250_items` y `grupo_500_items`. Slurm:

```bash
sbatch woa_abc_version_vieja/run_mkp.sh
```

En ejecución local el script también usa nueve workers. El archivo quedó fijado
para ejecutar la campaña completa predeterminada sin declarar cada parámetro en
la llamada; para pruebas pequeñas se puede invocar directamente el módulo Python
con sus opciones CLI.

El presupuesto MKP predeterminado es de 3 000 iteraciones, con parada temprana
si alcanza el mejor valor conocido. Para cambiar la selección desde el script,
se puede indicar `INSTANCES`; por ejemplo, `INSTANCES=0,1,2` conserva las tres
primeras instancias por archivo.

Verificación corta de una sola instancia/variante:

```bash
python3 -m woa_abc_version_vieja.runners.run_mkp --files 1 --instances 0 \
  --variants M0 --runs 1 --iterations 50 --workers 1
```

Los scripts CEC y HRES2 aceptan variables de entorno. Ejemplos:

```bash
RUNS=1 FUNCTIONS="F1 F12" VARIANTS="M0 M4 M8" ITERATIONS=100 ./woa_abc_version_vieja/run_cec.sh
RUNS=1 VARIANTS="M0 M6" ITERATIONS=100 ./woa_abc_version_vieja/run_hres2.sh
RUNS=1 WINDOW=30 BAND=3 DTW_MODE=dtw ./woa_abc_version_vieja/run_cec.sh
```

En los `.sh` se pueden ajustar `WINDOW`, `BAND`, `MIN_SLOPE`, `PLATEAU_MAX`,
`PATIENCE`, `DTW_MODE=dtw|ddtw`, `ADAPTIVE_THRESHOLDS=true|false`,
`ABC_GUIDE_STRENGTH`, `ABC_VECTOR_PROBABILITY`, `ABC_VECTOR_SCALE`,
`STEP_INITIAL`, `STEP_FINAL`, `MOMENTUM_FACTOR`, `ABC_LIMIT_DIVISOR` y
`MIN_ABC_LIMIT`. Para la campaña estadística aceptan `STATISTICAL_ANALYSIS`,
`REFERENCE_VARIANT`, `ALPHA` y `SAVE_ALL_RUN_ARTIFACTS`. La vecindad vectorial usa por defecto probabilidad 0.35 y
escala 0.10 del rango de cada variable.

Los ejecutores CEC/HRES2 escriben un checkpoint CSV atómico después de cada
corrida. Una campaña interrumpida se reanuda sobre la misma carpeta sin repetir
las parejas variante--seed ya guardadas:

```bash
RESUME_DIR=/ruta/a/run_D10_ddtw_... sbatch woa_abc_version_vieja/run_cec.sh
RESUME_DIR=/ruta/a/run_HRES2_ddtw_... sbatch woa_abc_version_vieja/run_hres2.sh
```

La configuración se valida antes de reanudar; no se permite mezclar distinto
número de seeds, iteraciones, variantes o parámetros DTW en la misma campaña.

Para MKP también están expuestas por CLI las opciones `--g1-initial/final`,
`--g2-initial/final`, `--g3-initial/final`, `--local-search-start`,
`--local-search-interval`, `--local-search-passes`, `--local-search-ejections`,
`--local-search-depth` y `--local-search-elites`.
`--abc-limit-population-factor` controla el límite
scout binario; se escala con la población porque cada intento cambia un vector
completo, no una sola coordenada. Si se activa, el pulido comienza por defecto
en la iteración 50, después del calentamiento DTW de 20 observaciones.
`--workers` controla los procesos paralelos de MKP. La búsqueda local está
desactivada de forma predeterminada (`--local-search-interval 0`); puede
habilitarse indicando un intervalo positivo.

Para CEC D=20 se crea un experimento separado:

```bash
DIMENSION=20 ./woa_abc_version_vieja/run_cec.sh
```

## Campaña estadística

Los scripts usan de forma predeterminada 31 semillas emparejadas y
`M0_no_dtw` como control. Por tanto, basta ejecutar:

```bash
./woa_abc_version_vieja/run_cec.sh
./woa_abc_version_vieja/run_hres2.sh
./woa_abc_version_vieja/run_mkp.sh
```

Se calculan Shapiro--Wilk, Wilcoxon bilateral pareado, corrección de Holm,
Mann--Whitney U, Friedman, mean ranks, correlación biserial de rangos e IC 95 %.
La inferencia principal es Wilcoxon pareado por seed con Holm; Mann--Whitney se
mantiene para conservar compatibilidad con los otros benchmarks.

En esta modalidad se guardan en CSV/JSON los valores de todas las corridas,
pero los historiales y gráficos detallados solo para `mejor_run` de cada
variante. Use `--save-all-run-artifacts` si necesita conservar el detalle de
las 31 corridas. Una campaña terminada también se puede volver a analizar sin
repetir las optimizaciones:

```bash
python3 -m woa_abc_version_vieja.runners.analyze_results RUTA_DE_LA_CAMPANA
```

## Resultados

Cada ejecución crea una carpeta única; repetir un comando nunca intenta
sobrescribir la carpeta anterior. Los gráficos de corridas y reportes
estadísticos se guardan solo como PDF. Este cambio afecta salidas nuevas:
los PNG de ejecuciones anteriores no se borran ni se regeneran.

En MKP, `parameters.pdf` muestra cuatro paneles: `a` base y efectivo de WOA,
amplitud `phi` efectiva de ABC, límite scout efectivo y fuerza de guía al
élite. El eje horizontal es la iteración; no son curvas de fitness. El
calendario de binarización G1/G2/G3 se guarda aparte en `lb2_parameters.pdf`.

Estructura de resultados:

```text
woa_abc_version_vieja/resultados/cec/run_D10_ddtw_<timestamp>/
├── configuracion.json
├── configuracion.txt
├── CEC_01_<nombre>/
│   ├── M0_no_dtw/run_01/       # una sola seed
│   │   ├── resultado.json
│   │   ├── resumen.txt
│   │   ├── historial_convergencia.csv
│   │   ├── historial_dtw.csv
│   │   ├── historial_parametros.csv
│   │   ├── eventos_control.json
│   │   ├── convergence.pdf
│   │   ├── dtw_adaptation.pdf
│   │   └── parameters.pdf
│   ├── M0_no_dtw/mejor_run/    # campaña estadística
│   ├── resultados.csv y resumen.csv
│   ├── analisis_estadistico_pvalues.csv/.json/.md
│   └── boxplot_estadistico.pdf
├── todos_los_runs.csv
├── todos_los_runs.json
├── analisis_estadistico_global.csv/.json/.md
├── analisis_estadistico_por_problema.csv
└── ranking_estadistico_global.pdf
```

HRES2 conserva la misma estructura básica y agrega la solución decodificada,
las métricas físicas y `metricas_hres2.csv`.

MKP agrupa las instancias por cantidad de ítems y crea una rama independiente
por familia e instancia:

```text
woa_abc_version_vieja/resultados/mkp/run_MKP_ddtw_<timestamp>/
├── configuracion.json y configuracion.txt
├── grupo_100_items/
│   ├── mknapcb1/inst_00 ... inst_02/ (junto a mknapcb4 y mknapcb7)
│   ├── todos_los_runs.csv y todos_los_runs.json
│   ├── resumen_grupo.md
│   └── análisis y ranking estadístico del grupo
├── grupo_250_items/ (misma estructura; mknapcb2, mknapcb5, mknapcb8)
├── grupo_500_items/ (misma estructura; mknapcb3, mknapcb6, mknapcb9)
├── todos_los_runs.csv y todos_los_runs.json
├── análisis y ranking estadístico global
└── resumen_global.md
```

Dentro de cada `inst_<índice>/` se guardan las siete variantes, sus runs,
gráficos de la mejor run, resumen y análisis pareado.

Cada resultado MKP informa beneficio, mejor conocido, gaps, factibilidad,
holguras, bits e ítems seleccionados. Puede detenerse al alcanzar mkcbres.
WOA--ABC es heurística: una seed no garantiza todos los óptimos. La campaña de
31 seeds informa la tasa observada y permite comparar las variantes.

`dtw_adaptation.pdf` copia el panel de `HRES2-H2/plots/dtw_delta.py`.
CEC y MKP usan el mismo cuerpo. La salida nueva sigue siendo solo PDF.

| Elemento | Lectura |
| --- | --- |
| Curva | delta real `D1 - D2`; el original une el punto válido anterior, también a través del calentamiento |
| Título | `WOA-ABC DTW Delta per Iteration  [+ = stagnation \| - = active improvement]`. El signo describe el patrón respecto de la rampa y la constante; no es el booleano de fire |
| Ejes | `Accumulated Iterations` y `DTW Delta`. No hay línea en cero |
| Segmentos | tramos de monitor/control entre fires confirmados. Todos llevan el nombre y el color de la variante |
| Frontera 0 | línea vertical del segmento inicial. No es un fire y no incrementa `fire_count` |
| Fires | fronteras siguientes en el índice exacto de `raw_fire`. No hay leyenda `Fire DTW` ni etiquetas PSO/SA |
| Visibilidad | las verticales se dibujan solo si hay menos de 50 segmentos, como la fuente |
| M0 | excepción necesaria: panel vacío `DTW desactivado`, sin curva, sin frontera y sin evento inventado |

No hay eje de intensidad, curva de D2, umbral ni perfil. La convergencia,
los parámetros y LB2 siguen en sus PDF. El calentamiento no se rellena con
cero. M4 y M1 también dibujan `D1 - D2`, no D2 ni el signo invertido.

## Diagnóstico temporal de una corrida

`temporal_diagnostics.pdf` se agrega en la carpeta de la corrida MKP, junto a
los cuatro PDF que ya existían. M0 no lo genera.

| Pieza | Decisión |
| --- | --- |
| Cuándo | solo si `save_mkp_run_plots` recibe la configuración real de esa corrida |
| M0 | no hay figura ni series inventadas |
| Datos | convergencia, estado aplicado y `a`/`phi` efectivos guardados |
| Ventana | `configuration["algorithm"]["dtw"]`; no se sustituye por W=200 ni por otro default |
| Archivo | únicamente `temporal_diagnostics.pdf` |

Sin ese argumento, el guardado sigue produciendo solo `convergence.pdf`,
`dtw_adaptation.pdf`, `parameters.pdf` y `lb2_parameters.pdf`.

Los resúmenes rotulan `fire_count` como “Fires confirmados"; las transiciones
de modo se cuentan por separado.

El valor predeterminado es `RUNS=31` y activa el protocolo estadístico. Para
una verificación rápida sin inferencia use `RUNS=1`.

## Pruebas

```bash
python3 -m unittest discover -s woa_abc_version_vieja/tests -v
```
