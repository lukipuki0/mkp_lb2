# Selección adaptativa sobre el pipeline híbrido MKP

Esta variante ejecuta **el mismo `hybrid_mkp.orchestrator.ejecutar_pipeline()`** que el sistema original. Mantiene los operadores, inicialización, alternancia poblacional/trayectoria, transferencia de la mejor solución y reglas DTW. La única diferencia de búsqueda es la política que elige la próxima MH.

**La unidad es iteraciones nativas, no evaluaciones.** Se retiraron el contador de evaluaciones y sus adaptadores. Por pedido posterior del usuario, se eliminó `instantaneous_convergence.png` y se agregó un diagnóstico temporal de cuatro paneles basado solo en los registros reales.

## Campaña agrupada: 27 casos y 31 corridas por caso

El ejecutor de campaña es `adaptive_mh_selection.campaign`. Su configuración predeterminada cubre las nueve familias `mknapcb1` a `mknapcb9`, índices **0, 1 y 2**, con **31 semillas (42–72)** por caso: **837 corridas** para una condición `adaptive_ddtw`.

La fuente de esos valores es `config.py` (`RunConfig` y `default_monitor`), no los números históricos de `rotating_benchmark`. Hoy: DDTW activo, ventana 70, banda 7, pendiente 2, meseta 25, paciencia 15, percentiles 30/70, umbrales adaptativos y tolerancia 0. Un flag de la CLI pisa solo su campo. `--modes dtw` sigue disponible y escribe `adaptive_dtw`.

Los grupos se obtienen del tamaño real de la instancia, no de una etiqueta inventada:

| Grupo | Familias | Restricciones |
|---|---|---|
| `items_100` | `mknapcb1`, `mknapcb4`, `mknapcb7` | 5, 10, 30 |
| `items_250` | `mknapcb2`, `mknapcb5`, `mknapcb8` | 5, 10, 30 |
| `items_500` | `mknapcb3`, `mknapcb6`, `mknapcb9` | 5, 10, 30 |

El tamaño intermedio es **250**, no 240. Cada familia aporta tres índices y cada grupo reúne nueve casos.

```bash
# Resolver y validar los 27 casos / 837 corridas, sin ejecutar ni escribir resultados.
python -B -m adaptive_mh_selection.campaign --dry-run

# Lanzar manualmente la campaña completa, una vez revisado el protocolo.
# Usa los defaults de config.py (DDTW, banda 7, percentiles 30/70). No se ejecuta al verificar el código.
python -B -m adaptive_mh_selection.campaign --iterations 1000 --runs 31 --seed 42

# Ver las opciones para seleccionar familias, índices, modos y políticas.
python -B -m adaptive_mh_selection.campaign --help
```

### Retención para análisis estadístico

Por caso y condición se conservan **todas las 31 filas numéricas**, pero **solo el ganador** tiene solución, historiales y gráficos completos:

```text
<campaña>/
  protocol.json
  environment.json
  estado_ejecucion.csv
  resumen_global.csv
  resumen_global.txt
  items_100/
    resumen_grupo.csv
    resumen_grupo.txt
    mknapcb1/
      inst_00/
        adaptive_ddtw/
          runs_resultados.csv       # las 31 mediciones, no solo el mejor fitness
          resumen_pipeline.txt
          boxplot_runs.png
          best_run_XX/              # ordinal original del ganador
            config.json
            summary.json
            native_result.json
            resumen_pipeline_best_run.txt
            historial_dtw.csv
            historial_dtw_detalle.txt
            ... gráficos nativos y diagnóstico temporal ...
      inst_01/
      inst_02/
    mknapcb4/
    mknapcb7/
  items_250/
    ... mknapcb2, mknapcb5, mknapcb8 ...
  items_500/
    ... mknapcb3, mknapcb6, mknapcb9 ...
```

La mejor corrida maximiza fitness; en empate gana la primera. Se conserva su número original, incluso si una ejecución anterior falló. No se crean directorios `run_XX` con trazas de los perdedores ni se acumulan los 31 historiales completos en memoria.

Cada CSV queda identificado por su ruta de tamaño/familia/índice/condición. Cada fila incluye semilla, fitness, gap si existe referencia, tiempo, iteraciones solicitadas/reales/sobrepaso, bloques/transiciones y estado/error. Los fallos no se convierten en fitness cero ni se ocultan: un caso con 30 éxitos y un fallo no se anuncia como 31 corridas válidas. Si todos fallan, no se fabrica un ganador. Un error de exportación también produce estado de fallo y salida no cero.

Las métricas se actualizan tras cada intento y el estado registra los casos pendientes, activos y terminados. Los resúmenes de grupo/global tienen una fila por caso y condición: **no promedian fitness crudo entre instancias diferentes**. La desviación estándar es muestral (`ddof=1`) y queda ausente si hay menos de dos éxitos. Las referencias desconocidas permanecen ausentes; la del índice 0 nunca se copia a los índices 1 o 2.

Si se seleccionan otras políticas o detectores, se repiten las mismas semillas por condición, útil para comparaciones pareadas posteriores. El número de corridas aumenta con el número de condiciones. El lector de `analysis` expone las mediciones ligeras de todas las corridas; guardar solo el ganador numérico impediría un análisis estadístico correcto. Las pruebas inferenciales son una etapa posterior, no una conclusión automática de esta campaña.

Las carpetas existentes no se sobrescriben. Esta primera versión es secuencial y no ofrece reanudación ni reintentos automáticos; en una interrupción, los registros parciales deben revisarse y no se consideran una campaña completa. El ganador completo se exporta al terminar el caso, no después de cada intento.

El lector de datos es de solo lectura y conserva todas las filas, incluyendo tiempo, bloques, transiciones y valores ausentes de los fallos:

```bash
python -B -m adaptive_mh_selection.analysis --campaign resultados/adaptive_mh_selection/<carpeta_de_campaña>
```

El protocolo de campaña usa schema 3; el `summary.json` del ganador conserva schema 2 para seguir siendo compatible con el redibujado del diagnóstico temporal. Los resúmenes TXT por caso también se guardan si el caso falla o queda interrumpido, sin anunciarlo como terminado.

### Verificación de esta implementación

- `--dry-run` comprobado sobre los nueve archivos reales: **27 casos, 9 por grupo y 837 corridas previstas**, semillas 42–72, sin crear salida ni ejecutar solvers.
- **41 pruebas del paquete y 41 generales pasaron**. Incluyen 31 filas conservadas, ganador `best_run_17` pese a un fallo previo, empates, casos sin éxitos, interrupción, errores de exportación y lectura de todas las mediciones.
- Piloto real reducido: familias 1/2/3, índice 0, dos semillas por caso, **una iteración con cap explícito de una época**. Terminó con seis éxitos, una sola carpeta de ganador por tamaño y todas las métricas/reportes/gráficos esperados. No prueba eficacia ni comportamiento de un estudio de 1000 iteraciones.
- Piloto final: `resultados/adaptive_mh_selection/campaign_20261022_184854_360468/`. Se verificaron factibilidad, objetivo del ganador, CSV de dos filas por caso, ausencia de historiales de perdedores y coincidencia de hashes de fuentes/instancias. El lector devuelve seis mediciones, no solo tres ganadores.
- **No se ejecutaron las 837 corridas completas**. El comando completo anterior es para lanzarlas explícitamente después de revisar el protocolo.

## HPC: tres arrays independientes (100, 250 y 500 ítems)

Desde la raíz del repositorio, revisar primero el plan y luego enviar **solo en el cluster**:

```bash
bash submit_adaptive_mkp_arrays.sh --dry-run estudio_01
bash submit_adaptive_mkp_arrays.sh estudio_01
bash submit_adaptive_mkp_arrays.sh --help
```

El dry-run imprime exactamente tres comandos `sbatch`, sin crear archivos ni requerir Slurm o Conda. El tag es opcional (por defecto fecha UTC/PID/azar); debe comenzar con letra/dígito y contener solo letras, dígitos, `.`, `_` o `-`. Un lote existente se rechaza, incluso en dry-run.

| Array | Familias en orden | Tareas |
|---|---|---|
| 100 | 1, 4, 7 | `0-8%9` |
| 250 | 2, 5, 8 | `0-8%9` |
| 500 | 3, 6, 9 | `0-8%9` |

La familia es `familias[task_id / 3]` (división entera) y el índice `task_id % 3`. Cada tarea ejecuta **un solo caso**, 31 corridas secuenciales con semillas 42–72: **27 casos × 31 = 837 corridas**, hasta **27 tareas concurrentes**; no hay jobs por semilla. Cada worker llama al módulo `campaign` con selector adaptativo, DDTW y `--selection-trace`. Mantiene el monitor de `config.py` (ventana 70, banda 7, pendiente 2, meseta 25, paciencia 15, percentiles 30/70). Solicita 1000 iteraciones nativas **sin cap de época**: las iteraciones reales pueden sobrepasarlas y los reportes conservan ese sobrepaso.

El worker común `run_adaptive_mkp_array.sh` pide partición `CPU`, QoS `normal`, un nodo/tarea/CPU, 4 GB y 24 horas. Activa `mkp_env` mediante `$HOME/miniconda3/etc/profile.d/conda.sh`, como el script CEC existente; ese entorno debe estar preparado con las dependencias del proyecto. Para otra instalación, exportar `MKP_CONDA_SH=/ruta/absoluta/conda.sh` antes del envío (también es el punto de aislamiento de las pruebas). No instala nada. Fija los cinco límites BLAS/OpenMP a 1, `PYTHONUNBUFFERED=1`, backend `Agg` y caché Matplotlib privada por tarea.

El submitter deriva la raíz desde su propio archivo, fija `sbatch --chdir` y pasa repo, tamaño y lote como argumentos absolutos al worker: no depende de la ruta del script temporal de Slurm. Crea logs/errors **antes del primer envío**; sus nombres absolutos incluyen tag, tamaño y `%A_%a`.

```text
resultados/adaptive_mh_selection/batch_estudio_01/
  jobs.tsv                         # tamaño e ID devuelto por sbatch --parsable
  logs/estudio_01_items_100_%A_%a.out
  errors/estudio_01_items_100_%A_%a.err
  items_100/mknapcb1_inst_00/        # reserva exclusiva; otros ocho casos al lado
    mplconfig/                     # caché privada
    campaign/                      # campaña independiente de UN caso
      protocol.json
      resumen_global.csv           # global solo de esta campaña/caso
      items_100/mknapcb1/inst_00/adaptive_ddtw/
        runs_resultados.csv        # las 31 filas numéricas
        best_run_XX/               # solo el ganador tiene trazas completas
  items_250/...                    # nueve casos, cada uno con su campaign/
  items_500/...                    # nueve casos, cada uno con su campaign/
```

Cada `campaign/` conserva el resto del árbol estándar descrito arriba: **no hay un protocolo o resumen global fusionado del lote** ni escrituras compartidas de CSV/JSON entre tareas. `jobs.tsv` es solo un registro de envíos, no evidencia de que terminaron las corridas. Si falla un envío, se detiene la secuencia, se devuelve el código de `sbatch` y se muestran los IDs ya aceptados; no se cancelan automáticamente. Si falla una tarea, su reserva/salida queda para inspección y un reintento con el mismo lote/caso se rechaza antes de activar Conda. No hay limpieza, reanudación ni fusión automática: revisar primero los jobs y resultados parciales antes de decidir otro lote.

Las pruebas locales usan Slurm, activación Conda y Python **falsos**, con rutas temporales que contienen espacios. No comprueban disponibilidad, asignación de recursos ni integración real del cluster, y no ejecutan las 837 corridas.

## Ejecución individual (para desarrollo)

`adaptive_mh_selection.run` mantiene el modo anterior de un solo caso, incluyendo carpetas `run_XX` para inspección. Para la campaña estadística con retención estricta del mejor, usar **`campaign`**, no ese modo de desarrollo.

Desde la raíz del repositorio:

```bash
# Validación sin correr solvers ni crear archivos.
python -B -m adaptive_mh_selection.run --dry-run

# Una corrida, primera familia y primera instancia: 1000 iteraciones nominales.
python -B -m adaptive_mh_selection.run --instance instancias/mknapcb1.txt --index 0 --seed 42 --runs 1 --iterations 1000

# Control uniforme usando el mismo pipeline y las mismas semillas.
python -B -m adaptive_mh_selection.run --index 0 --seed 42 --runs 3 --iterations 1000 --selectors uniform adaptive

# DDTW es el default. DTW solo si se pide.
python -B -m adaptive_mh_selection.run --index 0 --iterations 1000 --modes dtw

# Resumen de solo lectura de una campaña nueva.
python -B -m adaptive_mh_selection.analysis --campaign resultados/adaptive_mh_selection/<carpeta>
```

Se necesita Python 3.11+, NumPy, SciPy, Matplotlib y Requests del entorno del proyecto. Las instancias son locales; no se descargan paquetes ni datos. Las rutas de salida existentes se rechazan: cada ejecución crea una campaña nueva. `-B` evita modificar bytecode histórico versionado.

## Iteraciones: exactamente la semántica original

- `--iterations` / `--max-iters`: 1000 por defecto.
- Cada iteración es una entrada de `resultado.historial` del solver: generación, nivel de temperatura o paso propio de la MH, como en el híbrido original.
- El pipeline comprueba el límite **entre épocas completas**. Por eso 1000 solicitadas pueden terminar en más de 1000 reales; no se corta ni se inventa un historial más corto.
- Los reportes registran `requested_iterations`, `actual_iterations` e `iteration_overshoot`.
- `--max-epoch-iters N` expone el cap opcional que ya existe en el orquestador nativo. No se activa por defecto; cambia la duración de las épocas si se elige.
- Los flags retirados `--max-evaluations`, `--max-block-evaluations` y `--evaluation-reference` producen error, no se reinterpretan como iteraciones.

No hay equivalencia entre una iteración de GA y una de SA en número de candidatos evaluados. Se conserva esa limitación del protocolo híbrido solicitado, sin introducir otro presupuesto.

## Configuración

| Opción | Predeterminado | Uso |
|---|---|---|
| `--instance` / `--index` | `instancias/mknapcb1.txt` / `0` | Archivo OR-Library local e índice base cero. |
| `--seed` / `--runs` | `42` / `1` | Semillas consecutivas `seed + i`, empezando en 42. |
| `--seeds` | sin definir | Lista explícita de semillas distintas, alternativa a la expansión de runs. |
| `--selectors` | `adaptive` | `adaptive`, `uniform` o ambas. |
| `--modes` | `ddtw` | Detector de `RunConfig.use_ddtw`. `--modes dtw` selecciona DTW. |
| `--iterations` | `1000` | Iteraciones nativas solicitadas. |
| `--max-epoch-iters` | sin cap | Igual que la opción del pipeline original. |
| `--alpha` / `--epsilon` | `0.3` / `0.2` | Memoria exponencial y mezcla de exploración del selector. |
| `--window` / `--min-slope` | `70` / `2` | De `default_monitor` en `config.py`, no de `rotating_benchmark`. |
| `--plateau-max` / `--patience` | `25` / `15` | De `config.py`. La paciencia mínima por MH del solver original sigue aplicando. |
| `--reference-value` | cabecera del archivo | Referencia positiva explícita solo para reportar gap y graficar. |
| `--selection-trace` | desactivado | CSV opcional de elecciones, probabilidades y recompensas; sin nuevos gráficos. |
| `--no-plots` / `--dry-run` | desactivados | Reportes sin PNG / validación sin ejecución ni escritura. |

La banda 7, los percentiles 30/70, `adapt_thresholds` y `improvement_tol` no tienen flag. Viajan completos en el protocolo y en cada `RunConfig`. Cambiar `--window` o `--modes` no los reinicia.

Se mantiene la paciencia mínima 14 para GA, ILS, WOA y GWO. Los parámetros efectivos se registran en cada `summary.json`. Una referencia igual a cero significa gap desconocido, no gap cero.

## Lo que cambia: elegir la siguiente MH

Los pools y la alternancia son los nativos:

- Poblacionales: GA, PSO, GWO, WOA, EHO, ACO.
- Trayectoria: SA, TS, ILS, VNS.

Cada pool conserva su propio estado, reiniciado por corrida. El primer bloque recibe recompensa cero: establece la escala y no tiene un fitness global heredado finito con el que comparar. **No se evalúa una solución inicial adicional.** Para los bloques siguientes:

```text
ganancia = max(0, fitness_global_despues - fitness_global_antes)
escala = max(1, abs(fitness_global_al_terminar_el_primer_bloque))
recompensa = min(1, ganancia / escala / iteraciones_del_bloque)
q[mh] = (1 - alpha) * q[mh] + alpha * recompensa
```

Si todas las valoraciones son cero, las probabilidades son uniformes. Si no:

```text
p[mh] = epsilon / cantidad_mh + (1 - epsilon) * q[mh] / sum(q)
```

El control `uniform` conserva el `random.choice` y la secuencia RNG original de los solvers. La selección adaptativa tiene un generador local por pool. Los generadores globales Python/NumPy se siembran una vez por corrida y se restauran al salir. No se soportan corridas concurrentes en hilos.

## Gráficos y archivos

Se conservan cuatro funciones originales de `hybrid_mkp.plots`, sin copiar ni cambiar sus ejes, colores, leyendas o tipografía:

1. `fitness_convergence.png`: convergencia por iteraciones acumuladas, coloreada por MH.
2. `dtw_delta.png`: delta DTW con los offsets globales originales.
3. `switches_gantt.png`: turnos poblacionales/trayectoria en tiempo real.
4. `instantaneous_fitness_only.png`: fitness instantáneo por iteración.

`instantaneous_convergence.png` ya no se genera y se elimina de la carpeta de salida al redibujar. La función original del híbrido permanece intacta para otros consumidores.

La distribución `boxplot_runs.png` reutiliza el renderer del ejecutor híbrido de primeras instancias. Si no hay deltas DTW finitos, el renderer nativo omite esa imagen; no se rellenan observaciones inventadas.

```text
resultados/adaptive_mh_selection/run_iterations_<timestamp>/
  protocol.json
  environment.json
  adaptive_ddtw/
    resumen_pipeline.txt
    runs_resultados.csv
    boxplot_runs.png
    run_01/
      config.json
      summary.json
      native_result.json
      resumen_pipeline.txt
      historial_dtw.csv
      historial_dtw_detalle.txt
      selection_trace.csv          # solo con --selection-trace
    best_run_01/
      summary.json
      native_result.json
      resumen_pipeline_best_run.txt
      historial_dtw.csv
      historial_dtw_detalle.txt
      fitness_convergence.png
      dtw_delta.png
      switches_gantt.png
      instantaneous_fitness_only.png
      fig_temporal_diagnostics.png
      fig_temporal_diagnostics.pdf
      temporal_diagnostics.csv
      temporal_diagnostics.json
```

Cada condición adicional usa su propia carpeta. Igual que en los reportes originales, `n_switches` cuenta bloques; el JSON registra también las transiciones reales (`bloques - 1`). Las observaciones DTW detalladas se alinean por cada bloque: los solvers que no devuelven esos detalles quedan marcados como no disponibles, sin desplazar datos de la MH siguiente. Las salidas JSON convierten datos no disponibles a `null`.

No se lanzan automáticamente 31 corridas, benchmarks standalone de todos los algoritmos ni inferencias estadísticas nuevas para obtener estos gráficos. La corrida pedida es una sola. Los errores de ejecución o exportación se conservan en `failure.txt` y producen salida no cero.

## Diagnóstico temporal

La figura sigue el formato compacto de `fig_temporal_diagnostics (2).pdf/`: 7.4 × 8.6 in, tipografía serif/STIX y títulos a la izquierda. Esa carpeta es solo referencia de presentación. Su ejemplo BDE, los modos y los parámetros F/CR no se copiaron: este pipeline no los registra.

| Panel | Qué muestra | Qué no cambia |
|-------|-------------|---------------|
| (a) | Fitness registrado y fondo con el color real de cada MH | No es un estado binario de controlador |
| (b) | Dos ejes lado a lado: subida con ganancia proporcionalmente más cercana a su rampa verificada, y ventana plana | Solo cambia la selección ilustrativa, no la optimización ni los valores. Observado negro, rampa azul, constante naranja |
| (c) | `D1`, `D2`, `theta_r` y `theta_c` en **escala lineal**, después del calentamiento de cada bloque | Espera ventana completa y 10 muestras listas si los umbrales son adaptativos; conserva todas las magnitudes posteriores y el CSV completo |
| (d) | Delta, `theta_delta` y disparos registrados, sin franja inferior de MH | Los datos de MH siguen en el CSV; no inventa exploración/explotación ni F/CR |

Las ventanas elegidas se marcan con el mismo azul/naranja en los paneles globales. Si falta una ventana verificada, ese subeje dice que no está disponible; no se inventa.

Las referencias de (b) se reconstruyen con `window`, `band`, `min_slope` y `use_ddtw` guardados para la MH. Solo se muestran si recalcular las distancias reproduce los `D1/D2` registrados.

En las **series y CSV**, los faltantes siguen en blanco. En **(c)/(d)** los segmentos entre muestras existentes son guías visuales, no mediciones; la figura y `temporal_diagnostics.json` lo dicen. No se extrapola fuera de la primera o última muestra. Delta puede usar su historial nativo separado cuando existe.

Para redibujar una corrida guardada, sin optimizar otra vez ni cambiar su JSON/CSV original:

```bash
python -B -m adaptive_mh_selection.temporal_diagnostics --run resultados/adaptive_mh_selection/run_iterations_20261022_153247_220490/adaptive_dtw/best_run_01
```

Ese comando conserva los resultados y el resto de los archivos ajenos al dibujo, retira el PNG descartado y regenera las figuras en la misma carpeta. El `environment.json` original describe el código de la optimización; `temporal_diagnostics.json` registra por separado el hash del nuevo postprocesador.

En la corrida existente de **1041 iteraciones / fitness 24123**, hay **153 iteraciones sin detalles DTW** (GWO/PSO), además del calentamiento del monitor. Los registros siguen conservando esos faltantes. La curva delta une sus **300 muestras disponibles**; cada distancia de (c) tiene **264 muestras**, con máximos de 60044 y 60502. La referencia aportada usa DDTW y otro rango: no se copiaron su escala numérica ni sus parámetros. La nueva escala hace visible la meseta de 390 sin ocultar los picos. Se verificaron ventanas, puntos graficados, márgenes de texto y hashes de resultados/CSV/referencia PDF, sin reejecutar solvers.

## Verificación y resultados anteriores

```bash
python -B -m unittest discover -s adaptive_mh_selection/tests -v
python -B -m unittest discover -s tests -v
```

En la pasada de este cambio salieron 42 pruebas del paquete y 41 generales, ambas con código 0. Cubren el pipeline compartido, los cuatro renderers nativos, la ausencia del overlay y el diagnóstico temporal fiel a los datos. El diagnóstico ya no exige cuatro ejes: son cuatro grupos, con (b) partido y una franja de MH bajo (d). Cubren también comportamiento uniforme, alternancia, memoria global, límites de época, sobrepaso, CLI, CSV/TXT, ventanas verificadas, huecos y redibujado sin alterar resultados ni ejecutar solvers. La lectura visual de imágenes no está disponible en este entorno; se verificaron objetos Matplotlib, coordenadas, datos y archivos.

Las carpetas anteriores `campaign_*` y `verification_mkp_*` usaban evaluaciones. Se conservan como historial, pero **no corresponden a esta versión por iteraciones**. El resumen de análisis rechaza esas campañas para evitar una comparación con unidades mezcladas. Los gráficos válidos de la nueva corrida están en `run_iterations_*`.

Esta implementación no afirma que el selector adaptativo mejore al uniforme: esa hipótesis requiere un estudio separado con múltiples semillas y casos.
