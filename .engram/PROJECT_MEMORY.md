# Estado y decisiones del proyecto

## Estado Git comprobado al guardar esta memoria

- Repositorio: `https://github.com/lukipuki0/mkp_lb2.git`.
- Rama actual: `main`, commit `11cfd85`.
- Trabajo de selección adaptativa publicado en `feat/adaptive-exploratory-selector`, commit `1c87bdddd07d85c1fff5b9d2aa48cc5a4db8c718`. No se hizo PR ni merge en esta sesión. La referencia remota local conserva ese commit; no se hizo fetch al crear esta memoria.
- Cambios ajenos preservados: `a/`, `continuous_benchmark/__pycache__/funciones_cec2022.cpython-314.pyc`, `odd/tasks/time-series-presentation.md`. También aparece `6adf64c78fbcdfbb0016599d/`, trabajo de plantilla informado por otra sesión.
- Los gráficos y resultados experimentales quedaron locales; no se publicaron con el código.

## Mapa conceptual

- MKP maximiza beneficio con representación binaria y restricciones de recursos; los candidatos se reparan antes de evaluarse.
- DTW/DDTW observa una serie de mejor fitness histórico. No genera soluciones, no calcula el objetivo y no prueba optimalidad.
- D1 compara la ventana con una rampa de progreso; D2 con una constante; delta = D1 - D2.
- Una meseta del mejor fitness no demuestra falta de movimiento o diversidad de la población.
- DDTW sobre primeras diferencias elimina desplazamientos verticales, no toda dependencia de escala. Las distancias acumuladas dependen de la longitud de ventana.

## Detectores: no mezclar versiones

El detector compartido `dtw_stagnation.py` calcula tres condiciones simultáneas:
1. `no_improve_len >= plateau_max`.
2. `D2 <= theta_c`.
3. `D1 >= theta_r` o `delta >= theta_delta`.

La conjunción debe persistir `patience` observaciones; antes se requiere una ventana completa.

`ALGORITMO_CENTRAL.md`, leído completo, describe otra selección de variantes activas: `binary_simple` usa solo D2 <= theta_c; `binary_hysteresis` entra con delta >= theta_delta y sale con delta <= 0. En esa descripción las estrategias no consumen el fire interno de tres condiciones. Advierte además un presupuesto de 20 iteraciones con ventana 20, insuficiente para que una adaptación posterior al último paso influya en la búsqueda. Esto es evidencia del documento, no comprobación de todos sus valores contra código actual.

La carpeta `DTW_optimization-main` inspeccionada contiene variantes binarias y continuas que adaptan parámetros de una sola MH (GA/PSO/GWO/DE). Su README y el documento central no describen exactamente la misma selección de estrategias. Verificar el runner concreto antes de diseñar una comparación.

## Selección adaptativa implementada en la rama de trabajo

- `adaptive`: scores EWMA por pool, actualizados solo para la MH elegida; mezcla explotación normalizada con exploración uniforme mediante epsilon.
- `uniform`: comportamiento uniforme original, con pruebas de paridad RNG.
- `adaptive_exploratory`: variante optativa, exploración proporcional a `1/sqrt(uses+1)` mezclada con los scores; si todos los scores son cero, usa esa distribución de exploración.
- Limite importante: si solo una MH tiene score positivo, normalizarlo sigue asignándole toda la parte de explotación aunque su score decaiga. La variante modifica principalmente el componente exploratorio, no resuelve por completo esa concentración.
- `adaptive` permanece como predeterminado; ningún piloto justificó reemplazarlo.
- Pools optativos en `run`: `--population-pool GA PSO ACO --trajectory-pool SA ILS VNS`. No se modifican las constantes nativas globales. No se agregaron esos flags a `campaign`.
- Las probabilidades son condicionales a cada pool; no son frecuencias de selección ni probabilidades globales entre todos los algoritmos.

## Gráficos y salidas aceptados

- Temporal diagnostics: título sin semilla, leyenda de colores MH, líneas más gruesas, delta verde y umbral magenta discontinuo, sin texto inferior.
- Panel (c): conservar escala lineal y datos. Usuario pospuso symlog/zoom.
- Probabilidades: un mini-panel por MH, PNG y PDF; seis paneles en pools reducidos. Colores nativos, eje 0–100%, datos registrados `probabilities_before`.
- No generar `instantaneous_fitness_only.png` desde el pipeline adaptativo; conservar archivos antiguos. Su renderer sigue disponible para otros consumidores.
- Redibujar desde `selection_trace.csv` sin volver a optimizar cuando solo cambia la presentación.
- En `run`, el CSV está en `run_01/`; el gráfico ganador está en `best_run_01/`.
- Campañas nuevas: un caso escribe directamente bajo output; varios usan `items_N/mknapcbF_inst_XX`; carpeta de condición solo al comparar varias. HPC reserva cada caso en `.reservations/` separado de la salida. `run` conserva su estructura propia de condiciones y runs. Lectura de campañas anteriores preservada; no se migraron resultados.

## Referencias y gap

- Fuente: https://people.brunel.ac.uk/~mastjjb/jeb/orlib/files/mkcbres.txt
- Copia publicada en la rama: `adaptive_mh_selection/data/mkcbres.txt`.
- Primera tabla: 270 mejores valores factibles publicados, NO necesariamente óptimos certificados. Segunda tabla: relajación LP, no se usa como referencia factible.
- Ejemplo: mknapcb1 índice 0 = 5.100-00 = 24381.
- Cabecera no nula prevalece. Si es cero, aplicar la tabla solo cuando el SHA-256 del archivo coincida con el benchmark incluido. `--reference-value` explícito prevalece en `run`.
- Brecha = 100*(referencia - fitness)/referencia. Puede ser negativa al superar la referencia. Referencia desconocida implica gap ausente, no cero.
- Precaución de portabilidad pendiente: el mapeo usa hashes de bytes; un cambio de finales de línea puede invalidar la coincidencia aunque los números sean iguales.

## Propuesta actual: híbrido memético GA–SA — NO implementado

El profesor pidió dos MH que se comuniquen y usen DTW. Se discutieron cooperación por intercambio de soluciones, población compartida, relevo con memoria y enfoque memético.

La última propuesta discutida es:
1. GA ejecuta selección, cruce, mutación, reparación y evaluación.
2. DTW/DDTW observa el mejor fitness por generación.
3. Solo al detectar estancamiento, activar SA sobre individuos seleccionados del GA.
4. SA explora vecinos, acepta mejoras y acepta empeoramientos con exp((f_nuevo-f_actual)/T) en maximización; conserva su mejor solución aparte.
5. SA devuelve sus mejores soluciones a GA; el GA conserva su población/elitismo y continúa.
6. Rearmar el detector y esperar evidencia nueva antes de otra activación; evitar SA en cada generación de la misma meseta.

Decisiones todavía abiertas: regla de activación concreta (tres condiciones/simple/histéresis), individuos a refinar, vecindario/reparación, temperatura/enfriamiento, presupuesto SA, reinserción y rearme. La sugerencia de élite más individuos diversos no está aprobada como diseño definitivo.

Evaluación propuesta: GA solo, GA–SA periódico y GA–SA activado por DDTW, con semillas emparejadas y presupuesto de evaluaciones que incluya SA. Primero aislar el disparador DTW, sin cambiar a la vez parámetros de SA o selección de MH. No hay autorización vigente para lanzar esta implementación por esta nota.

## Otras ideas discutidas, sin implementación

Ventana adaptativa, DTW multiescala, biblioteca de patrones, control de parámetros, reinicios selectivos, memoria de trayectorias y coordinación entre poblaciones. Solo la ventana adaptativa fue recordada explícitamente por el usuario de una conversación anterior; el resto se presentó como reconstrucción/propuestas, no como lista histórica recuperada.

## Verificación y pendientes

- Entorno de pruebas: `.venv/Scripts/python.exe -B`; el Python del sistema falló por falta de `requests`.
- Verificación final publicada: 86/87 pruebas adaptativas + 41/41 generales = 127/128.
- Único fallo observado: `test_resource_directives_and_lf`, CRLF preexistente en `submit_adaptive_mkp_arrays.sh`. Usuario eligió publicar sin corregir. No declarar suite completa verde.
- Pools reducidos: 6 pruebas nuevas + 27 de compatibilidad pasaron.
- No hubo prueba real Slurm. Se verificaron aislamiento y rechazo de sobrescritura mediante simulaciones.
- Pendiente comparar múltiples semillas (piloto de 5 o estudio de 31), no decidir por semilla 60.
- Pendiente investigar variación de tiempos del piloto exploratory; no atribuirla automáticamente al selector.
- No hubo promoción de variante a default, ni PR/merge de la rama publicada.
