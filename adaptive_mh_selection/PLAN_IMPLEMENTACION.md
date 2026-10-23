# Variante adaptativa con el formato original de hybrid_mkp

> Estado: implementada para MKP y verificada por iteraciones nativas.
> Esta versión reemplaza el diseño inicial por evaluaciones. El usuario pidió conservar el flujo, las unidades y los gráficos del híbrido original.

## 1. Alcance confirmado

Corrección del usuario, textual:

> mira usa el mismo formato del hydrid mkp quie cambia de mh mo hagas cosas nuevas, usaos iteraciones okey?, reace lso graficos y crea lo mismo que hay ahi okey?

El sistema debe mantener el pipeline de `hybrid_mkp` y cambiar únicamente la selección de la siguiente metaheurística (MH). No se implementa otro motor de búsqueda, otro esquema de inicialización ni un presupuesto por evaluaciones.

## 2. Funcionamiento

```text
Instancia MKP
  → mismo orquestador híbrido
  → elegir MH dentro del pool de turno
  → ejecutar su época original
  → DTW detecta estancamiento o termina la época
  → conservar e inyectar la mejor solución global
  → actualizar rendimiento de la MH ejecutada
  → alternar pool poblacional / trayectoria
  → repetir hasta alcanzar las iteraciones solicitadas entre épocas
```

- **Cuándo cambiar:** las reglas y límites nativos del híbrido.
- **A qué MH cambiar:** sorteo adaptativo basado en rendimiento; el control uniforme conserva `random.choice`.
- **Unidad:** entradas del historial de cada algoritmo, como en el híbrido.

Pools originales:

| Pool | Algoritmos |
|---|---|
| Poblacional | GA, PSO, GWO, WOA, EHO, ACO |
| Trayectoria | SA, TS, ILS, VNS |

## 3. Único aprendizaje agregado: el selector

Cada pool mantiene valoraciones independientes `q[mh]`. Todo el estado se reinicia por corrida.

El primer bloque no tiene un fitness global heredado finito: recibe recompensa cero y fija una escala. No se agrega una evaluación de inicialización. Para los siguientes bloques:

```text
ganancia = max(0, fitness_despues - fitness_antes)
escala = max(1, abs(fitness_al_final_del_primer_bloque))
recompensa = min(1, ganancia / escala / iteraciones_reales_del_bloque)
q[mh] = (1 - alpha) * q[mh] + alpha * recompensa
```

Con valoraciones cero se elige uniformemente. De lo contrario:

```text
p[mh] = epsilon / cantidad_mh + (1 - epsilon) * q[mh] / sum(q)
```

Defaults: `alpha=0.3`, `epsilon=0.2`. Una mayor valoración aumenta la probabilidad, no fuerza la selección. Las MH no elegidas conservan su valoración. La probabilidad mínima no garantiza visitarlas todas en una corrida corta.

La recompensa es aporte reciente por iteración, no fitness absoluto heredado. No captura todo el crédito cooperativo: un algoritmo puede preparar una región que otro aprovecha. No se afirma superioridad científica por pasar pruebas funcionales.

## 4. Reutilización del código existente

`hybrid_mkp.orchestrator.ejecutar_pipeline()` tiene un argumento opcional `policy`. Sin política, conserva exactamente su `random.choice`, operadores y comportamiento original.

La política adaptativa implementa:

```python
class SelectionPolicy:
    def choose(self, pool_names, tipo):
        # Sortea una MH perteneciente al pool nativo.
        ...

    def observe(self, mh, tipo, fitness_before, epoch_result, n_iters, fitness_after):
        # Actualiza solo el pool del bloque completado.
        ...
```

El wrapper usa el pipeline compartido y retorna el `PipelineResult` nativo junto con un resumen y registros de selección. No contiene una copia del bucle del orquestador.

Se retiraron `budget.py`, el adaptador por evaluaciones y los hooks de cálculo de fitness que se habían agregado a las reparaciones y ACO. WOA/ABC independientes y la limpieza previa de la integración cooperativa obsoleta se conservan.

## 5. Iteraciones y sobrepaso

El valor predeterminado es `max_iters=1000`, sin cap adicional de época, igual que `hybrid_mkp/rotating_benchmark.py`.

El pipeline comprueba la parada al finalizar cada época. Por lo tanto, puede superar el número nominal. No se truncan épocas, resultados ni gráficos para ocultarlo. Se registran:

- `requested_iterations`: solicitadas.
- `actual_iterations`: longitud real del historial.
- `iteration_overshoot`: diferencia no negativa.

La opción nativa `max_epoch_iters` solo se usa si se solicita expresamente. No se convierte una evaluación en una iteración. Una generación de GA y un paso de SA siguen teniendo costos distintos; esto forma parte del protocolo original que se decidió mantener.

Defaults de monitor: ventana 40, pendiente mínima 0.5, meseta 15, paciencia 8, umbrales adaptativos 30/70 y DTW estándar. Se mantienen las particularidades nativas, incluida paciencia mínima 14 para GA, ILS, WOA y GWO.

## 6. Estructura implementada

```text
adaptive_mh_selection/
  config.py          # Configuración por iteraciones
  selection.py       # Sorteo adaptativo y control uniforme
  reward.py          # Mejora por iteración
  orchestrator.py    # Wrapper y política sobre el pipeline original
  plotting.py        # Llamadas a renderers originales, no gráficos nuevos
  exports.py         # CSV/TXT/JSON con el formato híbrido
  run.py             # Ejecución local de una o varias corridas
  analysis.py        # Lectura de resúmenes por iteraciones
  tests/             # Regresiones de integración, selección y exportación
  README.md          # Comandos y límites efectivos
```

## 7. Gráficos y resultados

Se reutilizan cuatro funciones originales de `hybrid_mkp.plots`. Por pedido posterior del usuario se retiró el overlay `instantaneous_convergence.png`; la función original permanece intacta para otros experimentos:

| Función nativa | Archivo |
|---|---|
| `grafico_convergencia` | `fitness_convergence.png` |
| `grafico_dtw_delta` | `dtw_delta.png` |
| `grafico_switches` | `switches_gantt.png` |
| `grafico_solo_instantaneo` | `instantaneous_fitness_only.png` |

Además, se reutiliza el boxplot de distribución de runs del ejecutor híbrido. No se generan gráficos adicionales de probabilidades, frecuencia ni recompensas. Un CSV opcional `selection_trace.csv` permite inspeccionar qué aprendió el selector.

Las salidas mantienen `resumen_pipeline.txt`, `runs_resultados.csv`, `historial_dtw.csv`, `historial_dtw_detalle.txt` y `best_run_XX/resumen_pipeline_best_run.txt`. El mejor run contiene los gráficos. Los detalles DTW no devueltos por algún solver se marcan como no disponibles sin desplazar la información del bloque siguiente.

## 8. Ejecución y verificación observadas

```bash
python -B -m adaptive_mh_selection.run --instance instancias/mknapcb1.txt --index 0 --seed 42 --runs 1 --selectors adaptive --modes dtw --iterations 1000 --selection-trace
```

La corrida real de la primera instancia (100 ítems, 5 restricciones) terminó correctamente:

- Iteraciones solicitadas: **1000**.
- Iteraciones reales: **1041**; sobrepaso de la última época: **41**.
- Mejor fitness: **24123**.
- Los cinco gráficos originales y el boxplot se generaron sin errores.
- Carpeta: `resultados/adaptive_mh_selection/run_iterations_20261022_153247_220490/`.

Verificación anterior a la campaña agrupada: **24** pruebas del paquete y **41** generales pasaron. Cuatro casos del pipeline original, con y sin cap, conservaron sus resultados frente al baseline anterior a la extensión. No hubo lectura visual de PNGs disponible: se verificó la llamada a los renderers originales, argumentos, offsets y archivos producidos.

Los resultados antiguos por evaluaciones se conservan como históricos. No son comparables directamente con esta corrida por iteraciones y no se renombraron para aparentar otra unidad. El lector de análisis rechaza el formato retirado.

## 9. Diagnóstico temporal solicitado posteriormente

El usuario pidió borrar `instantaneous_convergence` y tomar la carpeta `fig_temporal_diagnostics (2)` solo como referencia, sin inventar elementos de implementación. Se agregó `temporal_diagnostics.py`, que procesa la corrida guardada y no ejecuta el script de referencia ni los solvers.

Los cuatro paneles son: fitness/MH activa; ventanas reales frente a referencias del monitor; distancias y umbrales registrados; delta/umbral/disparos observados. No hay estados BDE, F/CR, histéresis ni cambios de parámetros ficticios. Las ventanas no cruzan bloques y las referencias derivadas solo se aceptan si reproducen los D1/D2 guardados.

La figura se guardó como `fig_temporal_diagnostics.png` y `.pdf` junto a los gráficos del mejor run. Se conservaron intactos los archivos originales de resultados: misma solución, fitness 24123 y 1041 iteraciones. Los 153 puntos sin detalles DTW permanecen ausentes en las series originales y el CSV. Se preservó `instantaneous_fitness_only.png`, que no fue pedido borrar. El hash del postprocesador se guarda por separado del entorno original de la optimización.

Ajuste posterior de presentación pedido al comparar el PDF: (c)/(d) unen solo muestras finitas existentes para evitar cortes visuales, sin rellenar numéricamente los faltantes. (c) usa escala symlog explícita para conservar cero y los picos de más de 60000 mientras permite leer mesetas de 390. No se copió el rango de la referencia DDTW a esta corrida DTW. Se aumentaron separación de títulos/etiquetas y legibilidad; la figura informa que las conexiones sobre intervalos sin muestras son guías visuales. Resultados, CSV y referencia PDF quedaron idénticos por hash.

## 10. Uso actual

Consultar [README.md](README.md) para las opciones completas. El ejecutor individual `run` conserva una corrida predeterminada y salidas de depuración. El nuevo ejecutor `campaign` tiene los 27 casos y 31 semillas por caso solicitados en la sección siguiente. Ninguno lanza benchmarks standalone de todos los algoritmos para verificar la implementación.

Para comparaciones, fijar los mismos casos, semillas, monitor e iteraciones nominales en control uniforme y adaptativo; reportar también sus iteraciones reales. La campaña extensa se lanza explícitamente, no como prueba automática de código.

## 11. Campaña agrupada y conservación para estadística

Pedidos del usuario:

> bien ahora organiza para ejecutar las 9 instancias los 3 primero índices, okey?, también quiero que lo agrupes en grupos de 100 240 y 500 ítems según la instancia dime cómo lo vas a a organizar la estructura de los resultados primero

> después ten en cuenta que se hará análisis estadística y serán 31 ejecuciones, por instancia solo se guardará la mejor de todas si?

> ya dale implementa

Se verificó en los archivos que el tamaño intermedio es 250, no 240. La organización usa los tamaños reales: 100 (`mknapcb1/4/7`), 250 (`mknapcb2/5/8`) y 500 (`mknapcb3/6/9`). Los índices son 0, 1 y 2 de cada archivo: 27 problemas concretos y 837 intentos con 31 semillas por problema para una condición.

Por caso/condición se conserva `runs_resultados.csv` con todas las mediciones y sus semillas/estados, pero solo el run de mayor fitness conserva solución, historiales y gráficos completos en `best_run_XX`. Empates: primera ejecución. `XX` es el ordinal original, no una numeración que oculte ejecuciones fallidas. No se guardan los 31 historiales en memoria ni en carpetas individuales.

Los resultados se agrupan en `items_N/mknapcbF/inst_XX/condicion/`. La raíz conserva protocolo, versiones, estado y resumen global; cada tamaño tiene su resumen de grupo. Los resúmenes globales son tablas por caso, no promedios de objetivos crudos de problemas diferentes.

La campaña es secuencial; las métricas se guardan tras cada intento y los fallos quedan explícitos. No se inventan fitness para fallos, no se copian referencias del índice 0 a otros índices, no se anuncia éxito con menos de 31 resultados válidos, y no se fabrica un ganador si todos fallan. No hay reintentos ni reanudación automática.

El protocolo y los defaults se inspeccionan con `python -B -m adaptive_mh_selection.campaign --dry-run`. La implementación se verifica con casos simulados y una tanda real reducida; no se ejecutan las 837 corridas para probar el código. El análisis inferencial posterior debe usar todas las mediciones, no únicamente los ganadores guardados.

Estado verificado: dry-run 27 casos / 837 intentos, 40 pruebas del paquete y 41 generales correctas. El piloto final de tres tamaños con dos corridas cortas por caso terminó sin fallos en `resultados/adaptive_mh_selection/campaign_20261022_184854_360468/`: seis mediciones accesibles, tres ganadores completos y ningún historial de perdedores. Reportes TXT por condición, hashes de fuentes/instancias y soluciones se comprobaron; las 837 corridas extensas siguen sin ejecutarse.
