# Plan de revisión de `paper_final.tex`

## Alcance correcto del paper

Este plan se construye únicamente sobre las tres líneas de trabajo que forman
el paper final:

1. `hybrid_mkp/`: problema de la mochila multidimensional (MKP).
2. `continuous_benchmark/`: funciones continuas IEEE CEC2022.
3. `HRES2-H2/`: dimensionamiento del sistema híbrido eólico--PV--electrolizador--batería para hidrógeno verde.

`paper/revisores.md` contiene comentarios de otro trabajo. Por eso, sus
algoritmos, cifras y parámetros no deben trasladarse literalmente. Los
comentarios sí sirven como lista de verificaciones metodológicas: estadística,
ablaciones, sensibilidad, formulación de DDTW, reproducibilidad y claridad del
manuscrito.

# Plan de revisión de `paper_final.tex`

## Alcance correcto del paper

Este plan se construye únicamente sobre las tres líneas de trabajo que forman
el paper final:

1. `hybrid_mkp/`: problema de la mochila multidimensional (MKP).
2. `continuous_benchmark/`: funciones continuas IEEE CEC2022.
3. `HRES2-H2/`: dimensionamiento del sistema híbrido eólico--PV--electrolizador--batería para hidrógeno verde.

`paper/revisores.md` contiene comentarios de otro trabajo. Por eso, sus
algoritmos, cifras y parámetros no deben trasladarse literalmente. Los
comentarios sí sirven como lista de verificaciones metodológicas: estadística,
ablaciones, sensibilidad, formulación de DDTW, reproducibilidad y claridad del
manuscrito.

El método que debe describirse es el framework rotacional: el monitor DTW/DDTW
observa la trayectoria `best-so-far`, detecta estancamiento y provoca el cambio
entre metaheurísticas, transfiriendo la mejor solución global. No debe
presentarse como un método que modifica continuamente los parámetros internos
de una única metaheurística.

---

## 1. Diagnóstico general

### Lo que ya existe en el proyecto

- Un monitor compartido en `dtw_stagnation.py` con DTW, DDTW, banda
  Sakoe--Chiba, referencias de rampa y meseta, percentiles adaptativos y filtro
  de persistencia.
- Un orquestador rotacional para MKP en `hybrid_mkp/orchestrator.py`.
- Un orquestador para CEC2022 en `continuous_benchmark/orchestrator.py`.
- Una extensión del orquestador para HRES2-H2 en `HRES2-H2/orchestrator.py`.
- Campañas de 31 ejecuciones en los scripts de benchmark de MKP y HRES2-H2 y
  en la evaluación de CEC2022.
- Comparaciones DTW frente a DDTW, especialmente en
  `continuous_benchmark/analisis_global_dtw_ddtw.py` y en las campañas finales
  de `resultados finales/`.
- Telemetría del monitor en CSV/TXT con `D1`, `D2`, `Delta`, umbrales, contador
  de no mejora, persistencia, disparo y estado del monitor. Esto aparece en
  `HRES2-H2/benchmark_hres2.py`, `hybrid_mkp/batch_benchmark.py` y
  `hybrid_mkp/parallel_mkp_first_inst.py`.
- Pruebas no paramétricas y generación de estadísticas en los módulos
  `analisis_estadistico.py` de MKP/CEC y en
  `HRES2-H2/analisis_estadistico_hres2.py`.

### Lo que no está demostrado todavía

- No existe una campaña sistemática de sensibilidad de hiperparámetros.
- No existe una ablación completa que separe monitor, rotación, memoria elitista
  y tipo de distancia.
- No existe una comparación clara contra un controlador adaptativo externo
  (control difuso, aprendizaje por refuerzo, selección adaptativa de
  operadores, etc.).
- La configuración del monitor no es única entre los scripts de las tres
  carpetas.
- No está suficientemente documentado qué pool de solvers se utilizó en cada
  dominio.
- Las figuras actuales no muestran todos los estados y señales que el revisor
  solicita.

---

## 2. Correspondencia entre los comentarios y este paper

| Comentario                                      | Estado para este paper                                                  | Decisión                                                                                                                        |
| ----------------------------------------------- | ----------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| R1.1: evidencia estadística de superioridad    | **Parcialmente cubierto**                                         | Rehacer la trazabilidad de cifras y moderar las conclusiones según cada dominio.                                                |
| R1.2: comparar con control adaptativo existente | **Falta**                                                         | Añadir al menos un baseline externo o declarar explícitamente que queda fuera del alcance.                                     |
| R1.3: conjunto experimental reducido            | **No aplica literalmente; la preocupación sigue siendo válida** | Defender los tres dominios y declarar límites: MKP seleccionado, CEC solo`D=10`, HRES con un perfil meteorológico.           |
| R1.4: sensibilidad de hiperparámetros          | **Falta**                                                         | Ejecutar una campaña separada de sensibilidad.                                                                                  |
| R1.5: efectos inconsistentes entre algoritmos   | **Parcialmente cubierto**                                         | Hacer un análisis por solver y por tipo de transición, no solo una discusión general.                                         |
| R2.1: explicar la estructura del artículo      | **Falta**                                                         | Añadir un párrafo al final de la introducción.                                                                                |
| R2.2: ablación que pruebe el efecto de DDTW    | **Parcialmente cubierto**                                         | Usar DTW vs. DDTW existente y añadir controles sin monitor/rotación/memoria.                                                   |
| R2.3: justificar rampa y meseta                 | **Parcialmente cubierto**                                         | Explicar que son referencias mínimas y validar su sensibilidad; no presentarlas como modelo completo de todas las trayectorias. |
| R2.4: baselines adaptativos de estado del arte  | **Falta**                                                         | Igual que R1.2: añadir uno o limitar la afirmación de novedad/superioridad.                                                    |
| R2.5: reproducibilidad                          | **Parcialmente cubierto**                                         | Consolidar configuración, semillas, comandos, datos y telemetría; añadir repositorio o material suplementario.                |
| R2.6: corregir el abstract                      | **Necesario**                                                     | Reescribirlo después de cerrar las cifras y las afirmaciones.                                                                   |
| R3.1: definir la brecha de investigación       | **Parcialmente cubierto**                                         | Diferenciar claramente cambio de solver, control de parámetros e hiperheurísticas.                                             |
| R3.2: fortalecer la literatura                  | **Parcialmente cubierto**                                         | La introducción cubre adaptación, RL y selección; falta reforzar control difuso y análisis de trayectorias.                  |
| R3.3: formulación completa de DDTW             | **Parcialmente cubierto**                                         | La matemática existe, pero debe coincidir exactamente con el código y aclarar la ausencia de normalización.                   |
| R3.4: figuras de señales y estados             | **Parcialmente cubierto**                                         | Los datos existen en parte; falta generar figuras unificadas para el manuscrito.                                                 |
| R3.5: selección de parámetros                 | **Falta documentarlo**                                            | Justificar la configuración y resolver las diferencias entre scripts.                                                           |
| R3.6: pseudocódigo completo                    | **Parcialmente cubierto**                                         | Existe un pseudocódigo de alto nivel; hay que incluir warm-up, umbrales, signo, memoria y rotación.                            |
| R3.7: interacción según el algoritmo          | **Parcialmente cubierto**                                         | Analizar solvers reales de estas tres campañas; no trasladar las conclusiones sobre BDE/BPSO/BGWO.                              |
| R3.8: figuras de transiciones/parámetros       | **Parcialmente cubierto**                                         | Mostrar transiciones entre solvers; los cambios internos de parámetros no corresponden al método actual.                       |

---

## 3. Correcciones prioritarias de coherencia

Estas tareas deben hacerse antes de volver a correr experimentos o interpretar
resultados.

### 3.1 Definir los pools reales por dominio

El manuscrito habla de diez solvers como si el mismo pool se utilizara en todos
los dominios, pero el código muestra una configuración específica por dominio:

- **MKP** (`hybrid_mkp/orchestrator.py`):
  - poblacionales: `GA`, `PSO`, `GWO`, `WOA`, `EHO`, `ACO`;
  - trayectoria: `SA`, `TS`, `ILS`, `VNS`.
- **CEC2022** (`continuous_benchmark/orchestrator.py`): el pool principal
  configurado contiene `WOA`, `PSO`, `EHO`, `GWO` y `ACO`; no debe describirse
  automáticamente como el pool de diez algoritmos.
- **HRES2-H2** (`HRES2-H2/orchestrator.py`):
  - poblacionales: `PSO`, `GWO`, `WOA`, `EHO`, `ACO`, `ABC`;
  - trayectoria: `ILS`, `SA`, `TS`, `VNS`.

Acciones:

- [ ] Decidir si el paper describirá pools específicos por dominio. Esta es la
  opción coherente con el código actual.
- [ ] Revisar todas las menciones a “ten representative solvers”, “all ten
  baselines” y “five standalone methods”.
- [ ] Construir una tabla del paper con: dominio, pool híbrido, baselines
  standalone, presupuesto y número de corridas.
- [ ] Verificar que las tablas y las pruebas estadísticas comparen exactamente
  los algoritmos que se mencionan en el texto.

### 3.2 Unificar la configuración del monitor

Hay diferencias entre los scripts que parecen representar campañas distintas:

- `HRES2-H2/run_hres2.py`: `W=40`, `s_min=0.0`, `K=15`, `P=8`, DTW.
- `HRES2-H2/benchmark_hres2.py`: `W=40`, `s_min=0.0`, `K=15`, `P=3`, DDTW.
- `hybrid_mkp/rotating_benchmark.py`: `W=40`, `s_min=0.5`, `K=15`, `P=8`, DTW.
- `hybrid_mkp/batch_benchmark.py`: `W=40`, `s_min=0.1`, `K=15`, `P=8`, DDTW.
- `hybrid_mkp/parallel_mkp_first_inst.py`: `W=75`, `s_min=0.1`, `K=15`, `P=25`, DDTW.
- `paper_final.tex`: reporta `s_min=0.1` y diferencia `P=8`/`P=25` según
  presupuesto, pero no deja claro qué script generó cada resultado.

Acciones:

- [ ] Crear una tabla maestra de configuración con una fila por campaña real.
- [ ] Asociar cada fila con el script exacto y la carpeta de resultados.
- [ ] Confirmar si las diferencias son deliberadas por dominio/presupuesto o
  si son restos de experimentos previos.
- [ ] Corregir código, resultados o manuscrito hasta que los tres coincidan.
- [ ] Aclarar que `band=0` significa banda automática y reportar el ancho
  efectivo utilizado.
- [ ] Revisar el `patience` específico aplicado en
  `hybrid_mkp/orchestrator.py` para algunos solvers, porque el orquestador
  puede aumentar la paciencia respecto de la configuración global.

### 3.3 Corregir discrepancias visibles del manuscrito

- [ ] En la tabla HRES2-H2, el código ejecuta diez baselines (`PSO`, `GWO`,
  `WOA`, `EHO`, `ACO`, `ABC`, `ILS`, `SA`, `TS`, `VNS`), pero la tabla
  actual muestra nueve. Añadir `ABC` o explicar y recalcular la comparación
  como una campaña de nueve.
- [ ] Revisar las frases “ten comparisons”, “10/0/0” y “all ten standalone
  baselines” después de resolver la tabla anterior.
- [ ] Corregir leyendas de figuras que muestran `ABC` cuando la campaña o la
  figura corresponde a `ACO`.
- [ ] Corregir títulos de figuras DDTW que todavía dicen DTW.
- [ ] Eliminar la frase duplicada en la explicación de los tres buffers de
  memoria.
- [ ] Resolver la contradicción entre la fórmula adaptativa de `s_min` y la
  tabla que lo presenta como fijo en `0.1`.
- [ ] Corregir la oración del abstract sobre “less sensitive to changes in
  objective scale mechanism”.
- [ ] Eliminar el fragmento suelto `microgrid.` que aparece al final de la
  síntesis estadística.
- [ ] Revisar `figures/Diagrama_flujo.png`, porque el paper lo referencia y el
  archivo debe estar disponible para compilar.
- [ ] Reemplazar `\bibliography{sample}` y los textos de plantilla por la
  bibliografía real y una declaración de disponibilidad de datos concreta.

---

## 4. Plan detallado según los revisores

### R1.1 — Evidencia estadística y afirmaciones de superioridad

**Estado:** parcialmente cubierto.

El paper actual ya contiene Friedman, Wilcoxon, Mann--Whitney, Holm y
comparaciones DTW--DDTW. Las tres carpetas también generan análisis estadístico.
El problema pendiente no es añadir pruebas al azar, sino garantizar que cada
cifra tenga una fuente reproducible y que la conclusión no exceda la evidencia.

Acciones:

- [ ] Crear una tabla de trazabilidad: afirmación del paper → CSV/resultado →
  script que lo produjo → número de corridas.
- [ ] Separar claramente las unidades estadísticas:
  instancia MKP, función CEC2022 y corrida HRES2-H2.
- [ ] No reutilizar el `chi^2` ni el `p` del trabajo anterior.
- [ ] Reportar tamaño del efecto o diferencia práctica junto con `p` cuando sea
  posible.
- [ ] Mantener separados los resultados pareados DTW--DDTW y las comparaciones
  independientes contra baselines standalone.
- [ ] Evitar “superioridad universal”, “optimal” o “state-of-the-art” cuando
  solo haya una ventaja promedio o resultados mixtos.
- [ ] Revisar la afirmación de que DTW/DDTW no difieren significativamente en
  todos los dominios y comprobar la corrección de multiplicidad usada en
  cada familia.
- [ ] Preservar los valores crudos por corrida para HRES2-H2; el script
  `benchmark_hres2.py` debe guardar las observaciones de todos los baselines,
  no solo medias y p-valores.

### R1.2 y R2.4 — Comparación con control adaptativo externo

**Estado:** no está implementado en las tres líneas de trabajo.

Los algoritmos standalone no son un baseline de control adaptativo. DTW frente
a DDTW son variantes del propio monitor, no métodos externos de control.

Decisión necesaria:

- [ ] **Opción recomendada si se quiere responder completamente al revisor:**
  implementar al menos un controlador adaptativo externo sencillo y
  reproducible, con el mismo presupuesto, semillas, pools y dominios.
- [ ] **Opción mínima:** no añadirlo y declarar explícitamente que la
  contribución se limita a comparar DTW/DDTW, solvers standalone y controles
  internos; eliminar cualquier frase que sugiera superioridad frente a todo
  el estado del arte adaptativo.

Si se implementa un baseline externo:

- [ ] Definir antes de correrlo qué señal controla, qué parámetros modifica y
  cuál es su presupuesto adicional.
- [ ] Usar las mismas 31 semillas cuando sea viable.
- [ ] Reportar tanto calidad como número de cambios, tiempo y sobrecosto.
- [ ] No comparar un controlador adaptativo con una configuración standalone
  usando presupuestos distintos.

### R1.3 — Cobertura experimental

**Estado:** la crítica literal no corresponde, porque el paper actual no se
limita a MKP. Sí corresponde responder la preocupación sobre generalidad.

La cobertura actual es:

- MKP: nueve instancias Chu--Beasley seleccionadas.
- CEC2022: doce funciones, pero solo dimensión `D=10` en el paper.
- HRES2-H2: una formulación de ingeniería y un perfil meteorológico sintético
  anual fijo.

Acciones:

- [ ] Explicar que los tres dominios fueron elegidos para cubrir optimización
  combinatoria, continua y de ingeniería.
- [ ] Justificar la selección de las nueve instancias MKP y no llamarla una
  cobertura exhaustiva de todos los tamaños posibles.
- [ ] Decidir si se incorporará también `D=20` de CEC2022, ya que existen datos
  de entrada en `continuous_benchmark/input_data/`.
- [ ] Si no se incorpora `D=20`, dejarlo como limitación explícita.
- [ ] Para HRES2-H2, declarar que un solo perfil meteorológico no permite
  generalizar a otros años, precios, demandas o escenarios de incertidumbre.
- [ ] No presentar los tres dominios como prueba de robustez universal.

### R1.4 — Sensibilidad de hiperparámetros

**Estado:** falta la campaña experimental.

Los scripts permiten configurar los parámetros, pero que un parámetro sea
editable no constituye un estudio de sensibilidad.

Parámetros mínimos:

- longitud de ventana `W`;
- banda Sakoe--Chiba `band`;
- pendiente mínima `s_min`;
- tolerancia de meseta `plateau_max`;
- persistencia `patience`;
- percentiles `p_low` y `p_high`;
- DTW frente a DDTW;
- activación/desactivación de umbrales adaptativos.

Diseño sugerido:

- [ ] Primero fijar una configuración canónica por presupuesto.
- [ ] Hacer un análisis de un factor a la vez para detectar parámetros críticos.
- [ ] Seleccionar después una matriz pequeña de combinaciones representativas;
  no es necesario ejecutar el producto cartesiano completo.
- [ ] Usar las mismas semillas y presupuesto de la campaña principal, o
  justificar formalmente una campaña exploratoria más pequeña.
- [ ] Medir: objetivo final, gap, desviación, factibilidad, número de switches,
  duración y porcentaje de iteraciones en warm-up.
- [ ] Reportar sensibilidad por dominio, porque las escalas y las dinámicas son
  distintas.
- [ ] Añadir una tabla/figura de sensibilidad o moverla al suplemento.

### R1.5 — Diferencias de efecto entre algoritmos

**Estado:** la discusión actual es descriptiva, pero no alcanza el análisis
solicitado.

El comentario sobre BDE, BPSO y BGWO pertenece al trabajo anterior. Para este
paper, el análisis debe usar los solvers realmente ejecutados en MKP, CEC2022 y
HRES2-H2.

Acciones:

- [ ] Calcular, por solver y dominio, media/mediana, dispersión, gap, número de
  wins/ties/losses y posición en el pool.
- [ ] Analizar qué solvers reciben más activaciones y cuáles suelen preceder o
  seguir a una mejora.
- [ ] Separar poblacionales y de trayectoria.
- [ ] Relacionar las diferencias con mecanismos concretos: diversidad,
  intensificación local, reparación de factibilidad, sensibilidad al
  incumbent e inyección de memoria.
- [ ] No atribuir causalidad solo a los conteos de switches; describirlo como
  evidencia diagnóstica salvo que se haga una ablación controlada.

---

### R2.1 — Estructura del artículo

**Estado:** falta un roadmap explícito.

Añadir al final de la introducción un párrafo que indique, como mínimo:

1. antecedentes y brecha;
2. formulación MKP, CEC2022 y HRES2-H2;
3. monitor DTW/DDTW y orquestador rotacional;
4. protocolo experimental y estadística;
5. resultados por dominio;
6. discusión, limitaciones y conclusiones.

### R2.2 — Ablación de DDTW

**Estado:** parcialmente cubierto.

Ya existe la comparación DTW--DDTW:

- MKP: configuraciones DTW/DDTW con presupuestos de 1K y 3K.
- CEC2022: `analisis_global_dtw_ddtw.py` y campañas de ambas modalidades.
- HRES2-H2: campañas DTW/DDTW y comparación pareada.

Pero esta comparación no separa todos los componentes del framework. Para una
ablación convincente:

- [ ] `DTW` y `DDTW` con el mismo orquestador, memoria, semillas y presupuesto.
- [ ] Control sin monitor, manteniendo el mismo pool y presupuesto total.
- [ ] Rotación fija por período, manteniendo el mismo pool.
- [ ] Rotación aleatoria con la misma cantidad esperada de oportunidades.
- [ ] Framework sin inyección de la solución élite.
- [ ] Si el coste lo permite, diseño factorial pequeño:
  monitor adaptativo sí/no × memoria élite sí/no.
- [ ] Reportar calidad, switches, factibilidad, estabilidad y costo temporal.
- [ ] No llamar “efecto de DDTW” a una diferencia que también cambia la
  configuración temporal o el pool.

### R2.3 — Suficiencia de la rampa y la meseta

**Estado:** implementado en el monitor, pero no suficientemente justificado.

El código usa una referencia lineal y una constante, y calcula `D1`, `D2` y
`Delta`. Estas referencias deben presentarse como un detector mínimo de dos
casos: progreso sostenido y ausencia de progreso.

Acciones:

- [ ] Explicar por qué son útiles como hipótesis nulas operativas y no como una
  descripción exhaustiva de todas las trayectorias.
- [ ] Aclarar cómo se comportan ante mejoras intermitentes, retrocesos y ruido.
- [ ] Documentar la construcción de la pendiente y sus casos límite.
- [ ] Evaluar, al menos en la sensibilidad, cambios en `s_min` y en la
  persistencia.
- [ ] Evitar afirmar que dos referencias son suficientes para caracterizar toda
  dinámica de búsqueda.

### R2.5 — Reproducibilidad

**Estado:** parcialmente cubierto.

Fortalezas existentes:

- semillas base y semillas por corrida (`42 + run_idx` en varias campañas);
- 31 corridas independientes en benchmarks principales;
- CSV/TXT con telemetría detallada;
- scripts de análisis y generación de figuras.

Faltantes:

- [ ] Crear un archivo de configuración canónico o una tabla equivalente que
  permita reconstruir cada campaña.
- [ ] Documentar comandos exactos para MKP, CEC2022 y HRES2-H2.
- [ ] Guardar versiones de Python, dependencias, commit y hardware relevante.
- [ ] Publicar código y resultados, si es posible; si no, adjuntar suplemento
  reproducible con scripts y CSV.
- [ ] Incluir los archivos de entrada CEC2022 y las instancias MKP usadas.
- [ ] Guardar resultados crudos por corrida, no solo resúmenes y p-valores.
- [ ] Explicar qué figuras se generan del mejor run y cuáles agregan las 31
  corridas.

### R2.6 — Abstract

Reescribirlo al final, después de corregir pools, conteos y resultados.

- [ ] Primera frase: presentar el problema de coordinación de solvers, no una
  conclusión de superioridad.
- [ ] Explicar en una frase que DTW/DDTW monitoriza trayectorias y dispara
  rotación con transferencia de memoria.
- [ ] Nombrar los tres dominios.
- [ ] Reportar solo cifras verificadas.
- [ ] Cambiar “significantly outperforming all ten baselines” si la tabla final
  no contiene exactamente diez baselines o si la prueba no es comparable.
- [ ] Eliminar la frase gramaticalmente incompleta sobre la escala del objetivo.
- [ ] Incluir una limitación breve: `D=10` en CEC2022 y un único perfil HRES2-H2,
  si esas decisiones se mantienen.

---

### R3.1 — Brecha de investigación

**Estado:** parcialmente cubierta.

La brecha debe quedar formulada así, o de manera equivalente:

> Existe abundante adaptación de parámetros, selección de operadores y
> hiperheurísticas, pero es menos claro cómo usar una señal temporal y
> solver-agnóstica de la trayectoria de calidad para decidir cuándo transferir
> el control entre solvers heterogéneos sin entrenamiento ni una política
> aprendida.

Después, diferenciar explícitamente:

- **control adaptativo de parámetros:** modifica `F`, `CR`, inercia, temperatura,
  tamaño de vecindad u otros parámetros dentro del solver;
- **selección adaptativa de operadores:** elige operadores o acciones internas;
- **hiperheurística/RL:** aprende o ajusta una política de selección;
- **este trabajo:** detecta patrones de estancamiento en `best-so-far`, termina
  el epoch y rota el solver, transfiriendo el incumbent.

Acciones:

- [ ] Cambiar expresiones que sugieren que el método adapta parámetros internos.
- [ ] Decir explícitamente que no hay entrenamiento ni modelo aprendido.
- [ ] Relacionar la contribución con la complementariedad poblacional--trayectoria.

### R3.2 — Revisión de literatura

**Estado:** parcialmente cubierta.

La sección actual ya trata híbridos, selección adaptativa, aprendizaje y
trayectorias. Agregar o reforzar subtemas sobre:

- control adaptativo de parámetros;
- control difuso aplicado a metaheurísticas;
- aprendizaje por refuerzo para selección de operadores/solvers;
- hiperheurísticas;
- análisis de trayectorias, fitness-distance y detección de estancamiento;
- portfolios y cooperación entre metaheurísticas.

Acciones:

- [ ] Añadir referencias representativas y actuales para cada subtema.
- [ ] Cerrar la sección explicando qué vacío queda después de esos trabajos.
- [ ] No convertir la revisión en una lista de algoritmos sin comparación
  conceptual con el framework propuesto.

### R3.3 — Formulación matemática completa de DDTW

**Estado:** parcialmente cubierta y con una discrepancia importante.

El paper ya incluye ventana, recurrencia, banda, diferencias finitas, `D1`,
`D2`, `Delta`, percentiles y persistencia. Debe alinearse con
`dtw_stagnation.py`:

- [ ] Especificar la convención de minimización/maximización. En MKP se maximiza
  el valor; en CEC/HRES se minimiza y el monitor puede trabajar con el valor
  negado para mantener una lógica común.
- [ ] Especificar exactamente la derivada usada:
  `diff(x, prepend=x[0])`, por lo que la primera diferencia es cero.
- [ ] Explicar las condiciones de frontera de la matriz DTW y la banda.
- [ ] Explicar que la implementación devuelve el costo terminal bruto y no una
  distancia normalizada por longitud de la ruta.
- [ ] Explicar el warm-up y los umbrales iniciales antes de tener diez muestras.
- [ ] Explicar que los percentiles se calculan sobre el historial del epoch.
- [ ] Corregir la afirmación de que el método es “scale invariant” o plenamente
  insensible a la escala: el código actual no aplica normalización min--max.
- [ ] Si se desea sostener una afirmación fuerte de invariancia, implementar la
  normalización y volver a ejecutar las tres campañas. La opción de menor
  riesgo es ajustar el texto al código actual.

### R3.4 y R3.8 — Figuras del monitor y transiciones

**Estado:** la instrumentación existe parcialmente; las figuras del paper no
son suficientes.

Generar una figura representativa por dominio o una figura suplementaria con
los siguientes paneles sincronizados:

1. `best-so-far` y fitness instantáneo;
2. ventana observada junto con referencia de progreso y referencia de meseta;
3. `D1`, `D2` y `Delta`;
4. `theta_c`, `theta_r` y `theta_delta`;
5. estado del monitor, warm-up, condición de estancamiento y `dtw_fire`;
6. solver activo y líneas verticales de transferencia.

Fuentes de datos:

- `HRES2-H2/benchmark_hres2.py` guarda los campos detallados del monitor.
- `hybrid_mkp/batch_benchmark.py` y
  `hybrid_mkp/parallel_mkp_first_inst.py` guardan telemetría equivalente.
- `continuous_benchmark/benchmark_continuo.py` genera trazas y análisis DTW/DDTW;
  comprobar que exporte también umbrales y estado, no solo `Delta`.

Acciones:

- [ ] Crear un exportador/plot común para no dibujar manualmente tres versiones
  incompatibles.
- [ ] Etiquetar claramente DTW y DDTW.
- [ ] Indicar si la traza es el mejor run o un promedio; las figuras actuales
  son principalmente del mejor run.
- [ ] Mostrar transiciones de solver, no “cambios de parámetros”, porque el
  método actual no modifica parámetros internos como resultado del monitor.
- [ ] Si una señal no se conserva para CEC2022, agregarla al CSV antes de
  generar la figura.

### R3.5 — Selección de parámetros

**Estado:** falta documentarla.

El paper debe indicar para cada valor si fue:

- derivado de una regla fija (`band=10%` de la ventana);
- elegido por una restricción computacional;
- fijado antes de los experimentos;
- distinto por presupuesto;
- ajustado usando una campaña piloto.

Acciones:

- [ ] No decir que los parámetros fueron “universales” si en realidad cambian
  entre `run_hres2.py`, `benchmark_hres2.py`, `batch_benchmark.py` y
  `parallel_mkp_first_inst.py`.
- [ ] Declarar cualquier ajuste específico por dominio o presupuesto.
- [ ] Separar selección de parámetros de resultados de sensibilidad.
- [ ] Asegurar que el valor de `s_min` de la ecuación, del código y de la tabla
  sea el mismo o explicar la diferencia.

### R3.6 — Pseudocódigo completo

**Estado:** existe una versión de alto nivel; debe ampliarse.

El algoritmo debe mostrar explícitamente:

1. inicialización del problema, semilla, pool y memoria global;
2. selección del pool y solver activo;
3. inyección del incumbent;
4. actualización del `best-so-far` y del contador de no mejora;
5. warm-up del monitor;
6. construcción de rampa/meseta;
7. cálculo DTW o DDTW, `D1`, `D2`, `Delta`;
8. cálculo de umbrales y condición triple;
9. actualización del streak/patience;
10. registro de telemetría;
11. transferencia de la mejor solución y rotación;
12. condición de parada global.

También debe indicar cómo se trata la minimización frente a la maximización y
qué ocurre cuando aún no hay una ventana de tamaño `W`.

### R3.7 — Interacción según el algoritmo

**Estado:** falta una evaluación sistemática.

No trasladar la explicación del comentario anterior sobre BDE, BPSO y BGWO.
Analizar los algoritmos que realmente participan en cada carpeta:

- MKP: GA, PSO, GWO, WOA, EHO, ACO, SA, TS, ILS y VNS;
- CEC2022: los cinco solvers del pool principal de la campaña;
- HRES2-H2: PSO, GWO, WOA, EHO, ACO, ABC, ILS, SA, TS y VNS.

Para cada dominio:

- [ ] comparar rendimiento standalone contra rendimiento como parte del pool;
- [ ] medir duración media de epochs y frecuencia de selección;
- [ ] medir qué solver suele producir la mejora posterior al cambio;
- [ ] observar si la inyección del incumbent ayuda o reduce diversidad;
- [ ] discutir reparación y factibilidad en MKP y HRES2-H2;
- [ ] separar hipótesis plausibles de resultados causalmente demostrados.

---

## 5. Campañas que faltan ejecutar

### Campaña A — Configuración canónica

- [ ] Resolver primero las discrepancias de pools y parámetros.
- [ ] Congelar una configuración para los resultados principales.
- [ ] Crear un archivo de metadatos por campaña: commit, seed base, seeds,
  `T_max`, `W`, `band`, `s_min`, `K`, `P`, percentiles, modo DTW/DDTW,
  población y criterio de parada.

### Campaña B — Ablación mínima

- [ ] DTW.
- [ ] DDTW.
- [ ] Sin monitor.
- [ ] Rotación fija.
- [ ] Rotación aleatoria.
- [ ] Sin transferencia de memoria elitista.

Todas deben usar el mismo dominio, semillas, pool y presupuesto. Si no se pueden
ejecutar en los tres dominios, priorizar un MKP, una función CEC representativa
y HRES2-H2, y declararlo como ablación parcial.

### Campaña C — Sensibilidad

- [ ] `W` corto/medio/largo.
- [ ] banda automática frente a bandas explícitas.
- [ ] `s_min` automático y valores fijos.
- [ ] `plateau_max` bajo/medio/alto.
- [ ] `patience` bajo/medio/alto.
- [ ] pares de percentiles conservadores y permisivos.
- [ ] DTW frente a DDTW con cada configuración principal.

### Campaña D — Baseline adaptativo externo

- [ ] Solo si se decide responder completamente R1.2/R2.4.
- [ ] Definir el baseline, implementarlo dentro del mismo presupuesto y
  conservar sus decisiones por corrida.
- [ ] Añadirlo a la tabla estadística y a la discusión de alcance.

### Campaña E — Robustez de dominios

- [ ] Opcional: CEC2022 en `D=20`.
- [ ] Opcional: más perfiles meteorológicos o escenarios de HRES2-H2.
- [ ] Si no se hacen, reforzar limitaciones en lugar de extrapolar resultados.

---

## 6. Cambios concretos en `paper_final.tex`

Orden recomendado:

1. **Introducción:** corregir brecha, alcance, diferencia entre rotación de
   solvers y adaptación de parámetros, y añadir roadmap de secciones.
2. **Related work:** añadir control difuso, RL, hiperheurísticas y análisis de
   trayectorias; terminar con la brecha exacta.
3. **Pools y metodología:** describir pools por dominio y no un pool universal
   si el código no lo respalda.
4. **Formulación DTW/DDTW:** alinear ecuaciones con `dtw_stagnation.py`, incluir
   frontera, derivada, warm-up, umbrales iniciales, minimización y falta de
   normalización.
5. **Memoria e interacción:** describir la transferencia real de la solución
   global, sin inventar adaptación de parámetros internos.
6. **Configuración experimental:** reemplazar la tabla actual por la
   configuración real de cada campaña.
7. **Pseudocódigo:** ampliar el algoritmo con todo el ciclo de monitorización y
   rotación.
8. **Resultados:** corregir cantidad de baselines, fuentes, pruebas, unidades y
   afirmaciones de significancia.
9. **Figuras:** añadir señales, umbrales, estados y transiciones sincronizadas.
10. **Discusión:** sustituir hipótesis fuertes por análisis por solver y
    diferenciar evidencia descriptiva de causalidad.
11. **Limitaciones:** mantener explícitamente la falta de sensibilidad/ablación
    si no se ejecutan, además de `D=10` y un único perfil meteorológico.
12. **Abstract y conclusiones:** reescribirlos al final con cifras cerradas y
    afirmaciones proporcionales a la evidencia.

---

## 7. Criterios de cierre

El paper está listo para una nueva revisión cuando se cumpla todo lo siguiente:

- [ ] Cada tabla y figura tiene un archivo de origen y una campaña identificada.
- [ ] Los pools, baselines y cantidades de corridas coinciden entre código,
  resultados y texto.
- [ ] DTW y DDTW están formulados exactamente como se implementan.
- [ ] Se corrigió la afirmación sobre normalización/insensibilidad a escala.
- [ ] Existe una tabla o figura de sensibilidad, o se declara formalmente que
  es trabajo futuro.
- [ ] Existe una ablación suficiente, o el paper deja claro que solo presenta
  comparación DTW--DDTW y no atribuye causalidad completa.
- [ ] Se añadió un baseline adaptativo externo o se limitó la afirmación de
  estado del arte.
- [ ] Las figuras muestran `best-so-far`, referencias, `D1`, `D2`, `Delta`,
  umbrales, estado y cambios de solver.
- [ ] Se eliminaron nombres, cifras y parámetros del trabajo anterior.
- [ ] Se eliminaron placeholders de plantilla, se incluyó la bibliografía real y
  se dejó una declaración de disponibilidad de código/datos verificable.

## Resultado esperado

La conclusión final debe sostener únicamente algo como: el monitor DTW/DDTW
ofrece una señal model-free para coordinar la rotación de un portfolio
heterogéneo en tres tipos de problemas, con evidencia estadística y
limitaciones claramente delimitadas. No debe afirmar que el método es un
controlador adaptativo de parámetros, que supera todo el estado del arte o que
es invariante a la escala si esas propiedades no se demuestran en las campañas
de `hybrid_mkp`, `continuous_benchmark` y `HRES2-H2`.

El método que debe describirse es el framework rotacional: el monitor DTW/DDTW
observa la trayectoria `best-so-far`, detecta estancamiento y provoca el cambio
entre metaheurísticas, transfiriendo la mejor solución global. No debe
presentarse como un método que modifica continuamente los parámetros internos
de una única metaheurística.

---

## 1. Diagnóstico general

### Lo que ya existe en el proyecto

- Un monitor compartido en `dtw_stagnation.py` con DTW, DDTW, banda
  Sakoe--Chiba, referencias de rampa y meseta, percentiles adaptativos y filtro
  de persistencia.
- Un orquestador rotacional para MKP en `hybrid_mkp/orchestrator.py`.
- Un orquestador para CEC2022 en `continuous_benchmark/orchestrator.py`.
- Una extensión del orquestador para HRES2-H2 en `HRES2-H2/orchestrator.py`.
- Campañas de 31 ejecuciones en los scripts de benchmark de MKP y HRES2-H2 y
  en la evaluación de CEC2022.
- Comparaciones DTW frente a DDTW, especialmente en
  `continuous_benchmark/analisis_global_dtw_ddtw.py` y en las campañas finales
  de `resultados finales/`.
- Telemetría del monitor en CSV/TXT con `D1`, `D2`, `Delta`, umbrales, contador
  de no mejora, persistencia, disparo y estado del monitor. Esto aparece en
  `HRES2-H2/benchmark_hres2.py`, `hybrid_mkp/batch_benchmark.py` y
  `hybrid_mkp/parallel_mkp_first_inst.py`.
- Pruebas no paramétricas y generación de estadísticas en los módulos
  `analisis_estadistico.py` de MKP/CEC y en
  `HRES2-H2/analisis_estadistico_hres2.py`.

### Lo que no está demostrado todavía

- No existe una campaña sistemática de sensibilidad de hiperparámetros.
- No existe una ablación completa que separe monitor, rotación, memoria elitista
  y tipo de distancia.
- No existe una comparación clara contra un controlador adaptativo externo
  (control difuso, aprendizaje por refuerzo, selección adaptativa de
  operadores, etc.).
- La configuración del monitor no es única entre los scripts de las tres
  carpetas.
- No está suficientemente documentado qué pool de solvers se utilizó en cada
  dominio.
- Las figuras actuales no muestran todos los estados y señales que el revisor
  solicita.

---

## 2. Correspondencia entre los comentarios y este paper

| Comentario                                      | Estado para este paper                                                  | Decisión                                                                                                                        |
| ----------------------------------------------- | ----------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| R1.1: evidencia estadística de superioridad    | **Parcialmente cubierto**                                         | Rehacer la trazabilidad de cifras y moderar las conclusiones según cada dominio.                                                |
| R1.2: comparar con control adaptativo existente | **Falta**                                                         | Añadir al menos un baseline externo o declarar explícitamente que queda fuera del alcance.                                     |
| R1.3: conjunto experimental reducido            | **No aplica literalmente; la preocupación sigue siendo válida** | Defender los tres dominios y declarar límites: MKP seleccionado, CEC solo`D=10`, HRES con un perfil meteorológico.           |
| R1.4: sensibilidad de hiperparámetros          | **Falta**                                                         | Ejecutar una campaña separada de sensibilidad.                                                                                  |
| R1.5: efectos inconsistentes entre algoritmos   | **Parcialmente cubierto**                                         | Hacer un análisis por solver y por tipo de transición, no solo una discusión general.                                         |
| R2.1: explicar la estructura del artículo      | **Falta**                                                         | Añadir un párrafo al final de la introducción.                                                                                |
| R2.2: ablación que pruebe el efecto de DDTW    | **Parcialmente cubierto**                                         | Usar DTW vs. DDTW existente y añadir controles sin monitor/rotación/memoria.                                                   |
| R2.3: justificar rampa y meseta                 | **Parcialmente cubierto**                                         | Explicar que son referencias mínimas y validar su sensibilidad; no presentarlas como modelo completo de todas las trayectorias. |
| R2.4: baselines adaptativos de estado del arte  | **Falta**                                                         | Igual que R1.2: añadir uno o limitar la afirmación de novedad/superioridad.                                                    |
| R2.5: reproducibilidad                          | **Parcialmente cubierto**                                         | Consolidar configuración, semillas, comandos, datos y telemetría; añadir repositorio o material suplementario.                |
| R2.6: corregir el abstract                      | **Necesario**                                                     | Reescribirlo después de cerrar las cifras y las afirmaciones.                                                                   |
| R3.1: definir la brecha de investigación       | **Parcialmente cubierto**                                         | Diferenciar claramente cambio de solver, control de parámetros e hiperheurísticas.                                             |
| R3.2: fortalecer la literatura                  | **Parcialmente cubierto**                                         | La introducción cubre adaptación, RL y selección; falta reforzar control difuso y análisis de trayectorias.                  |
| R3.3: formulación completa de DDTW             | **Parcialmente cubierto**                                         | La matemática existe, pero debe coincidir exactamente con el código y aclarar la ausencia de normalización.                   |
| R3.4: figuras de señales y estados             | **Parcialmente cubierto**                                         | Los datos existen en parte; falta generar figuras unificadas para el manuscrito.                                                 |
| R3.5: selección de parámetros                 | **Falta documentarlo**                                            | Justificar la configuración y resolver las diferencias entre scripts.                                                           |
| R3.6: pseudocódigo completo                    | **Parcialmente cubierto**                                         | Existe un pseudocódigo de alto nivel; hay que incluir warm-up, umbrales, signo, memoria y rotación.                            |
| R3.7: interacción según el algoritmo          | **Parcialmente cubierto**                                         | Analizar solvers reales de estas tres campañas; no trasladar las conclusiones sobre BDE/BPSO/BGWO.                              |
| R3.8: figuras de transiciones/parámetros       | **Parcialmente cubierto**                                         | Mostrar transiciones entre solvers; los cambios internos de parámetros no corresponden al método actual.                       |

---

## 3. Correcciones prioritarias de coherencia

Estas tareas deben hacerse antes de volver a correr experimentos o interpretar
resultados.

### 3.1 Definir los pools reales por dominio

El manuscrito habla de diez solvers como si el mismo pool se utilizara en todos
los dominios, pero el código muestra una configuración específica por dominio:

- **MKP** (`hybrid_mkp/orchestrator.py`):
  - poblacionales: `GA`, `PSO`, `GWO`, `WOA`, `EHO`, `ACO`;
  - trayectoria: `SA`, `TS`, `ILS`, `VNS`.
- **CEC2022** (`continuous_benchmark/orchestrator.py`): el pool principal
  configurado contiene `WOA`, `PSO`, `EHO`, `GWO` y `ACO`; no debe describirse
  automáticamente como el pool de diez algoritmos.
- **HRES2-H2** (`HRES2-H2/orchestrator.py`):
  - poblacionales: `PSO`, `GWO`, `WOA`, `EHO`, `ACO`, `ABC`;
  - trayectoria: `ILS`, `SA`, `TS`, `VNS`.

Acciones:

- [ ] Decidir si el paper describirá pools específicos por dominio. Esta es la
  opción coherente con el código actual.
- [ ] Revisar todas las menciones a “ten representative solvers”, “all ten
  baselines” y “five standalone methods”.
- [ ] Construir una tabla del paper con: dominio, pool híbrido, baselines
  standalone, presupuesto y número de corridas.
- [ ] Verificar que las tablas y las pruebas estadísticas comparen exactamente
  los algoritmos que se mencionan en el texto.

### 3.2 Unificar la configuración del monitor

Hay diferencias entre los scripts que parecen representar campañas distintas:

- `HRES2-H2/run_hres2.py`: `W=40`, `s_min=0.0`, `K=15`, `P=8`, DTW.
- `HRES2-H2/benchmark_hres2.py`: `W=40`, `s_min=0.0`, `K=15`, `P=3`, DDTW.
- `hybrid_mkp/rotating_benchmark.py`: `W=40`, `s_min=0.5`, `K=15`, `P=8`, DTW.
- `hybrid_mkp/batch_benchmark.py`: `W=40`, `s_min=0.1`, `K=15`, `P=8`, DDTW.
- `hybrid_mkp/parallel_mkp_first_inst.py`: `W=75`, `s_min=0.1`, `K=15`, `P=25`, DDTW.
- `paper_final.tex`: reporta `s_min=0.1` y diferencia `P=8`/`P=25` según
  presupuesto, pero no deja claro qué script generó cada resultado.

Acciones:

- [ ] Crear una tabla maestra de configuración con una fila por campaña real.
- [ ] Asociar cada fila con el script exacto y la carpeta de resultados.
- [ ] Confirmar si las diferencias son deliberadas por dominio/presupuesto o
  si son restos de experimentos previos.
- [ ] Corregir código, resultados o manuscrito hasta que los tres coincidan.
- [ ] Aclarar que `band=0` significa banda automática y reportar el ancho
  efectivo utilizado.
- [ ] Revisar el `patience` específico aplicado en
  `hybrid_mkp/orchestrator.py` para algunos solvers, porque el orquestador
  puede aumentar la paciencia respecto de la configuración global.

### 3.3 Corregir discrepancias visibles del manuscrito

- [ ] En la tabla HRES2-H2, el código ejecuta diez baselines (`PSO`, `GWO`,
  `WOA`, `EHO`, `ACO`, `ABC`, `ILS`, `SA`, `TS`, `VNS`), pero la tabla
  actual muestra nueve. Añadir `ABC` o explicar y recalcular la comparación
  como una campaña de nueve.
- [ ] Revisar las frases “ten comparisons”, “10/0/0” y “all ten standalone
  baselines” después de resolver la tabla anterior.
- [ ] Corregir leyendas de figuras que muestran `ABC` cuando la campaña o la
  figura corresponde a `ACO`.
- [ ] Corregir títulos de figuras DDTW que todavía dicen DTW.
- [ ] Eliminar la frase duplicada en la explicación de los tres buffers de
  memoria.
- [ ] Resolver la contradicción entre la fórmula adaptativa de `s_min` y la
  tabla que lo presenta como fijo en `0.1`.
- [ ] Corregir la oración del abstract sobre “less sensitive to changes in
  objective scale mechanism”.
- [ ] Eliminar el fragmento suelto `microgrid.` que aparece al final de la
  síntesis estadística.
- [ ] Revisar `figures/Diagrama_flujo.png`, porque el paper lo referencia y el
  archivo debe estar disponible para compilar.
- [ ] Reemplazar `\bibliography{sample}` y los textos de plantilla por la
  bibliografía real y una declaración de disponibilidad de datos concreta.

---

## 4. Plan detallado según los revisores

### R1.1 — Evidencia estadística y afirmaciones de superioridad

**Estado:** parcialmente cubierto.

El paper actual ya contiene Friedman, Wilcoxon, Mann--Whitney, Holm y
comparaciones DTW--DDTW. Las tres carpetas también generan análisis estadístico.
El problema pendiente no es añadir pruebas al azar, sino garantizar que cada
cifra tenga una fuente reproducible y que la conclusión no exceda la evidencia.

Acciones:

- [ ] Crear una tabla de trazabilidad: afirmación del paper → CSV/resultado →
  script que lo produjo → número de corridas.
- [ ] Separar claramente las unidades estadísticas:
  instancia MKP, función CEC2022 y corrida HRES2-H2.
- [ ] No reutilizar el `chi^2` ni el `p` del trabajo anterior.
- [ ] Reportar tamaño del efecto o diferencia práctica junto con `p` cuando sea
  posible.
- [ ] Mantener separados los resultados pareados DTW--DDTW y las comparaciones
  independientes contra baselines standalone.
- [ ] Evitar “superioridad universal”, “optimal” o “state-of-the-art” cuando
  solo haya una ventaja promedio o resultados mixtos.
- [ ] Revisar la afirmación de que DTW/DDTW no difieren significativamente en
  todos los dominios y comprobar la corrección de multiplicidad usada en
  cada familia.
- [ ] Preservar los valores crudos por corrida para HRES2-H2; el script
  `benchmark_hres2.py` debe guardar las observaciones de todos los baselines,
  no solo medias y p-valores.

### R1.2 y R2.4 — Comparación con control adaptativo externo

**Estado:** no está implementado en las tres líneas de trabajo.

Los algoritmos standalone no son un baseline de control adaptativo. DTW frente
a DDTW son variantes del propio monitor, no métodos externos de control.

Decisión necesaria:

- [ ] **Opción recomendada si se quiere responder completamente al revisor:**
  implementar al menos un controlador adaptativo externo sencillo y
  reproducible, con el mismo presupuesto, semillas, pools y dominios.
- [ ] **Opción mínima:** no añadirlo y declarar explícitamente que la
  contribución se limita a comparar DTW/DDTW, solvers standalone y controles
  internos; eliminar cualquier frase que sugiera superioridad frente a todo
  el estado del arte adaptativo.

Si se implementa un baseline externo:

- [ ] Definir antes de correrlo qué señal controla, qué parámetros modifica y
  cuál es su presupuesto adicional.
- [ ] Usar las mismas 31 semillas cuando sea viable.
- [ ] Reportar tanto calidad como número de cambios, tiempo y sobrecosto.
- [ ] No comparar un controlador adaptativo con una configuración standalone
  usando presupuestos distintos.

### R1.3 — Cobertura experimental

**Estado:** la crítica literal no corresponde, porque el paper actual no se
limita a MKP. Sí corresponde responder la preocupación sobre generalidad.

La cobertura actual es:

- MKP: nueve instancias Chu--Beasley seleccionadas.
- CEC2022: doce funciones, pero solo dimensión `D=10` en el paper.
- HRES2-H2: una formulación de ingeniería y un perfil meteorológico sintético
  anual fijo.

Acciones:

- [ ] Explicar que los tres dominios fueron elegidos para cubrir optimización
  combinatoria, continua y de ingeniería.
- [ ] Justificar la selección de las nueve instancias MKP y no llamarla una
  cobertura exhaustiva de todos los tamaños posibles.
- [ ] Decidir si se incorporará también `D=20` de CEC2022, ya que existen datos
  de entrada en `continuous_benchmark/input_data/`.
- [ ] Si no se incorpora `D=20`, dejarlo como limitación explícita.
- [ ] Para HRES2-H2, declarar que un solo perfil meteorológico no permite
  generalizar a otros años, precios, demandas o escenarios de incertidumbre.
- [ ] No presentar los tres dominios como prueba de robustez universal.

### R1.4 — Sensibilidad de hiperparámetros

**Estado:** falta la campaña experimental.

Los scripts permiten configurar los parámetros, pero que un parámetro sea
editable no constituye un estudio de sensibilidad.

Parámetros mínimos:

- longitud de ventana `W`;
- banda Sakoe--Chiba `band`;
- pendiente mínima `s_min`;
- tolerancia de meseta `plateau_max`;
- persistencia `patience`;
- percentiles `p_low` y `p_high`;
- DTW frente a DDTW;
- activación/desactivación de umbrales adaptativos.

Diseño sugerido:

- [ ] Primero fijar una configuración canónica por presupuesto.
- [ ] Hacer un análisis de un factor a la vez para detectar parámetros críticos.
- [ ] Seleccionar después una matriz pequeña de combinaciones representativas;
  no es necesario ejecutar el producto cartesiano completo.
- [ ] Usar las mismas semillas y presupuesto de la campaña principal, o
  justificar formalmente una campaña exploratoria más pequeña.
- [ ] Medir: objetivo final, gap, desviación, factibilidad, número de switches,
  duración y porcentaje de iteraciones en warm-up.
- [ ] Reportar sensibilidad por dominio, porque las escalas y las dinámicas son
  distintas.
- [ ] Añadir una tabla/figura de sensibilidad o moverla al suplemento.

### R1.5 — Diferencias de efecto entre algoritmos

**Estado:** la discusión actual es descriptiva, pero no alcanza el análisis
solicitado.

El comentario sobre BDE, BPSO y BGWO pertenece al trabajo anterior. Para este
paper, el análisis debe usar los solvers realmente ejecutados en MKP, CEC2022 y
HRES2-H2.

Acciones:

- [ ] Calcular, por solver y dominio, media/mediana, dispersión, gap, número de
  wins/ties/losses y posición en el pool.
- [ ] Analizar qué solvers reciben más activaciones y cuáles suelen preceder o
  seguir a una mejora.
- [ ] Separar poblacionales y de trayectoria.
- [ ] Relacionar las diferencias con mecanismos concretos: diversidad,
  intensificación local, reparación de factibilidad, sensibilidad al
  incumbent e inyección de memoria.
- [ ] No atribuir causalidad solo a los conteos de switches; describirlo como
  evidencia diagnóstica salvo que se haga una ablación controlada.

---

### R2.1 — Estructura del artículo

**Estado:** falta un roadmap explícito.

Añadir al final de la introducción un párrafo que indique, como mínimo:

1. antecedentes y brecha;
2. formulación MKP, CEC2022 y HRES2-H2;
3. monitor DTW/DDTW y orquestador rotacional;
4. protocolo experimental y estadística;
5. resultados por dominio;
6. discusión, limitaciones y conclusiones.

### R2.2 — Ablación de DDTW

**Estado:** parcialmente cubierto.

Ya existe la comparación DTW--DDTW:

- MKP: configuraciones DTW/DDTW con presupuestos de 1K y 3K.
- CEC2022: `analisis_global_dtw_ddtw.py` y campañas de ambas modalidades.
- HRES2-H2: campañas DTW/DDTW y comparación pareada.

Pero esta comparación no separa todos los componentes del framework. Para una
ablación convincente:

- [ ] `DTW` y `DDTW` con el mismo orquestador, memoria, semillas y presupuesto.
- [ ] Control sin monitor, manteniendo el mismo pool y presupuesto total.
- [ ] Rotación fija por período, manteniendo el mismo pool.
- [ ] Rotación aleatoria con la misma cantidad esperada de oportunidades.
- [ ] Framework sin inyección de la solución élite.
- [ ] Si el coste lo permite, diseño factorial pequeño:
  monitor adaptativo sí/no × memoria élite sí/no.
- [ ] Reportar calidad, switches, factibilidad, estabilidad y costo temporal.
- [ ] No llamar “efecto de DDTW” a una diferencia que también cambia la
  configuración temporal o el pool.

### R2.3 — Suficiencia de la rampa y la meseta

**Estado:** implementado en el monitor, pero no suficientemente justificado.

El código usa una referencia lineal y una constante, y calcula `D1`, `D2` y
`Delta`. Estas referencias deben presentarse como un detector mínimo de dos
casos: progreso sostenido y ausencia de progreso.

Acciones:

- [ ] Explicar por qué son útiles como hipótesis nulas operativas y no como una
  descripción exhaustiva de todas las trayectorias.
- [ ] Aclarar cómo se comportan ante mejoras intermitentes, retrocesos y ruido.
- [ ] Documentar la construcción de la pendiente y sus casos límite.
- [ ] Evaluar, al menos en la sensibilidad, cambios en `s_min` y en la
  persistencia.
- [ ] Evitar afirmar que dos referencias son suficientes para caracterizar toda
  dinámica de búsqueda.

### R2.5 — Reproducibilidad

**Estado:** parcialmente cubierto.

Fortalezas existentes:

- semillas base y semillas por corrida (`42 + run_idx` en varias campañas);
- 31 corridas independientes en benchmarks principales;
- CSV/TXT con telemetría detallada;
- scripts de análisis y generación de figuras.

Faltantes:

- [ ] Crear un archivo de configuración canónico o una tabla equivalente que
  permita reconstruir cada campaña.
- [ ] Documentar comandos exactos para MKP, CEC2022 y HRES2-H2.
- [ ] Guardar versiones de Python, dependencias, commit y hardware relevante.
- [ ] Publicar código y resultados, si es posible; si no, adjuntar suplemento
  reproducible con scripts y CSV.
- [ ] Incluir los archivos de entrada CEC2022 y las instancias MKP usadas.
- [ ] Guardar resultados crudos por corrida, no solo resúmenes y p-valores.
- [ ] Explicar qué figuras se generan del mejor run y cuáles agregan las 31
  corridas.

### R2.6 — Abstract

Reescribirlo al final, después de corregir pools, conteos y resultados.

- [ ] Primera frase: presentar el problema de coordinación de solvers, no una
  conclusión de superioridad.
- [ ] Explicar en una frase que DTW/DDTW monitoriza trayectorias y dispara
  rotación con transferencia de memoria.
- [ ] Nombrar los tres dominios.
- [ ] Reportar solo cifras verificadas.
- [ ] Cambiar “significantly outperforming all ten baselines” si la tabla final
  no contiene exactamente diez baselines o si la prueba no es comparable.
- [ ] Eliminar la frase gramaticalmente incompleta sobre la escala del objetivo.
- [ ] Incluir una limitación breve: `D=10` en CEC2022 y un único perfil HRES2-H2,
  si esas decisiones se mantienen.

---

### R3.1 — Brecha de investigación

**Estado:** parcialmente cubierta.

La brecha debe quedar formulada así, o de manera equivalente:

> Existe abundante adaptación de parámetros, selección de operadores y
> hiperheurísticas, pero es menos claro cómo usar una señal temporal y
> solver-agnóstica de la trayectoria de calidad para decidir cuándo transferir
> el control entre solvers heterogéneos sin entrenamiento ni una política
> aprendida.

Después, diferenciar explícitamente:

- **control adaptativo de parámetros:** modifica `F`, `CR`, inercia, temperatura,
  tamaño de vecindad u otros parámetros dentro del solver;
- **selección adaptativa de operadores:** elige operadores o acciones internas;
- **hiperheurística/RL:** aprende o ajusta una política de selección;
- **este trabajo:** detecta patrones de estancamiento en `best-so-far`, termina
  el epoch y rota el solver, transfiriendo el incumbent.

Acciones:

- [ ] Cambiar expresiones que sugieren que el método adapta parámetros internos.
- [ ] Decir explícitamente que no hay entrenamiento ni modelo aprendido.
- [ ] Relacionar la contribución con la complementariedad poblacional--trayectoria.

### R3.2 — Revisión de literatura

**Estado:** parcialmente cubierta.

La sección actual ya trata híbridos, selección adaptativa, aprendizaje y
trayectorias. Agregar o reforzar subtemas sobre:

- control adaptativo de parámetros;
- control difuso aplicado a metaheurísticas;
- aprendizaje por refuerzo para selección de operadores/solvers;
- hiperheurísticas;
- análisis de trayectorias, fitness-distance y detección de estancamiento;
- portfolios y cooperación entre metaheurísticas.

Acciones:

- [ ] Añadir referencias representativas y actuales para cada subtema.
- [ ] Cerrar la sección explicando qué vacío queda después de esos trabajos.
- [ ] No convertir la revisión en una lista de algoritmos sin comparación
  conceptual con el framework propuesto.

### R3.3 — Formulación matemática completa de DDTW

**Estado:** parcialmente cubierta y con una discrepancia importante.

El paper ya incluye ventana, recurrencia, banda, diferencias finitas, `D1`,
`D2`, `Delta`, percentiles y persistencia. Debe alinearse con
`dtw_stagnation.py`:

- [ ] Especificar la convención de minimización/maximización. En MKP se maximiza
  el valor; en CEC/HRES se minimiza y el monitor puede trabajar con el valor
  negado para mantener una lógica común.
- [ ] Especificar exactamente la derivada usada:
  `diff(x, prepend=x[0])`, por lo que la primera diferencia es cero.
- [ ] Explicar las condiciones de frontera de la matriz DTW y la banda.
- [ ] Explicar que la implementación devuelve el costo terminal bruto y no una
  distancia normalizada por longitud de la ruta.
- [ ] Explicar el warm-up y los umbrales iniciales antes de tener diez muestras.
- [ ] Explicar que los percentiles se calculan sobre el historial del epoch.
- [ ] Corregir la afirmación de que el método es “scale invariant” o plenamente
  insensible a la escala: el código actual no aplica normalización min--max.
- [ ] Si se desea sostener una afirmación fuerte de invariancia, implementar la
  normalización y volver a ejecutar las tres campañas. La opción de menor
  riesgo es ajustar el texto al código actual.

### R3.4 y R3.8 — Figuras del monitor y transiciones

**Estado:** la instrumentación existe parcialmente; las figuras del paper no
son suficientes.

Generar una figura representativa por dominio o una figura suplementaria con
los siguientes paneles sincronizados:

1. `best-so-far` y fitness instantáneo;
2. ventana observada junto con referencia de progreso y referencia de meseta;
3. `D1`, `D2` y `Delta`;
4. `theta_c`, `theta_r` y `theta_delta`;
5. estado del monitor, warm-up, condición de estancamiento y `dtw_fire`;
6. solver activo y líneas verticales de transferencia.

Fuentes de datos:

- `HRES2-H2/benchmark_hres2.py` guarda los campos detallados del monitor.
- `hybrid_mkp/batch_benchmark.py` y
  `hybrid_mkp/parallel_mkp_first_inst.py` guardan telemetría equivalente.
- `continuous_benchmark/benchmark_continuo.py` genera trazas y análisis DTW/DDTW;
  comprobar que exporte también umbrales y estado, no solo `Delta`.

Acciones:

- [ ] Crear un exportador/plot común para no dibujar manualmente tres versiones
  incompatibles.
- [ ] Etiquetar claramente DTW y DDTW.
- [ ] Indicar si la traza es el mejor run o un promedio; las figuras actuales
  son principalmente del mejor run.
- [ ] Mostrar transiciones de solver, no “cambios de parámetros”, porque el
  método actual no modifica parámetros internos como resultado del monitor.
- [ ] Si una señal no se conserva para CEC2022, agregarla al CSV antes de
  generar la figura.

### R3.5 — Selección de parámetros

**Estado:** falta documentarla.

El paper debe indicar para cada valor si fue:

- derivado de una regla fija (`band=10%` de la ventana);
- elegido por una restricción computacional;
- fijado antes de los experimentos;
- distinto por presupuesto;
- ajustado usando una campaña piloto.

Acciones:

- [ ] No decir que los parámetros fueron “universales” si en realidad cambian
  entre `run_hres2.py`, `benchmark_hres2.py`, `batch_benchmark.py` y
  `parallel_mkp_first_inst.py`.
- [ ] Declarar cualquier ajuste específico por dominio o presupuesto.
- [ ] Separar selección de parámetros de resultados de sensibilidad.
- [ ] Asegurar que el valor de `s_min` de la ecuación, del código y de la tabla
  sea el mismo o explicar la diferencia.

### R3.6 — Pseudocódigo completo

**Estado:** existe una versión de alto nivel; debe ampliarse.

El algoritmo debe mostrar explícitamente:

1. inicialización del problema, semilla, pool y memoria global;
2. selección del pool y solver activo;
3. inyección del incumbent;
4. actualización del `best-so-far` y del contador de no mejora;
5. warm-up del monitor;
6. construcción de rampa/meseta;
7. cálculo DTW o DDTW, `D1`, `D2`, `Delta`;
8. cálculo de umbrales y condición triple;
9. actualización del streak/patience;
10. registro de telemetría;
11. transferencia de la mejor solución y rotación;
12. condición de parada global.

También debe indicar cómo se trata la minimización frente a la maximización y
qué ocurre cuando aún no hay una ventana de tamaño `W`.

### R3.7 — Interacción según el algoritmo

**Estado:** falta una evaluación sistemática.

No trasladar la explicación del comentario anterior sobre BDE, BPSO y BGWO.
Analizar los algoritmos que realmente participan en cada carpeta:

- MKP: GA, PSO, GWO, WOA, EHO, ACO, SA, TS, ILS y VNS;
- CEC2022: los cinco solvers del pool principal de la campaña;
- HRES2-H2: PSO, GWO, WOA, EHO, ACO, ABC, ILS, SA, TS y VNS.

Para cada dominio:

- [ ] comparar rendimiento standalone contra rendimiento como parte del pool;
- [ ] medir duración media de epochs y frecuencia de selección;
- [ ] medir qué solver suele producir la mejora posterior al cambio;
- [ ] observar si la inyección del incumbent ayuda o reduce diversidad;
- [ ] discutir reparación y factibilidad en MKP y HRES2-H2;
- [ ] separar hipótesis plausibles de resultados causalmente demostrados.

---

## 5. Campañas que faltan ejecutar

### Campaña A — Configuración canónica

- [ ] Resolver primero las discrepancias de pools y parámetros.
- [ ] Congelar una configuración para los resultados principales.
- [ ] Crear un archivo de metadatos por campaña: commit, seed base, seeds,
  `T_max`, `W`, `band`, `s_min`, `K`, `P`, percentiles, modo DTW/DDTW,
  población y criterio de parada.

### Campaña B — Ablación mínima

- [ ] DTW.
- [ ] DDTW.
- [ ] Sin monitor.
- [ ] Rotación fija.
- [ ] Rotación aleatoria.
- [ ] Sin transferencia de memoria elitista.

Todas deben usar el mismo dominio, semillas, pool y presupuesto. Si no se pueden
ejecutar en los tres dominios, priorizar un MKP, una función CEC representativa
y HRES2-H2, y declararlo como ablación parcial.

### Campaña C — Sensibilidad

- [ ] `W` corto/medio/largo.
- [ ] banda automática frente a bandas explícitas.
- [ ] `s_min` automático y valores fijos.
- [ ] `plateau_max` bajo/medio/alto.
- [ ] `patience` bajo/medio/alto.
- [ ] pares de percentiles conservadores y permisivos.
- [ ] DTW frente a DDTW con cada configuración principal.

### Campaña D — Baseline adaptativo externo

- [ ] Solo si se decide responder completamente R1.2/R2.4.
- [ ] Definir el baseline, implementarlo dentro del mismo presupuesto y
  conservar sus decisiones por corrida.
- [ ] Añadirlo a la tabla estadística y a la discusión de alcance.

### Campaña E — Robustez de dominios

- [ ] Opcional: CEC2022 en `D=20`.
- [ ] Opcional: más perfiles meteorológicos o escenarios de HRES2-H2.
- [ ] Si no se hacen, reforzar limitaciones en lugar de extrapolar resultados.

---

## 6. Cambios concretos en `paper_final.tex`

Orden recomendado:

1. **Introducción:** corregir brecha, alcance, diferencia entre rotación de
   solvers y adaptación de parámetros, y añadir roadmap de secciones.
2. **Related work:** añadir control difuso, RL, hiperheurísticas y análisis de
   trayectorias; terminar con la brecha exacta.
3. **Pools y metodología:** describir pools por dominio y no un pool universal
   si el código no lo respalda.
4. **Formulación DTW/DDTW:** alinear ecuaciones con `dtw_stagnation.py`, incluir
   frontera, derivada, warm-up, umbrales iniciales, minimización y falta de
   normalización.
5. **Memoria e interacción:** describir la transferencia real de la solución
   global, sin inventar adaptación de parámetros internos.
6. **Configuración experimental:** reemplazar la tabla actual por la
   configuración real de cada campaña.
7. **Pseudocódigo:** ampliar el algoritmo con todo el ciclo de monitorización y
   rotación.
8. **Resultados:** corregir cantidad de baselines, fuentes, pruebas, unidades y
   afirmaciones de significancia.
9. **Figuras:** añadir señales, umbrales, estados y transiciones sincronizadas.
10. **Discusión:** sustituir hipótesis fuertes por análisis por solver y
    diferenciar evidencia descriptiva de causalidad.
11. **Limitaciones:** mantener explícitamente la falta de sensibilidad/ablación
    si no se ejecutan, además de `D=10` y un único perfil meteorológico.
12. **Abstract y conclusiones:** reescribirlos al final con cifras cerradas y
    afirmaciones proporcionales a la evidencia.

---

## 7. Criterios de cierre

El paper está listo para una nueva revisión cuando se cumpla todo lo siguiente:

- [ ] Cada tabla y figura tiene un archivo de origen y una campaña identificada.
- [ ] Los pools, baselines y cantidades de corridas coinciden entre código,
  resultados y texto.
- [ ] DTW y DDTW están formulados exactamente como se implementan.
- [ ] Se corrigió la afirmación sobre normalización/insensibilidad a escala.
- [ ] Existe una tabla o figura de sensibilidad, o se declara formalmente que
  es trabajo futuro.
- [ ] Existe una ablación suficiente, o el paper deja claro que solo presenta
  comparación DTW--DDTW y no atribuye causalidad completa.
- [ ] Se añadió un baseline adaptativo externo o se limitó la afirmación de
  estado del arte.
- [ ] Las figuras muestran `best-so-far`, referencias, `D1`, `D2`, `Delta`,
  umbrales, estado y cambios de solver.
- [ ] Se eliminaron nombres, cifras y parámetros del trabajo anterior.
- [ ] Se eliminaron placeholders de plantilla, se incluyó la bibliografía real y
  se dejó una declaración de disponibilidad de código/datos verificable.

## Resultado esperado

La conclusión final debe sostener únicamente algo como: el monitor DTW/DDTW
ofrece una señal model-free para coordinar la rotación de un portfolio
heterogéneo en tres tipos de problemas, con evidencia estadística y
limitaciones claramente delimitadas. No debe afirmar que el método es un
controlador adaptativo de parámetros, que supera todo el estado del arte o que
es invariante a la escala si esas propiedades no se demuestran en las campañas
de `hybrid_mkp`, `continuous_benchmark` y `HRES2-H2`.
