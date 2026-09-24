# Guion para la presentación

## Duración y mensaje general

Este guion está pensado para unos 20 minutos, dejando dos o tres minutos al final para preguntas. La idea no es leer todas las ecuaciones, sino explicar qué problema se resuelve, por qué DTW/DDTW es útil y cómo el detector se conecta con el dimensionamiento HRES2--H$_2$.

El texto principal tiene aproximadamente 3.200 palabras: a un ritmo de 150--160 palabras por minuto dura cerca de 20 minutos. Las secciones “Explicación corta...” y “Tres precauciones...” son apoyo para preguntas y no forman parte de la lectura principal.

El hilo conductor es:

> El problema no es solamente encontrar una buena solución, sino decidir cuándo la metaheurística dejó de progresar. La trayectoria temporal del mejor fitness permite detectar ese momento y coordinar otro solucionador. En HRES2--H$_2$, cada candidato se valida mediante un despacho horario de 8.760 horas.

## Distribución sugerida del tiempo

| Parte | Tiempo aproximado |
|---|---:|
| Contexto, problema y brecha | 4 min |
| DTW/DDTW y arquitectura | 5 min 30 s |
| Modelo HRES2--H$_2$ y resultados | 7 min 15 s |
| Posicionamiento, plan y cierre | 3 min |
| **Total** | **20 min** |

## Diccionario de variables para explicar el guion

Antes de memorizar el texto conviene tener clara la diferencia entre tres grupos de variables: las que el optimizador **elige**, las que el simulador **calcula** y las que DTW/DDTW **usa para observar la búsqueda**. Una frase que ayuda a no confundirse es:

> El optimizador decide capacidades; el HRES calcula la operación; DTW calcula si la búsqueda sigue progresando.

### 1. Variables que decide el optimizador

| Variable | Qué significa | Cómo explicarla oralmente |
|---|---|---|
| `x` | Vector numérico que recibe el optimizador | “Es la propuesta codificada que el decodificador transforma en equipos reales.” |
| `z` | Diseño físico ya decodificado | `z = [P_WT, N_el, P_bat, tau_bat]`: las capacidades y opciones que efectivamente evalúa el simulador. |
| `P_WT` | Potencia eólica instalada, en MW | “Cuánta capacidad eólica construyo.” Es continua entre 0 y 200 MW. |
| `P_PV` | Potencia fotovoltaica instalada, en MW | No es independiente: `P_PV = 200 - P_WT`, porque la capacidad renovable total se fija en 200 MW. |
| `N_el` | Número de módulos del electrolizador PEM | Es entero, entre 10 y 20. Cada módulo aporta 5 MW. |
| `P_el` | Potencia total del electrolizador | Se calcula como `P_el = 5 × N_el`. Por ejemplo, 14 módulos representan 70 MW. |
| `P_bat` | Potencia máxima de la batería, en MW | Es el límite de potencia de carga o descarga. Se usa de 0 a 50 MW, en pasos de 5 MW; la energía se expresa en MWh. |
| `tau_bat` | Duración nominal de la batería, en horas | Indica cuántas horas puede sostener su potencia nominal: 1, 2 o 4 horas. |
| `E_bat` | Capacidad energética de la batería, en MWh | Se calcula como `E_bat = P_bat × tau_bat`. Una batería de 50 MW y 4 h tiene 200 MWh. |
| `P_el <= 100 MW` | Límite superior del electrolizador | Es 50 % de los 200 MW renovables; equivale a un máximo de 20 módulos de 5 MW. |

**Detalle importante de la implementación:** el vector numérico es `x = [x_0,x_1,x_2,x_3]`. `x_0` es la potencia eólica; `x_1` se redondea a módulos enteros; `x_2` se redondea a pasos de 5 MW; y `x_3` es un índice de la lista `[1,2,4]` horas. La cuarta coordenada no es directamente la duración: por ejemplo, `x_3=2` se decodifica como `tau_bat=4 h`. En la explicación oral usamos `z`, el diseño físico después de decodificar.

### 2. Variables físicas que calcula el HRES

| Variable | Qué significa | Cómo explicarla oralmente |
|---|---|---|
| `t` | Índice de hora | “La simulación recorre `t = 1, ..., 8760`, es decir, un año.” |
| `v(t)` | Velocidad del viento en la hora `t` | Determina qué fracción de la potencia eólica se produce. |
| `v_i`, `v_r`, `v_o` | Velocidades de arranque, nominal y corte | En el caso usado son 2.5, 10.5 y 25 m/s. |
| `g(t)` | Irradiancia horaria del perfil `ghi_kwh_m2` | En intervalos de 1 h, su valor es numéricamente equivalente a kW/m² y determina la generación fotovoltaica. |
| `T_a(t)` | Temperatura ambiente | Se usa para estimar la temperatura de la celda solar. |
| `T_c(t)` | Temperatura de la celda fotovoltaica | Es mayor que la ambiente cuando hay irradiancia y afecta el rendimiento. |
| `NOCT` | Temperatura nominal de operación de la celda | En el modelo vale 47 °C. |
| `gamma_T` | Coeficiente de temperatura de la PV | Vale -0.5 % por °C: al aumentar la temperatura, baja la eficiencia. |
| `f_der` | Factor de pérdidas o derating de la PV | Vale 0.90; representa pérdidas del sistema. |
| `f_WT(t)`, `f_PV(t)` | Factores de producción horario normalizados | Fracciones entre 0 y 1 aplicadas a la capacidad instalada de cada tecnología. |
| `P_WT × f_WT(t)` | Potencia eólica generada en hora `t` | `P_WT` es la capacidad instalada en MW; `f_WT(t)` es su perfil normalizado. |
| `P_PV × f_PV(t)` | Potencia solar generada en hora `t` | `P_PV` es la capacidad instalada en MW; `f_PV(t)` incorpora irradiancia, temperatura y pérdidas. |
| `P_gen(t)` | Potencia renovable total disponible, en MW | `P_WT f_WT(t) + P_PV f_PV(t)`. No es energía; energía se obtiene multiplicando por `Δt`. |
| `P_el,min` | Carga mínima del electrolizador | Es `0.30 × P_el`; con 70 MW equivale a 21 MW. |
| `P_el,usada(t)` | Potencia realmente entregada al electrolizador | Si `P_gen(t) >= P_el,min`, es `min(P_gen(t),P_el)`; bajo ese umbral se aplica la regla de respaldo de batería. |
| `P_dis,max(t)` | Potencia máxima descargable durante esa hora | `min(P_bat, SOC(t) × eta_dis / Δt)`; el código usa `Δt=1 h`. |
| `SOC(t)` | Energía que queda almacenada en la batería | Se mide en MWh, no en porcentaje. Comienza en 0. |
| `SOC_max` | Capacidad máxima del estado de carga | Coincide con `P_bat × tau_bat`. |
| `eta_ch`, `eta_dis` | Eficiencias de carga y descarga | Se calculan como `sqrt(0.90)`, aproximadamente 0.9487, para obtener 90 % de eficiencia round-trip. |
| `P_ch(t)`, `P_dis(t)` | Potencias horarias de carga y descarga | Se miden en MW y actualizan el SOC considerando las eficiencias. |
| `Δt` | Duración de cada intervalo | Es 1 h; permite convertir potencia (MW) a energía (MWh). |
| `P_grid_sales(t)` | Excedente vendido a la red en la hora `t` | Es lo que queda después de atender al PEM y cargar la batería. |

La secuencia física para explicar la lámina es: “Primero sumo viento y PV. Luego alimento el electrolizador si alcanzo su mínimo. Si falta poco, la batería respalda; si sobra, la batería se carga y el resto se vende a la red.”

### 3. Variables de producción y economía

| Variable | Qué significa | Cómo explicarla oralmente |
|---|---|---|
| `E_el,annual` | Energía anual que realmente recibió el electrolizador | Se obtiene sumando la potencia usada por el PEM durante las 8.760 horas. |
| `E_grid,sales,annual` | Energía anual vendida a la red | Es la suma de todos los excedentes horarios. |
| `E_entregada,annual` | Energía usada en el denominador del LCOE | `1000 × (E_el,annual + E_grid,sales,annual)` kWh/año; las dos energías del paréntesis están en MWh/año. |
| `eta_el` | Eficiencia del electrolizador | Vale 0.75; representa la conversión de electricidad en energía química. |
| `HHV_H2` | Poder calorífico superior del hidrógeno | Vale 39.4 kWh/kg y permite convertir energía en kilogramos de H₂. |
| `m_H2` | Masa anual de hidrógeno producida | `m_H2 = E_el,annual × 1000 × eta_el / HHV_H2`. El 1000 convierte MWh a kWh. |
| `NPC` | Net Present Cost | Suma descontada de CAPEX, reemplazos y operación/mantenimiento. |
| `P_j` | Capacidad de la tecnología `j` | En MW; `j` recorre eólica, PV, electrolizador y batería. Para batería el costo se aplica a MW de potencia. |
| `C_cap,j`, `C_rep,j`, `C_OM,j` | Costos unitarios por tecnología | CAPEX y reemplazo en CNY/kW; O&M en CNY/(kW·año). `1000 × P_j` convierte MW a kW. |
| `R_j`, `y` | Años de reemplazo y año de un flujo | Cada costo futuro se descuenta por `(1+r)^y`; `R_j` depende de la vida útil del equipo. |
| `r` | Tasa real de descuento | En el caso de estudio es 4.35 %. |
| `N` | Vida económica del proyecto | Es de 25 años. |
| `CRF(r,N)` | Factor de recuperación de capital | `r(1+r)^N/[(1+r)^N-1]`; convierte el NPC en costo anual equivalente. |
| `LCOE` | Costo nivelado de electricidad, CNY/kWh | `LCOE = NPC × CRF / E_entregada,annual`; objetivo principal que se minimiza. |
| `LCOH` | Costo nivelado del hidrógeno, CNY/kg | `LCOH = costo anualizado / m_H2`; en esta versión es indicador secundario. |
| `E_ren,annual` | Generación renovable anual, MWh/año | Suma horaria de `P_gen(t) × Δt`. |
| `AGSR` | Annual Grid Surplus Ratio | `E_grid,sales,annual / E_ren,annual`; es adimensional y debe ser `<=0.20`. |
| `f_pen` | Fitness artificial para una solución inviable | Si `AGSR>0.20`, el código devuelve `100 + 10 × AGSR` en lugar del LCOE. |

Cuando aparezca LCOE, conviene decir explícitamente: “No es el costo de una hora; es un costo anual equivalente dividido por la energía anual entregada.” Cuando aparezca LCOH: “Es cuánto cuesta producir cada kilogramo, pero actualmente no es el objetivo que dirige la búsqueda.”

### 4. Variables que usa DTW/DDTW

| Variable | Qué significa | Cómo explicarla oralmente |
|---|---|---|
| `f_t` | Mejor fitness acumulado en la iteración `t` | Es la trayectoria que el detector observa. En HRES, como se minimiza LCOE, el valor mejora cuando baja. |
| `gbest` o incumbente | Mejor solución global encontrada | Es el vector de diseño y su valor objetivo que se conserva al cambiar de algoritmo. |
| `X` | Ventana reciente de la trayectoria | Contiene los últimos 40 valores del fitness. |
| `W` o `window` | Tamaño de la ventana | En HRES2 vale 40. El detector espera tener 40 observaciones recientes. |
| `Y` | Trayectoria de referencia | Puede ser una rampa de progreso o una constante de estancamiento. |
| `d(x_i,y_j)` | Costo local entre dos puntos | Se calcula como `abs(x_i - y_j)`. |
| `D(i,j)` | Costo acumulado hasta una celda de la matriz DTW | Guarda el mejor camino de alineamiento encontrado hasta ese punto. |
| `band` | Banda de Sakoe–Chiba | Limita cuánto puede deformarse el alineamiento. En el código 0 significa una banda automática cercana al 10 % de la ventana. |
| `D1` | Distancia de la ventana a la rampa | Si es alta, la trayectoria se aleja del patrón de progreso. |
| `D2` | Distancia de la ventana a la constante | Si es baja, la trayectoria se parece a una meseta. |
| `Delta` | Diferencia entre las distancias | `Delta = D1 - D2`; ayuda a separar progreso de estancamiento. |
| `theta_c` | Umbral para considerar cercana la constante | Se adapta usando el percentil 30 del historial de `D2`. |
| `theta_r` | Umbral para considerar alejada la rampa | Se adapta usando el percentil 70 del historial de `D1`. |
| `theta_Delta` | Umbral para la diferencia de distancias | Se adapta usando el percentil 70 del historial de `Delta`. |
| `plateau_max` | Máximo de iteraciones sin mejora | En HRES2 vale 15. |
| `patience` | Persistencia necesaria para disparar | En HRES2 vale 3 confirmaciones consecutivas. |
| `trigger_streak` | Contador de confirmaciones consecutivas | Se incrementa si se cumplen las tres condiciones y vuelve a cero si el patrón desaparece. |
| `x'_t` | Cambio aproximado entre puntos consecutivos | En DDTW se calcula como `x_t - x_(t-1)` para comparar pendientes. |
| `epoch` | Bloque de iteraciones de un solucionador | Cuando una época se estanca, el orquestador puede cambiar de metaheurística. |

**Detalle crucial para el discurso:** el monitor genérico espera que un valor mayor sea mejor. Como HRES minimiza LCOE, el código le entrega `-mejor_lcoe`. Si el LCOE baja de 0.28 a 0.27, el monitor ve que el valor sube de -0.28 a -0.27 y lo interpreta correctamente como progreso.

### 5. Texto listo para decir en la diapositiva 19

“En esta diapositiva se muestran las cuatro decisiones del dimensionamiento. `P_WT` es la potencia eólica instalada. La potencia fotovoltaica no se elige de manera independiente, porque siempre completa los 200 MW renovables: `P_PV = 200 - P_WT`. `N_el` es el número entero de módulos PEM y cada módulo aporta 5 MW, por lo que 14 módulos representan 70 MW de electrólisis. Finalmente, `P_bat` es la potencia máxima de la batería y `tau_bat` es su duración; multiplicándolos obtengo la capacidad energética disponible.

El diagrama muestra lo que ocurre después de elegir esas variables. El viento y el sol forman la generación renovable. Esa generación alimenta primero al electrolizador. Si hay excedente, se carga la batería y lo que todavía sobra se vende a la red. Si hay una caída temporal, la batería respalda la carga mínima del electrolizador. El producto final de la operación es hidrógeno verde y el indicador económico se calcula después de repetir este despacho durante las 8.760 horas.”

## Guion por diapositiva

### 1. Portada — 20 segundos

“Buenos días. Soy Lucas Erazo y presentaré el desarrollo de arquitecturas híbridas para la optimización de un sistema HRES2--H$_2$ con tecnologías de hidrógeno verde. El foco de esta presentación es mostrar cómo usamos DTW y DDTW para coordinar metaheurísticas y cómo esa coordinación se integra con un modelo energético horario.”

No explicar todavía los resultados. Solo dejar claro que se hablará de un método de optimización y de un caso de estudio energético.

### 2. Estructura de la presentación — 15 segundos

“Primero presentaré el problema de coordinación entre metaheurísticas. Luego explicaré DTW y DDTW, incluyendo cómo se transforma una trayectoria de fitness en una señal de decisión. Después describiré el modelo HRES2--H$_2$, sus variables, el despacho horario y los resultados obtenidos. Finalmente comentaré el posicionamiento y el plan de trabajo.”

### 3. Separador: problema y sistemas cooperativos — 10 segundos

“Comencemos por la motivación: una metaheurística no se comporta igual durante toda la ejecución.”

### 4. El desafío de coordinar la búsqueda — 45 segundos

“En una optimización difícil aparecen dos necesidades que compiten. La exploración global busca regiones distintas y mantiene diversidad; evita que el algoritmo se quede atrapado demasiado pronto. La explotación local toma una región prometedora y la refina para acercarse rápidamente a un óptimo.

El problema es el equilibrio. Si exploro demasiado, gasto presupuesto sin consolidar una solución. Si exploto demasiado, puedo converger prematuramente a una solución mediocre. Además, la estrategia conveniente cambia durante la ejecución: al principio suele convenir explorar y, cuando aparece una buena región, conviene intensificar. Por eso una combinación fija de algoritmos no necesariamente es la mejor.”

### 5. La decisión que controla el sistema híbrido — 40 segundos

“La pregunta operativa es: ¿cuándo termina la fase del solucionador activo y cuándo se transfiere el control a otro?

Un presupuesto fijo es fácil de implementar, pero no observa lo que está ocurriendo. Un umbral manual depende de la escala del objetivo y del problema: un cambio de 0.001 puede ser enorme en un caso y despreciable en otro. Un único indicador escalar también puede confundir una meseta breve con estancamiento real. Una política aprendida puede ser potente, pero exige estados, entrenamiento y recompensas. Nuestra propuesta busca una señal interpretable, basada solamente en la forma temporal del fitness observado.”

### 6. Cómo son hoy los sistemas cooperativos — 35 segundos

“La cooperación suele implementarse de tres formas. En un portafolio o sistema de islas, cada algoritmo mantiene su población y se intercambian soluciones. En una hiperheurística, una capa superior selecciona el algoritmo u operador que actuará. En una configuración adaptativa, esa decisión cambia según el estado observado.

En todos los casos hay tres preguntas concretas: cuándo comunicar, qué información transferir y qué algoritmo continúa. Esta presentación se concentra en la primera pregunta, sin olvidar que la transferencia debe ser compatible con el algoritmo receptor.”

### 7. Ciclo de cooperación en la literatura — 45 segundos

“Esta figura es una síntesis de la literatura sobre hiperheurísticas, selección adaptativa y migración; no corresponde a un único artículo. Primero se inicializan los solucionadores. Cada uno ejecuta una época, es decir, un bloque de iteraciones. Luego se mide el fitness y el estado de búsqueda. La política decide si hay una alerta. Si no hay alerta, continúa la época; si la hay, se selecciona un destino y el contenido a transferir.

La transferencia conserva el mejor candidato factible, adapta la representación al algoritmo receptor y reinicializa la diversidad necesaria. Después comienza una nueva época. DTW/DDTW reemplaza el disparador escalar por una lectura de la trayectoria reciente del mejor fitness; el resto del ciclo queda visible y auditable.”

### 8. Brecha de investigación — 35 segundos

“La literatura ya utiliza mejora reciente, diversidad, características de población, historiales de operadores y trayectorias para seleccionar algoritmos o construir un warm start. La brecha que abordamos es usar explícitamente la forma temporal del fitness incumbente para distinguir progreso sostenido de estancamiento persistente.

La contribución no es afirmar que DTW reemplace a todas las políticas adaptativas. Es proponer una señal que no necesita entrenamiento supervisado, que puede compararse entre algoritmos distintos y que toma en cuenta el orden temporal de las mejoras.”

### 9. Separador: DTW/DDTW y arquitectura — 10 segundos

“Ahora explicaré el núcleo del método: cómo se comparan dos trayectorias aunque avancen a velocidades diferentes.”

### 10. Objetivo y alcance — 35 segundos

“El objetivo general es desarrollar y validar un marco cooperativo para dimensionar HRES--H$_2$, usando DTW y DDTW como detector adaptativo de estancamiento y como señal de coordinación.

Para lograrlo hay cinco tareas: caracterizar el modelo energético, diseñar el detector sobre ventanas móviles del mejor fitness, definir el intercambio entre algoritmos heterogéneos, orquestar la alternancia entre exploración y explotación y validar el marco antes de extenderlo a compromisos multiobjetivo mediante $ε$-restricción adaptativa.”

### 11. Diagrama de flujo de la propuesta — 30 segundos

“El diagrama resume las tres capas. Abajo está el problema que entrega candidatos y métricas. En el centro está el motor DTW/DDTW, que recibe una ventana de fitness, la transforma según el modo de comparación y la contrasta con referencias de progreso y estancamiento. Arriba está la capa cooperativa, que selecciona solucionadores, ejecuta épocas y transfiere el incumbente cuando se confirma una alerta.

La idea importante es que el detector no optimiza directamente el sistema energético: observa la búsqueda y decide cuándo conviene cambiar su mecanismo.”

### 12. DTW y DDTW: misma lógica, distinta señal — 1 minuto

“Aquí es importante separar las dos variantes. DTW y DDTW no son dos optimizadores distintos: son dos formas de calcular las distancias dentro del mismo detector.

El detector toma una ventana reciente `X` del mejor fitness y construye dos referencias desde su primer valor. La primera, `r`, es una rampa que representa progreso sostenido. La segunda, `c`, es una línea constante que representa una meseta. Para la misma ventana calculamos una distancia al progreso, que llamamos `D_prog`, y una distancia a la meseta, que llamamos `D_mes`.

En la variante DTW se comparan directamente los niveles de `X`, `r` y `c`. En la variante DDTW primero calculamos las diferencias entre puntos consecutivos y aplicamos DTW sobre esas diferencias. Por eso DTW observa nivel y forma, mientras DDTW enfatiza la pendiente.

Como HRES minimiza LCOE, el código entrega al monitor el LCOE con signo negativo. De esta manera, cuando el LCOE baja, la señal interna aumenta y coincide con la rampa ascendente de progreso. Después de esta transformación, ambas variantes usan exactamente la misma regla de decisión.”

“En el benchmark de `HRES2-H2` la configuración activa es `use_ddtw=True`: los resultados reportados corresponden a DDTW, aunque la presentación muestra también DTW para comparar ambas señales.”

### 13. Banda Sakoe–Chiba: limitar el alineamiento — 50 segundos

“Esta banda no cambia la fórmula de DTW; restringe los caminos que se pueden recorrer en la matriz de costos. El eje vertical es la ventana observada `X` y el eje horizontal es la referencia `r` o `c`. La condición es `|i-j| <= b`, es decir, solo se visitan las celdas cercanas a la diagonal.

La recurrencia que usa el código es `D(i,j) = |X_i-Y_j| + min{D(i-1,j), D(i,j-1), D(i-1,j-1)}`. Los tres términos representan los tres movimientos permitidos: avanzar en `X`, avanzar en la referencia o avanzar en ambas series. La inicialización también coincide con el código: `D(0,0)=0` y el resto de la fila y columna iniciales se deja en infinito.

En el ejemplo didáctico usamos seis puntos y `b = 2`. Una celda como `(3,5)` está permitida porque `|3-5| = 2`; una celda como `(1,5)` está prohibida porque `|1-5| = 4`. El camino rojo usa movimientos verticales, diagonales y horizontales dentro de la zona azul.

En el código de HRES2-H2 se deja `STAG_BAND = 0` para activar la regla automática: `b = max(1, floor(0.1 W))`. Como la ventana real es `W = 40`, la banda queda aproximadamente en 4. Por lo tanto, no se permite que un punto de la ventana se alinee con cualquier punto distante de la referencia.”

### 14. Ejemplo numérico: DTW con los tres movimientos — 1 minuto 30 segundos

“La ventana observada es `X = [0.50, 0.50, 0.50, 0.56, 0.60, 0.65]` y la referencia de progreso es `r = [0.50, 0.55, 0.60, 0.64, 0.65, 0.65]`. Tienen una forma ascendente semejante, pero progresan con distinto ritmo. Usamos banda 2, de modo que el camino puede separarse hasta dos posiciones de la diagonal.

La matriz muestra los costos locales `Cij = |Xi-rj|`. Las celdas grises quedan fuera de la banda y no pueden visitarse; las verdes forman un camino óptimo.

Primero hay dos movimientos verticales: `(1,1)` a `(2,1)` y luego a `(3,1)`. Eso permite alinear los tres valores 0.50 de `X` con el primer 0.50 de la referencia. Después aparecen tres movimientos diagonales: `(3,1)` a `(4,2)`, a `(5,3)` y a `(6,4)`. Finalmente hay dos movimientos horizontales hasta `(6,6)`, que alinean el último 0.65 de `X` con los valores finales de la referencia.

Los costos recorridos suman `0 + 0 + 0 + 0.01 + 0 + 0.01 + 0 + 0 = 0.02`, por lo que `D_prog = 0.02`. Contra la referencia constante `c = [0.50, 0.50, 0.50, 0.50, 0.50, 0.50]`, el código entrega `D_mes = 0.31`. Así, `delta = 0.02 - 0.31 = -0.29`: la ventana se parece más al progreso, aunque las dos series no estén sincronizadas punto a punto.”

### 15. Delta y condiciones de estancamiento — 1 minuto 20 segundos

“Este es el núcleo de la propuesta. En el código, `D1` es la distancia a la rampa de progreso y `D2` es la distancia a la referencia constante. El discriminante se calcula en este orden:

`delta = D_prog - D_mes = D1 - D2`.

El orden importa. Si delta es negativo, la distancia al progreso es menor y la ventana se parece más a una búsqueda que sigue avanzando. Si delta está cerca de cero, la evidencia es ambigua. Si delta es positivo y grande, la distancia al progreso es mayor que la distancia a la meseta; por lo tanto, la trayectoria se parece más a un estancamiento.

Por ejemplo, si la distancia al progreso es 0.80 y la distancia a la meseta es 0.20, delta vale 0.60. Eso es evidencia a favor de la meseta. Estos valores son ilustrativos, porque los umbrales reales se adaptan a la escala observada.

El algoritmo no cambia de solucionador usando solamente el signo de delta. Exige tres condiciones de estancamiento. Primera: al menos 15 iteraciones sin mejora. Segunda: la distancia a la meseta debe ser suficientemente baja, es decir, la ventana debe parecerse a una constante. Tercera: la distancia al progreso debe ser alta o el delta debe superar su umbral. Además, la condición debe repetirse tres veces consecutivas. La ventana contiene 40 valores y los umbrales se estiman mediante percentiles del historial.

En DDTW el delta tiene exactamente el mismo significado. La única diferencia es que `D_prog` y `D_mes` se calcularon sobre cambios entre puntos, no sobre los niveles originales. Tampoco hablamos de un progreso futuro real: la comparación se hace con una rampa de referencia construida por el detector.”

### 16. Conjunto de metaheurísticas — 35 segundos

“El pool combina algoritmos con comportamientos distintos. GA, PSO, GWO, WOA, EHO y ACO son poblacionales y aportan diversificación y exploración global. SA, TS, ILS y VNS usan memoria, perturbaciones o vecindades y tienden a intensificar la búsqueda.

La hipótesis es que un algoritmo puede ser adecuado para una etapa y otro para la siguiente. No se fija de antemano una secuencia óptima. DTW/DDTW observa la trayectoria y decide cuándo vale la pena probar otro mecanismo.”

### 17. Comunicación entre solucionadores — 30 segundos

“Cuando aparece una alerta, el algoritmo activo entrega su mejor solución factible. El orquestador la conserva como incumbente y la adapta al receptor. Si el receptor necesita una población, esa solución se combina con perturbaciones factibles y candidatos adicionales para recuperar diversidad.

Esto es warm start: no se reinicia desde cero, pero tampoco se copia ciegamente un estado incompatible. La transferencia sigue el sentido poblacional, trayectoria, poblacional, y la señal DTW/DDTW es la condición de transición.”

### 18. Separador: modelo HRES2--H$_2$ — 10 segundos

“Con el mecanismo de coordinación claro, paso al caso de estudio donde cada evaluación de una solución implica simular un sistema energético durante un año completo.”

### 19. WPEB publicado versus implementación HRES2--H$_2$ — 1 minuto 15 segundos

“Esta tabla es importante porque separa el punto de partida de la extensión implementada.

El WPEB de referencia fija la capacidad total eólica más fotovoltaica en 200 MW, simula 8.760 horas, minimiza LCOE y limita el excedente vendido a la red mediante AGSR menor o igual a 20 por ciento. Sus variables se expresan como razones de capacidad continua y el método de búsqueda combina grid search con descenso de gradiente. El caso reportado es W190--P10--E95--B30, con LCOE de 0.2692 CNY por kWh.

Nuestra implementación conserva la física principal y el horizonte horario, pero cambia el dominio de decisión. El viento es continuo; el electrolizador se expresa en módulos enteros de 5 MW; la batería tiene potencia y duración discretas. Además, el simulador queda dentro de un framework DTW/DDTW que cambia solucionadores y transfiere el incumbente. Así podemos medir la diferencia entre optimizar capacidades con una búsqueda estática y optimizarlas con cooperación adaptativa.”

### 20. HRES2--H$_2$: qué se dimensiona — 50 segundos

“El optimizador recibe un vector numérico de cuatro coordenadas, x=[x_0,x_1,x_2,x_3]. El decodificador transforma x_0 en P_WT, x_1 en N_el, x_2 en P_bat y x_3 en el índice de duración; el resultado físico es z=[P_WT,N_el,P_bat,tau_bat]. P_WT es la capacidad eólica instalada en MW; N_el es el número entero de módulos PEM; P_bat es la potencia de la batería en MW; y tau_bat es su duración en horas.

P_PV=200-P_WT significa que la potencia solar instalada completa los 200 MW renovables; P_PV y P_WT se miden en MW. Cada módulo PEM aporta 5 MW, por eso P_el=5×N_el, donde P_el es la potencia nominal del electrolizador. Se permiten de 10 a 20 módulos, y además P_el<=100 MW, el 50 % de los 200 MW renovables. La batería permite de 0 a 50 MW, en pasos de 5, y duraciones de 1, 2 o 4 horas. Su capacidad energética es E_bat=P_bat×tau_bat: por ejemplo, 50 MW durante 4 h son 200 MWh. En el vector codificado, x_3 es un índice de duración: el valor 2 se decodifica como 4 horas.”

### 21. Generación renovable horaria — 55 segundos

“Cada candidato se evalúa con perfiles meteorológicos para 8.760 horas. La potencia renovable disponible se calcula como P_gen(t)=P_WT×f_WT(t)+P_PV×f_PV(t). Aquí t es la hora del año; P_WT y P_PV son capacidades instaladas en MW; f_WT(t) y f_PV(t) son factores horarios entre 0 y 1; y P_gen(t) es la potencia disponible en MW, no energía.

En el viento, v(t) es la velocidad horaria. El factor f_WT(t) vale cero si el viento es menor que v_i=2.5 m/s o llega al corte v_o=25 m/s; crece según (v(t)^3-v_i^3)/(v_r^3-v_i^3) entre arranque y velocidad nominal v_r=10.5 m/s; y vale uno entre la velocidad nominal y el corte. El simulador multiplica ese factor por P_WT para obtener MW eólicos.

En la PV, la irradiancia horaria es el factor principal: durante la noche la generación es cero y durante el día aumenta con la radiación. El modelo aplica además una corrección por temperatura, porque una celda más caliente produce con menor eficiencia, y un factor de pérdidas de 0.90. En forma conceptual, la potencia solar es la capacidad instalada multiplicada por la irradiancia, la corrección térmica y las pérdidas. El factor horario se limita entre cero y uno para mantener un resultado físicamente válido. Finalmente, P_PV×f_PV entrega la potencia solar en MW.”

### 22. Cómo opera cada hora — 1 minuto

“Primero, el mínimo de operación es P_el,min=0.30×P_el: P_el,min es la potencia mínima en MW, P_el la potencia nominal y 0.30 el porcentaje de carga mínima. Si P_gen(t)>=P_el,min, el consumo es P_el,usada(t)=min(P_gen(t),P_el), donde min escoge el menor valor para no superar ni la generación ni el tamaño del PEM.

Si la generación queda bajo el mínimo, se intenta usar la batería. SOC(t) es su energía almacenada en MWh; inicia en cero y no puede superar SOC_max=P_bat×tau_bat. La descarga máxima es P_dis,max=min(P_bat,SOC×eta_dis/Δt): P_bat limita potencia en MW, SOC es energía en MWh, eta_dis considera pérdidas y Δt=1 h convierte esa energía disponible a potencia. Si P_gen+P_dis,max no alcanza P_el,min, el electrolizador se detiene. Si sí alcanza, el código descarga P_dis=min(P_dis,max,P_el-P_gen): entrega toda la potencia posible sin superar la potencia nominal del PEM; el mínimo es una condición para operar, no el nivel al que se limita la descarga. El excedente carga la batería respetando tanto P_bat como el espacio libre en SOC_max; lo que sobra se exporta. eta_ch=eta_dis=sqrt(eta_rt)=sqrt(0.90)=0.9487, donde eta_rt es la eficiencia round-trip. La energía anual se obtiene sumando potencia por Δt=1 h; así E_el,annual es la energía del electrolizador en MWh/año.

La potencia horaria realmente recibida por el PEM se suma durante las 8.760 horas: así se obtiene E_el,annual en MWh/año. El intervalo es Δt=1 h, y no se debe confundir esa energía consumida con la potencia nominal del electrolizador. En la siguiente diapositiva uso esa energía para calcular la masa de hidrógeno.”

### 23. De electricidad renovable a producción de H$_2$ — 45 segundos

“La primera ecuación suma la potencia que realmente recibió el electrolizador en cada hora: E_el,annual=Σ P_el,usada(t)×Δt. E_el,annual es energía anual en MWh; P_el,usada(t) es potencia del PEM en la hora t, medida en MW; t recorre las 8.760 horas; y Δt vale una hora. Es el consumo después del despacho, no la potencia nominal multiplicada por las horas del año.

Luego convertimos esa electricidad a masa: m_H2=E_el,annual×1000×eta_el/HHV_H2. El factor 1000 convierte MWh a kWh; eta_el=0.75 es la eficiencia, sin unidades; y HHV_H2=39.4 kWh/kg es la energía por kilogramo de hidrógeno. El resultado m_H2 queda en kg por año.

Por ejemplo, 100 MWh consumidos equivalen a 100.000 kWh eléctricos; al aplicar 0.75 quedan 75.000 kWh de energía química y, al dividir por 39.4, obtenemos 1.903,6 kg de H2, o 1,90 toneladas al año. En este modelo se llama verde porque la electricidad del PEM viene de eólica/PV o de la batería cargada con excedentes renovables; no se modela compra a red. Esto no es una certificación ni un análisis de emisiones de ciclo de vida.”

### 24. Objetivos y restricciones — 1 minuto 10 segundos

“El objetivo es minimizar el costo nivelado de electricidad: LCOE=(NPC×CRF)/E_entregada,annual. NPC es el costo presente neto en CNY; incorpora inversión, reemplazos y operación/mantenimiento de eólica, PV, electrolizador y batería. Para cada tecnología j, el costo inicial usa su capacidad P_j en MW por 1000 para convertir a kW y por el costo unitario C_cap,j; los reemplazos usan C_rep,j en CNY/kW y O&M usa C_OM,j en CNY/(kW·año). Cada flujo futuro se descuenta por (1+r)^y. Aquí r=0.0435 es la tasa anual y y el año del gasto. El proyecto dura N=25 años y CRF=r(1+r)^N/[(1+r)^N-1] convierte el NPC a costo anual equivalente en CNY/año.

El denominador E_entregada,annual=1000×(E_el,annual+E_grid,sales,annual) está en kWh/año: E_el,annual es energía recibida por el electrolizador y E_grid,sales,annual es energía vendida a la red; ambas están en MWh/año antes de multiplicar por 1000. Así el LCOE resulta en CNY/kWh. LCOH es el costo anualizado dividido por m_H2 en kg/año, por eso se expresa en CNY/kg; aquí es un indicador, no el objetivo.

La restricción es AGSR=E_grid,sales,annual/E_ren,annual<=0.20. E_ren,annual es la generación renovable total anual; E_grid,sales,annual es el excedente anual exportado; ambos están en MWh/año, así que AGSR es una razón sin unidades. Si excede 0.20, el diseño es inviable y el fitness pasa a 100+10×AGSR, una penalización artificial mucho peor que el LCOE factible. El límite evita favorecer un diseño que exporta demasiado excedente.”

### 25. Cómo se integra el optimizador con el HRES — 40 segundos

“La integración es un bucle cerrado. El solucionador propone un vector de capacidades. El simulador ejecuta las 8.760 horas. Luego devuelve LCOE, producción de H$_2$, AGSR, energía exportada y factibilidad. Con esa trayectoria de fitness, el detector actualiza DTW o DDTW.

Si la solución mejora, se conserva como incumbente. Si la búsqueda se estanca durante la condición de paciencia, el orquestador cambia de solucionador e inicia una nueva época con warm start. Así, el HRES actúa como evaluador físico y económico, mientras el marco decide cómo continuar la búsqueda.”

### 26. Protocolo experimental — 45 segundos

“El protocolo tiene tres niveles. Para el detector se miden sensibilidad, especificidad, falsos positivos y evaluaciones ahorradas. Para la optimización se miden calidad, convergencia, tiempo y robustez entre ejecuciones. Para HRES2--H$_2$ se miden LCOE, LCOH, hidrógeno anual, AGSR y energía exportada.

Los resultados que mostraré corresponden a 31 corridas independientes y 1.000 iteraciones por corrida, usando el mismo perfil anual sintético. Las comparaciones estadísticas se plantean con pruebas no paramétricas. En la comparación directa DTW--DDTW del paper final, la corrección de Holm no detectó una diferencia significativa entre ambas variantes.”

### 27. Resultados: métricas y configuración — 1 minuto

“Aquí aparecen los resultados principales. Tanto DTW como DDTW alcanzan un LCOE medio de 0.267160 CNY por kWh. El mejor valor observado es 0.267159. Las desviaciones estándar son del orden de $10^{-6}$, lo que indica una dispersión muy pequeña entre corridas.

Ambas variantes son factibles en el 100 por ciento de las corridas y quedan en AGSR de 20 por ciento, es decir, la restricción está activa o prácticamente activa. También convergen a la misma configuración: 174.47 MW eólicos, 25.53 MW solares, 70 MW de electrólisis, equivalentes a 14 módulos, y una batería de 50 MW durante 4 horas. La producción anual de hidrógeno es de aproximadamente 8.86 millones de kilogramos.

LCOH es ligeramente distinto entre las variantes, pero se reporta como métrica secundaria. La conclusión central es la estabilidad y la factibilidad completa.”

### 28. Comparación con solucionadores base — 45 segundos

“Esta tabla pone el resultado en contexto. Los híbridos DTW y DDTW tienen la menor media de LCOE, el mejor LCOE observado y la menor dispersión. Los solucionadores base pueden alcanzar ocasionalmente valores cercanos, pero sus medias son mayores o presentan variabilidad más alta.

También se ve la factibilidad: los híbridos llegan al 100 por ciento, mientras que los métodos base tienen porcentajes inferiores. AGSR se mantiene cerca del límite en todos los métodos, pero alcanzar un AGSR bajo no garantiza por sí mismo una solución estable o factible. La ventaja que buscamos mostrar es la combinación entre costo, robustez y cumplimiento de restricciones.”

### 29. Convergencia del pipeline DTW — 50 segundos

“En la curva DTW se observa una caída rápida al principio: el pipeline encuentra mejoras importantes durante las primeras épocas. Luego aparecen plateaus. Las líneas verticales marcan los cambios de solucionador; no representan reinicios completos, sino transferencias del incumbente.

La lectura importante es que el cambio de algoritmo ocurre sobre evidencia de la trayectoria. Cuando el fitness deja de mejorar, otro mecanismo puede intentar salir de la meseta. Después de las primeras mejoras, la curva se estabiliza cerca de 0.2672 y termina en el mejor valor observado, 0.267159 CNY por kWh.”

### 30. Convergencia del pipeline DDTW — 50 segundos

“La variante DDTW llega a la misma región final. La diferencia es la señal usada para interpretar la trayectoria: DDTW enfatiza la pendiente y, por lo tanto, la pérdida de progreso.

En este caso no vemos una mejora estadísticamente significativa respecto de DTW, pero sí el mismo comportamiento de estabilidad y factibilidad. Esto es relevante: la derivada no necesariamente mejora el valor final en este HRES, pero ofrece una segunda forma de detectar la transición entre progreso y estancamiento. La elección entre DTW y DDTW puede depender del ruido y de la forma de las trayectorias en otros escenarios.”

### 31. Posicionamiento frente al trabajo relacionado — 55 segundos

“Los enfoques existentes pueden migrar soluciones, seleccionar por características de trayectoria o usar aprendizaje y recompensa. La propuesta se diferencia en tres aspectos.

Primero, la señal es temporal: importa el orden y la forma de las mejoras. Segundo, es independiente del solucionador: la misma lógica puede observar PSO, SA, GWO u otro algoritmo porque solo necesita la trayectoria del incumbente. Tercero, no requiere un modelo aprendido para activar el cambio; usa plantillas, distancias y persistencia, por lo que la decisión se puede inspeccionar después.

Esto no significa que sea universal. Hay que calibrar ventana, percentiles y paciencia, y evaluar diferentes perfiles meteorológicos. La contribución es una arquitectura interpretable y comprobable.”

### 32. Plan de trabajo — 1 minuto

“El plan cubre 19 semanas entre agosto y diciembre de 2026. Las primeras seis semanas corresponden a la validación inicial del marco y cierran con la primera entrega. Desde la semana 7 se concentra la extensión multiobjetivo del HRES2--H$_2$, la incorporación de nuevos solucionadores y la ampliación del protocolo.

Entre las semanas 12 y 18 se ejecutan campañas experimentales, comparación, análisis de limitaciones y redacción. Los hitos principales son la primera entrega en la semana 6, el software en la semana 15, el informe final en la semana 16 y las presentaciones finales entre las semanas 17 y 19.

La condición metodológica se mantiene durante todo el plan: presupuestos equivalentes, corridas independientes y control explícito de factibilidad.”

### 33. Contribuciones esperadas — 45 segundos

“Las contribuciones esperadas son cinco. Primero, un detector de estancamiento que usa la forma temporal del fitness y funciona con metaheurísticas heterogéneas. Segundo, un protocolo de transferencia que conserva el incumbente y adapta la inicialización al receptor. Tercero, un orquestador que alterna exploración y explotación con evidencia de la ejecución. Cuarto, una validación que conecta el marco de optimización con el dimensionamiento HRES2--H$_2$. Quinto, una extensión futura para estudiar compromisos económicos y ambientales mediante $ε$-restricción adaptativa.

La frase de cierre puede ser: “En síntesis, DTW/DDTW no reemplaza al optimizador energético; observa su comportamiento y decide cuándo conviene cambiar el mecanismo de búsqueda, mientras el modelo HRES2--H$_2$ verifica cada decisión contra un año completo de operación.”

### 34. Referencias y preguntas — 10 segundos

“Estas son las referencias principales. Muchas gracias por su atención. Quedo atento a sus preguntas sobre el detector, el despacho HRES2--H$_2$ y la interpretación de los resultados.”

## Explicación corta de DTW para una pregunta del jurado

Si te preguntan “¿qué hace exactamente DTW?”, puedes responder:

> DTW recibe dos secuencias, calcula un costo local entre cada par de puntos y usa programación dinámica para encontrar el camino de alineamiento de costo mínimo. Ese camino puede avanzar diagonalmente, horizontalmente o verticalmente, por lo que permite comparar una mejora rápida con una mejora lenta. En el marco, una secuencia es la ventana reciente del fitness y la otra es una plantilla de progreso o estancamiento. La distancia y su persistencia se convierten en la señal para continuar o cambiar de metaheurística.

Si preguntan por DDTW:

> DDTW aplica la misma idea después de aproximar la derivada de la trayectoria. De esa forma compara pendientes y cambios, reduciendo la influencia del nivel absoluto del fitness. Una pendiente cercana a cero durante varias iteraciones es evidencia de meseta.

## Explicación corta de HRES2--H$_2$ para una pregunta del jurado

Si te preguntan “¿qué evalúa una solución?”, responde:

> Una solución fija cuatro capacidades: potencia eólica, número de módulos PEM, potencia de batería y duración de almacenamiento. El simulador calcula generación eólica y solar hora a hora, aplica la regla de despacho durante 8.760 horas, obtiene producción de hidrógeno, energía vendida y costos, y finalmente calcula LCOE, LCOH, AGSR y factibilidad.

Si preguntan “¿qué optimizas realmente?”:

> El objetivo actual es minimizar LCOE bajo la restricción AGSR menor o igual a 20 por ciento y los dominios mixtos de las capacidades. LCOH, producción de hidrógeno y emisiones se reportan como indicadores secundarios. La formulación multiobjetivo queda como extensión del plan de trabajo.

## Tres precauciones al exponer

1. No decir que LCOH es el objetivo principal: en esta versión se minimiza LCOE.
2. No decir que DTW siempre supera a DDTW: en HRES2--H$_2$ ambos llegan a resultados prácticamente iguales y no hubo diferencia significativa tras Holm.
3. No presentar AGSR como “más bajo es siempre mejor”: es una restricción de excedente. Debe mantenerse bajo 20 por ciento, pero una reducción excesiva puede implicar sobredimensionar electrólisis o almacenamiento y aumentar el costo.
