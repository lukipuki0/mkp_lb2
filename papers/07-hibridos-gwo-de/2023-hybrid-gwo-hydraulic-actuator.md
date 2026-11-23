# A Hybridization Grey Wolf Optimizer to Identify Parameters of Helical Hydraulic Rotary Actuator

**Clave de cita:** `Zheng_2023`  
**PDF:** [PDF](./2023-hybrid-gwo-hydraulic-actuator.pdf) · **Bibliografía:** [referencias.bib](../referencias.bib)

| Campo | Dato |
| --- | --- |
| Autores | Yukun Zheng, Ruyue Sun, Yixiang Liu, Yanhong Wang, Rui Song, Yibin Li |
| Año | 2023 (recibido 27-04-2023, publicado 25-05-2023) |
| Venue | Actuators, 12, 220 (MDPI) |
| DOI | 10.3390/act12060220 |
| Sección primaria | `07-hibridos-gwo-de` |
| Etiquetas | H-GWO; GWO y DE en serie por individuo; DE/best/1 escalado; oposición en la inicialización; identificación de parámetros; horario fijo |

## Aporte para el estado del arte

Sirve como antecedente del acoplamiento secuencial más simple: en cada iteración GWO mueve al lobo y después un paso DE con selección greedy decide si el resultado sobrevive. Todo el control es de horario fijo. Es un buen contraste para mostrar qué cambia cuando los dos operadores compiten o cuando la configuración responde a la búsqueda.

## Problema y método

El problema es identificar los coeficientes `a_i`, `b_i` de un modelo discreto de tercer orden de un actuador hidráulico rotativo helicoidal (ec. 13, p. 5), con ITAE como fitness (ec. 30, p. 12). Un FFRLS da una estimación previa y el rango de búsqueda de H-GWO se abre alrededor de ella con una ganancia de expansión (ec. 29, p. 11).

H-GWO usa una sola población. La inicialización combina un mapa caótico tienda con la población opuesta y conserva los `n` mejores de `P ∪ OP` (ecs. 24-27, pp. 7-8). El bucle de los pasos 1-11 (p. 8) actualiza alfa, beta y delta, calcula la posición GWO (ec. 19), arma un mutante `V = λ·X_alpha + F·(X_r1 − X_r2)` (ec. 20, p. 6), cruza binomialmente (ec. 21), corrige límites (ec. 22) y aplica selección greedy uno a uno (ec. 23, p. 7). Cada individuo recibe un solo candidato por iteración; GWO y DE no compiten.

Ningún parámetro reacciona al estado de la búsqueda. `a` baja linealmente de 2 a 0 (ec. 16), `F = 0.5` es fijo, `Cr = 0.5(1 + rand)` es aleatorio y `λ = 0.25` es constante (p. 9). No hay detección de estancamiento, reinicio ni archivo élite. La oposición aparece solo en la inicialización: los pasos 1-11 no incluyen un salto opuesto dentro del bucle, aunque el resumen (p. 1) y la conclusión (p. 17) dicen que la oposición "update the GWO population".

## Qué sostiene el texto

En el banco sintético, nueve funciones clásicas (tres unimodales, tres multimodales y tres de dimensión fija, Tabla 1, p. 10), con `NP = 30`, `G = 100` iteraciones y 30 corridas, comparan H-GWO con PSO, GWO y DE mediante media y desviación (Tabla 2, p. 10). No hay prueba estadística. La Tabla 2 sale desalineada en la extracción de texto, así que esta ficha no transcribe sus cifras por función.

En la identificación, con datos medidos a 250 Hz durante 20 s (pp. 12-13), comparan con LS, RLS, FFRLS, PSO y GWO usando `NP = 100` y `G = 200` (p. 14). La Tabla 3 (p. 16) informa para la propuesta MSE 12.73, VAF 96.18 % y 118 iteraciones, frente a PSO con 13.94, 93.21 % y 142, y GWO con 40.68, 71.24 % y 175. El texto no informa cuántas corridas respaldan la Tabla 3 ni su dispersión.

## Límites

**Hallazgos de los autores.** Atribuyen la baja precisión de FFRLS a la cantidad de datos (p. 16) y dejan el uso del modelo en control práctico como trabajo futuro (p. 17). No discuten límites del optimizador.

**Lectura crítica.** La comparación de identificación mezcla dos factores: H-GWO arranca en el rango expandido desde FFRLS (ec. 29), mientras GWO usa `[-5, 5]` (p. 14). La ventaja de la Tabla 3 no aísla el aporte del híbrido, y DE no figura entre los comparadores de esa tabla. En el banco, cinco de las seis funciones de dimensión 30 de la Tabla 1 tienen el óptimo en el origen (la excepción es la de tipo Rosenbrock), y la base `λ·X_alpha` con `λ = 0.25` contrae la mutación hacia el origen; el texto no controla ese posible sesgo con funciones desplazadas. Las comparaciones igualan iteraciones, no evaluaciones de función. El texto tampoco precisa si el vector objetivo de las ecs. 21 y 23 es la posición recién movida por GWO o la anterior. Por último, la numeración de referencias del texto no coincide con la lista (por ejemplo, GWO se cita como [24] en p. 2 y aparece como 28 en p. 19).

## Relación con el GWO-DE actual

Comparte la población única, la presencia de GWO y DE en la misma iteración y la aceptación greedy. Difiere en lo central: GWO y DE se aplican en serie dentro del individuo, sin dos candidatos que compitan, y `a`, `F` y `Cr` siguen horarios o ruido, no una señal de la trayectoria del mejor fitness. Es un antecedente del modo secuencial, no evidencia sobre control realimentado. El problema es continuo; no hay binarización ni reparación.

## Evidencia

- p. 1: resumen; afirma oposición para actualizar la población.
- p. 5: modelo discreto (ecs. 11-13) y definiciones de oposición.
- p. 6: GWO (ecs. 14-19) y mutante DE/best/1 escalado (ec. 20).
- p. 7: cruce (ec. 21), límites (ec. 22), selección greedy (ec. 23), mapa tienda (ecs. 24-25).
- p. 8: población opuesta (ecs. 26-27), pasos 1-11 y Fig. 3.
- p. 9: `NP = 30`, `G = 100`, parámetros de PSO, GWO, DE y H-GWO.
- p. 10: 30 corridas, Tablas 1 y 2.
- p. 11: FFRLS (ec. 28) y rango expandido (ec. 29).
- p. 12: ITAE, MSE y VAF (ecs. 30-32).
- p. 14: parámetros de identificación; GWO con rango inicial `[-5, 5]`.
- p. 16: Tabla 3.
- p. 17: conclusión.
