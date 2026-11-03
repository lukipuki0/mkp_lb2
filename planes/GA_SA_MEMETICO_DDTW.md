# Propuesta: GA–SA memético activado por DTW/DDTW para MKP

Estado: diseño documentado para implementar después. No se implementa ni se ejecuta un experimento con este documento.
Fecha: 2026-10-26. Fuente: decisiones de la conversación con el usuario e inspección del SA del repositorio.

## 1. Objetivo

Construir un híbrido entre dos metaheurísticas que se comuniquen: GA mantiene y evoluciona una población; SA refina cada individuo únicamente cuando el monitor detecta estancamiento de la trayectoria del GA. Las mejores soluciones de SA regresan al GA mediante reinserción por pareja.

La hipótesis por evaluar es que activar búsqueda local según DTW/DDTW aprovecha mejor el presupuesto que activarla periódicamente. No se presupone superioridad, optimalidad ni que una meseta implique pérdida de diversidad.

## 2. Decisiones acordadas

| Aspecto | Decisión |
|---|---|
| Problema | MKP binario, maximización de beneficio, soluciones factibles mediante reparación. |
| Híbrido | GA como búsqueda poblacional y SA como búsqueda local. |
| Activación de SA | Solo al confirmar estancamiento del GA. |
| Regla inicial | Las tres condiciones del monitor compartido, con paciencia. D2 simple e histéresis quedan para después. |
| Individuos refinados | Cada individuo de la población del GA, no solo el mejor. |
| Inicio de cada SA | Copia del individuo correspondiente; no inicialización aleatoria. |
| Detector dentro de SA | Desactivado. El monitor externo observa generaciones del GA. |
| Reinserción | Comparar cada original con la mejor solución de su SA; reemplazar solo con mejora estricta. Empate: conservar original. |
| Vecindario inicial | `flip_bits`, 3 bits aleatorios, con reparación, reutilizando el SA del repo. |
| Temperatura inicial | 5000. |
| Enfriamiento | Geométrico: `T ← 0.97 × T`. |
| Trabajo por temperatura | **10 intentos de vecino por nivel de temperatura**. |
| Terminación térmica | `T ≤ 1`. No limitar inicialmente a 10 niveles. |
| Presupuesto general | Contabilizar trabajo GA y SA dentro del mismo presupuesto total; su valor y la política exacta de interrupción se fijan antes de implementar. |

Los parámetros térmicos y el vecindario se toman del código existente como punto de partida. No se realizó todavía una revisión bibliográfica que los valide para este híbrido. El usuario pidió parámetros respaldados por literatura: esa búsqueda sigue pendiente.

### Aclaración del presupuesto de SA

Se descartó la propuesta anterior de 10 niveles × 50 vecinos, porque enfriaba demasiado poco. La decisión posterior fue:

> "hagamos 10 vecinos y mas niveles de temperatura hasta q llegue a un criterio de termino"

Con `T0=5000`, `Tf=1` y `alpha=0.97`, el número de niveles completos es:

`ceil(log(1/5000) / log(0.97)) = 280`.

Son aproximadamente 2800 intentos de vecino por individuo y, para una población ilustrativa de 30, 84000 por activación. La población de 30 es un ejemplo de costo, no un valor aprobado para la nueva implementación. Estas cifras excluyen inicialización, reevaluaciones y otros costos adicionales que el contador final deberá especificar.

## 3. Qué existe en el repositorio

### SA inspeccionado

- `hybrid_mkp/mh/sa.py`: `SAParams`, `ejecutar_epoch()` y resultados.
- `hybrid_mkp/mh/sa_neighborhood.py`: `flip_bits`, `swap_bits`, `block_flip`.
- `hybrid_mkp/orchestrator.py`: integración actual de SA en el pipeline de selección.
- `dtw_stagnation.py`: monitor compartido con las tres condiciones y paciencia.

El SA actual acepta `sol_inicial`, repara la solución inicial y cada vecino, conserva el mejor visitado y devuelve ese mejor. Una iteración registrada por SA equivale a un nivel de temperatura, no a un intento de vecino. `max_levels` es un límite opcional de niveles; `use_stagnation` activa un monitor interno que aborta al dispararse.

Para el nuevo híbrido se propone reutilizar el núcleo de un epoch, con `iter_por_T=10`, `use_stagnation=False` y sin el límite arbitrario de diez niveles. No usar `ejecutar_sa()` para lanzar diez epochs independientes por individuo: esa no es la propuesta acordada.

**Pendiente técnico:** el SA inspeccionado limita por temperatura o niveles, pero no ofrece un presupuesto global compartido de evaluaciones. Habrá que diseñar ese control antes de ejecutar comparaciones justas. No afirmar que ya está implementado.

### Selección de implementación GA

El archivo `ALGORITMO_CENTRAL.md` describe GA generacional con cruce uniforme, mutación bit-flip y elitismo. Hay implementaciones en distintas carpetas; todavía no se eligió formalmente cuál será la base del híbrido. Revisar reparación, representación, acceso a población y evaluación antes de decidir.

No confundir:
- Control de parámetros de una única MH (`DTW_optimization-main`).
- Selección entre MH (`adaptive_mh_selection`).
- Este nuevo híbrido memético, que inserta búsqueda SA dentro del ciclo de GA.

## 4. Flujo completo

### 4.1 Inicialización

1. Cargar instancia y referencia de calidad, si está disponible.
2. Inicializar generadores aleatorios reproducibles y población GA.
3. Reparar y evaluar candidatos.
4. Conservar mejor solución global y fitness.
5. Crear monitor DTW/DDTW y contador del presupuesto total.

### 4.2 Una generación GA

1. Seleccionar padres.
2. Aplicar cruce y mutación.
3. Reparar y evaluar descendientes.
4. Formar la población siguiente conservando el elitismo definido.
5. Actualizar el mejor fitness histórico.
6. Enviar **una observación por generación** al monitor.

Los pasos internos y vecinos de SA no se agregan como observaciones de generaciones GA. Evitar mezclar unidades temporales.

### 4.3 Detector externo

Una vez completada la ventana de W observaciones, comparar la trayectoria X con una rampa R y una constante C:

- `D1 = DTW(X, R)` o su versión DDTW según configuración.
- `D2 = DTW(X, C)` o su versión DDTW.
- `delta = D1 - D2`.

Condiciones simultáneas:

1. `no_improve_len >= plateau_max`.
2. `D2 <= theta_c`.
3. `(D1 >= theta_r) OR (delta >= theta_delta)`.

Incrementar `trigger_streak` solo si se cumplen las tres; si no, reiniciarlo. Activar SA cuando `trigger_streak >= patience`. Respetar el warm-up antes de evaluar condiciones.

Reutilizar la semántica comprobada del monitor; no sustituirla por `delta > 0` ni por D2 solamente. Ventana, banda, pendiente, umbrales adaptativos, tolerancia de mejora y paciencia de este experimento aún deben fijarse explícitamente. Los valores usados en pilotos de otra variante no se adoptan automáticamente aquí.

### 4.4 Intervención SA

Al activarse el detector, pausar la evolución generacional y tomar los individuos actuales como originales de la intervención.

Para cada individuo `x_i`:

1. Inicializar SA desde una copia de `x_i`.
2. Guardar solución actual y mejor local por separado.
3. Mientras `T > 1` y quede presupuesto autorizado:
   - Generar hasta diez vecinos a esa temperatura.
   - Invertir tres bits aleatorios, reparar y evaluar.
   - Calcular `delta_sa = fitness_vecino - fitness_actual`.
   - Aceptar mejoras; para empeoramientos usar `exp(delta_sa / T)`.
   - Actualizar el mejor local cuando corresponda.
   - Después del nivel, aplicar `T *= 0.97`.
4. Devolver el mejor local, no el último estado aceptado.

Los diez intentos de vecino son secuenciales: parten del estado actual de SA, que puede cambiar por aceptación. No son necesariamente diez vecinos independientes del individuo GA original. La población puede procesarse secuencialmente; no se requiere paralelismo.

### 4.5 Reinserción por pareja

Para cada pareja original/refinado:

```text
si fitness(mejor_SA_i) > fitness(original_GA_i):
    población[i] = mejor_SA_i
si no:
    población[i] = original_GA_i
```

Mantener el tamaño de población; actualizar fitness asociados y mejor global. Las soluciones aceptadas transitoriamente por SA pueden ser peores, pero no deben degradar al individuo reinsertado.

Esta regla no garantiza diversidad: distintos SA pueden terminar en la misma solución. El tratamiento de duplicados sería una decisión adicional, no una modificación automática del diseño inicial.

### 4.6 Rearme

Propuesta pendiente de precisar técnicamente: reiniciar la evidencia del monitor tras la intervención, sin reiniciar GA ni borrar su mejor global. Exigir observaciones nuevas antes de reactivar SA, evitando disparos repetidos de la misma meseta.

Decidir antes de implementar si la observación inicial del nuevo monitor es el mejor post-SA o la siguiente generación GA, y si se reinician también historiales de umbrales. Registrar ese contrato; afecta el tiempo hasta la próxima activación.

### 4.7 Terminación

- Cada SA termina al enfriar hasta `T <= 1` o al agotarse el presupuesto general.
- El híbrido termina al agotar el criterio global acordado.
- Si el presupuesto se agota a mitad del refinamiento de la población, no inventar resultados para individuos no procesados ni sobrepasar silenciosamente el límite. Definir y documentar la política exacta antes de implementar.

## 5. Pseudocódigo conceptual

```text
inicializar población GA, mejor global, monitor y presupuesto

mientras quede presupuesto:
    ejecutar generación GA dentro del presupuesto
    actualizar mejor global
    señal = monitor.actualizar(mejor_global.fitness)

    si señal está lista y confirma estancamiento:
        para cada individuo original de la población:
            si no queda presupuesto: terminar refinamiento
            mejor_SA = SA(original,
                          T0=5000, Tf=1, alpha=0.97,
                          vecinos_por_nivel=10,
                          flip_bits=3, DTW_interno=False,
                          presupuesto_compartido)
            si mejor_SA mejora estrictamente al original:
                reemplazar su pareja
        actualizar mejor global
        rearmar monitor según política acordada

retornar mejor solución factible, métricas e historiales
```

Es pseudocódigo de diseño: no afirma que exista un argumento `presupuesto_compartido` en el SA actual.

## 6. Evaluación propuesta

Comparar con semillas emparejadas y mismo presupuesto total de evaluaciones:

1. GA solo.
2. GA–SA con activación periódica.
3. GA–SA con activación DTW/DDTW de tres condiciones y paciencia.

Mantener parámetros GA/SA y reglas de reinserción iguales entre híbridos; inicialmente estudiar solo cuándo activar SA. Definir frecuencia del control periódico antes de ejecutar.

Registrar al menos:
- Fitness final, solución factible y brecha contra referencia identificada.
- Evaluaciones GA, evaluaciones SA y total.
- Generaciones, intentos de vecino y niveles térmicos: unidades separadas.
- Activaciones de SA, instante de activación y métricas del detector.
- Ganancia por individuo y por intervención, individuos efectivamente procesados.
- Tiempo, configuración completa, semillas y versiones de fuentes.

Los mejores valores factibles de OR-Library no deben etiquetarse automáticamente como óptimos certificados. Una sola corrida sirve como prueba de funcionamiento, no como validación estadística.

## 7. Pruebas necesarias antes de experimentar

- Ninguna intervención antes de ventana completa y paciencia cumplida.
- Cada condición falsa impide disparo; verificar límites exactos del contador.
- SA recibe el individuo correcto, no una solución aleatoria ni una referencia compartida mutable.
- Diez intentos por nivel completo y enfriamiento correcto; interrupción segura por presupuesto.
- Mejor solución devuelta aunque el último estado SA sea peor.
- Aceptación probabilística determinista con RNG controlado en pruebas.
- Reinserción estricta por pareja; empate conserva original; tamaño y factibilidad preservados.
- Mejor global nunca empeora.
- Monitor interno SA desactivado; monitor externo recibe solo generaciones GA.
- Rearme exige nueva evidencia y no dispara inmediatamente por el historial anterior.
- Contabilidad de evaluaciones sin omisiones ni doble conteo según el contrato elegido.
- Baselines existentes y corridas previas no se modifican.

## 8. Decisiones aún abiertas

1. Implementación GA base y carpeta/módulo del nuevo híbrido.
2. Tamaño de población y parámetros GA.
3. Configuración externa DTW/DDTW completa.
4. Presupuesto global y definición exacta de una evaluación, incluyendo inicialización y reparación.
5. Orden de refinamiento si el presupuesto no alcanza para todos; política para un nivel parcial.
6. Detalle de rearme e historiales de umbrales.
7. Revisión bibliográfica de SA/MKP, citas y justificación de parámetros.
8. Instancias, semillas, repeticiones y frecuencia del baseline periódico.

No lanzar una campaña, promover una configuración a definitiva ni modificar código por el solo hecho de leer este plan.

## 9. Decisiones textuales que sustentan el plan

- "1. empezaremos cin la que yo he trabajado la de 3 condiciones de paciencia, despues veremos las otras 2 okey?, 2 no sele puede hacer busqueda local a cada individuo? no entendi lo que me dijiste 3. parametros mas comunes en la literature 4. al reinsertar seria comparar las soluciones del ga con las del sa y ver cuales son las mejores?"
- "ya si por pareja"
- "hagamos 10 vecinos y mas niveles de temperatura hasta q llegue a un criterio de termino"
- "ya documenta todo bien en un .md apra hacerlo despues"
