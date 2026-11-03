# Plan: GWO–DE integrado con control DTW/DDTW para MKP binario

Handoff de diseño para un híbrido GWO–DE sobre MKP binario, con la trayectoria del mejor fitness gobernando parámetros mediante DTW o DDTW. No implementa código, no borra nada y no reporta corridas. WOA–ABC está retirado: no es alternativa, baseline ni variante (`planes/futuro_mcdp/README.md`).

## Lectura rápida

1. Tratar como confirmada solo la intención integrada y la propuesta de cambiar parámetros y luego llenar una ventana nueva.
2. No confundir eso con el abort de epoch de `hybrid_mkp/mh/gwo.py` ni con el rolling de `DTW_optimization-main`.
3. No implementar hasta cerrar operador, disparador, reinicio del historial y perfil. Este archivo no afirma superioridad ni optimalidad.

| Marca | Significa |
|---|---|
| Confirmado | Intención recuperada de la conversación. No es código. |
| Observado | Leído en las fuentes de la sección 15. |
| Sugerido | Contrato de este handoff para implementar después. No acordado. |
| Abierto | Elección humana pendiente. |

## 1. Objetivo y motivación

**Confirmado:** el trabajo previsto es un GWO–DE integrado para el MKP binario, con control de parámetros por DTW/DDTW. La población busca un subconjunto de ítems factible que maximice el beneficio. La reparación produce factibilidad; no sustituye al objetivo.

**Motivación, no resultado:** un perfil fijo puede explorar de más o explotar una meseta. La curva del mejor fitness es una señal barata de progreso o estancamiento. DTW/DDTW compara esa curva con una rampa y una constante para decidir parámetros. La hipótesis por evaluar es si una ventana fresca tras cada cambio describe mejor el régimen nuevo que una ventana rodante mezclada. No está demostrado.

**Confirmado como propuesta, no como implementación:** al detectar estancamiento se cambian parámetros y no se vuelve a decidir hasta llenar una ventana nueva de W observaciones. Cita recuperada:

> sipo si cada vez que se detecte estancamiento cambio de parametros hasta q se llene la nueva ventana no?

La lectura usada aquí es un cambio y después W observaciones nuevas, no retocar parámetros en cada paso mientras la ventana se llena. El disparador exacto, si se borra el historial auxiliar y qué perfil se aplica siguen **abiertos**.

## 2. Qué no es este híbrido

| Pieza observada | Qué hace | Qué no es |
|---|---|---|
| `hybrid_mkp/mh/gwo.py` | GWO con LB2. Si el monitor dispara, aborta el epoch. | No cambia `a`, `F`, `CR` ni `G`. No llama a `adaptar_G_por_dtw`. No contiene DE. |
| `DTW_optimization-main/mkp_common/runner.py` | Una clase MH por corrida. Tras cada paso, si la ventana está lista, adapta el paso siguiente. | GWO y DE no comparten población. No espera una ventana nueva. |
| Runners continuos B1/B3 | Intensidad `q` en cada iteración lista, sobre las últimas W. | No son la política de ventana fresca. |
| `adaptive_mh_selection` | Selección entre metaheurísticas. | No es control de parámetros de un integrado. |
| WOA–ABC | Retirado del flujo activo. | Fuera de alcance. No reactivarlo desde este plan. |

No hay módulo DE en `hybrid_mkp/mh/`. Las ecuaciones DE de la sección 8 salen de `DTW_optimization-main/mkp_common/mh/de.py`, no de un híbrido existente.

## 3. GWO observado en el pipeline híbrido

`hybrid_mkp/mh/gwo.py`, defaults: población 30, 300 iteraciones, 10 epochs, `v_max=6`. Esos números son defaults del archivo, no presupuesto aprobado del híbrido nuevo.

Inicialización: posiciones uniformes en `[-1, 1]`. El bit inicial es 1 solo si la coordenada es `> 0`; cero y negativos pasan a 0. Después se repara. Eso no es la sigmoide de `mkp_common`.

Cada lobo, en cada iteración:

```text
a = 2 - 2 * it / max(1, iterations - 1)    # solo si no se sustituye el paso
A = 2*a*r1 - a
C = 2*r2
D = |C ⊙ X_lider - X|
X_lider_nuevo = X_lider - A ⊙ D
X_new = (X_alpha + X_beta + X_delta) / 3
X_new = clip(X_new, -v_max, v_max)
```

La posición real se escribe siempre. LB2 voltea el binario actual con L1 y con L2, repara ambas y se queda con la de mayor valor. Empate L1/L2: gana L2, porque el reemplazo es `val_1 > val_2`. El binario del lobo se reemplaza si `nueva_val >= fitness` del lobo; empate sí cambia bits. El mejor global solo mejora con `>` del fitness de alpha. Por tanto el real puede moverse aunque el binario se rechace, y un empate binario puede cambiar la solución factible sin mejorar el valor.

Los líderes son vistas de fila (`X_alpha = posiciones[alpha_idx]`). Si el índice de un líder se actualiza en el mismo barrido, los lobos posteriores ven la fila nueva. El orden es el índice de población, no el rango. **Sugerido, no implementado:** copiar alpha, beta y delta antes del barrido. No hay medición de que el alias explique algún resultado.

Con monitor activo, `G1`, `G2` y `G3` quedan en sus valores iniciales. `fire` incrementa el contador y hace `break`. La etiqueta verbal de explorar/explotar solo se imprime. `adaptar_G_por_dtw` está definida en `lb2/binarization.py` y exportada en `lb2/__init__.py`. En `hybrid_mkp`, `lb2` y `DTW_optimization-main` no hay otra llamada; el export no ejecuta la función. Un archivo reservado `nul` impidió un barrido único de todo el árbol.

`use_stagnation` default es verdadero, pero `stag_cfg` default es `None`. El monitor solo existe si ambas cosas están puestas. Si no, `interpolar_G` mueve G hacia el extremo. Ese es el camino default si el caller no inyecta configuración.

`dtw_info_hist` se arma y no se guarda en `GWOEpochResult`. No es un export usable del diagnóstico detallado.

## 4. LB2, reparación y la advertencia de G2

Fórmulas observadas en `lb2/binarization.py`:

```text
L1 = clip(-G1 * v / (v_max - G2) + G3, 0, 1)
L2 = clip( G1 * v / (v_max - G2) + G3, 0, 1)
```

Si `rand < L`, el bit actual se invierte. No se descarta la solución ni se muestrea una sigmoide desde cero. Cada candidata se repara. Defaults de `GWOParams`: `G1` 0.5→1.0, `G2` 0.5→7.2, `G3` 0.5→0.0.

**Advertencia, no fallo observado:** `G2_f=7.2` supera `v_max=6`. Si la interpolación acerca G2 a 7.2, `v_max - G2` cambia de signo e invierte la pendiente de L1/L2. El único guardia es `abs(denom) < 1e-12`; no implica una excepción. No se ejecutó una corrida para este documento, así que no se afirma un crash ni un iteración concreta de cruce.

Reparación de `hybrid_mkp/mkp_core/repair.py` y de `mkp_core/repair.py`, misma estrategia: densidad `min_i(p_j / r_ij)`, documentada en `hybrid_mkp/mkp_core/problem.py`. Infinitos por división entre cero pasan a 10× el máximo finito. Mientras sea infactible se apagan ítems de menor densidad; después se encienden, de mayor a menor, los que caben. La inicialización usa la copia `hybrid_mkp`; LB2 llama a `mkp_core.repair`. No se afirma identidad byte a byte de los dos módulos.

`DTW_optimization-main/mkp_common/problem.py` también expulsa bajo y reinserta alto, pero su densidad es `p / media(r_ij / b_i)`. El orden puede diferir. Un híbrido nuevo no debe mezclar las dos densidades en la misma corrida. **Sugerido:** una sola reparación, la de densidad mínima por restricción, para GWO y DE. **Abierto**, porque cambia la comparabilidad con las corridas sigmoide ya existentes.

## 5. GWO y DE en DTW_optimization-main

`mkp_common/config.py`: `window=20`, `use_ddtw=True`, banda 2, `min_slope=2`, umbrales adaptativos, población 20, 1000 iteraciones, 31 epochs. Son referencia de ese paquete, no parámetros cerrados del híbrido.

Las estrategias eligen **una** `MH_CLASS`. GWO y DE se ejecutan por separado. La binarización es sigmoide, no LB2:

```text
prob = 1 / (1 + exp(-clip(X, -10, 10)))   # GWO; DE no clipea el exponente
bit = 1 si rand < prob
```

Perfiles binarios fijos, observados:

| Modo | GWO | DE/rand/1/bin |
|---|---|---|
| Explotar | `a=0.5` | `F=0.5`, `CR=0.9` |
| Explorar | `a=2.0` | `F=0.9`, `CR=0.3` |

`adapt(fire)` conmuta esos extremos. `adapt_continuous(q)` interpola, con `q` recortado a `[0, 1]`:

```text
a  = 0.5 + 1.5 q
F  = 0.5 + 0.4 q
CR = 0.9 - 0.6 q
```

B3, `continuous_simple/runner.py`, `scale=2`, `eps=1e-10`:

```text
q = 1 - clip(D2 / (theta_c * scale + eps), 0, 1)
```

B1, `continuous_complex/runner.py`, `k=5`, centro 0.5:

```text
q = sigmoide(5 * (delta / (theta_delta + eps) - 0.5))
```

Esos runners aplican `q` en la misma iteración lista y el efecto cae en el **paso siguiente**. La ventana es `best_so_far[-W:]`. No reinician historial ni sostienen el perfil W pasos. Copiar ese loop no implementa la cita de la ventana fresca.

DE observado: mutación `v = x_r1 + F*(x_r2 - x_r3)`, cruce binomial con al menos una coordenada del mutante, clip a `[-v_max, v_max]`, sigmoide, reparación, y aceptación `>=` que escribe continuo y binario juntos. GWO de ese paquete siempre reemplaza al lobo; el histórico solo mejora con `>`. Inicialización DE: continuo uniforme en `[-6, 6]`. Inicialización GWO de ese paquete: bits aleatorios y reparación, sin vector continuo persistente.

El monitor de `mkp_common/monitor.py` y el de `dtw_stagnation.py` comparten la idea D1/D2/delta y las tres condiciones, pero no los defaults. El de la raíz usa ventana 40, `use_ddtw=False`, `plateau_max=15`, `patience=8`. No intercambiar configuraciones sin decirlo.

## 6. Señal temporal: DTW y DDTW

Ambos monitores reciben un escalar por paso y, si no mejora al último valor guardado, reescriben la observación con ese último valor e incrementan `no_improve_len`. La serie comparada es el mejor histórico, monótona no decreciente. No es el fitness medio, no es diversidad y no es `historial_inst` de alpha. Una meseta del elite puede coexistir con movimiento de la población. No interpretar D2 como pérdida de diversidad.

Sobre las últimas W observaciones listas:

```text
D1 = distancia(X, rampa que parte en X[0])
D2 = distancia(X, constante X[0])
delta = D1 - D2
```

Si `min_slope` es 0, la pendiente automática es `0.01 * max(1, |X[-1]-X[0]|) / W`. La banda Sakoe–Chiba limita el alineamiento.

DTW acumula diferencias absolutas de nivel. DDTW aplica la misma DTW a la primera diferencia, con `prepend=X[0]`, así que la primera derivada muestral es 0. DDTW mira la forma de los cambios; al comparar directamente dos mesetas a distinta altura, sus diferencias pueden coincidir aunque sus niveles no. En este monitor las referencias parten en X[0], de modo que no debe deducirse de ese ejemplo una ventaja automática frente a DTW. `use_ddtw` elige cuál distancia se usa. No son el mismo detector con otro nombre, ni uno está elegido para el híbrido.

`fire` observado, cuando se usa, exige a la vez meseta (`no_improve_len >= plateau_max`), cercanía a la constante (`D2 <= theta_c`) y lejanía de la rampa (`D1 >= theta_r` o `delta >= theta_delta`), repetida `patience` veces. Si falla una condición, la racha vuelve a 0. Antes de W observaciones, `ready=False` y no hay fire. Los umbrales adaptativos esperan al menos 10 distancias; si no, usan fracciones fijas de W. A3 (`D2 <= theta_c`) y B1/B3 no usan ese `fire`. Ninguno quedó elegido.

## 7. Ventana fresca: línea de tiempo y presupuesto restante

**Observado, rolling:** el paso `t` usa el perfil vigente; la observación `t` entra en la ventana; si ya hay W puntos, la adaptación afecta a `t+1`. La ventana de la decisión siguiente sigue incluyendo puntos del régimen anterior.

**Sugerido para hacer operativa la cita, no acordado en el detalle:**

```text
t_c     último paso con perfil P; su observación puede disparar
        el cambio se aplica después del paso, como en los runners
t_c+1   primer paso con P'
t_c+W   se completa la W-ésima observación generada con P'
t_c+W+1 primer paso que podría usar un perfil P'' decidido en t_c+W
```

La ventana nueva sugerida empieza en `t_c+1`. Incluir la observación que disparó el cambio mezclaría un punto de P. La cita no pide eso. Hace falta distinguir dos conteos, ambos **abiertos** como contrato final: W pasos nuevos antes de poder decidir, y un paso más antes de que la segunda decisión afecte la búsqueda. No es un bug hasta que el contrato lo fije.

Presupuesto restante, sugerido y distinto del abort observado:

- El warm-up inicial tampoco decide antes de W observaciones. No es un reinicio; es la misma regla de evidencia.
- Si tras `t_c` quedan menos de W iteraciones, no hay segunda decisión. Se termina con P'. No abortar el epoch porque la ventana no alcanza: eso copiaría el `break` observado, que este plan no adopta.
- Una decisión en la última iteración no tiene paso posterior. Queda como diagnóstico, no como cambio de búsqueda.
- Las W observaciones nuevas consumen presupuesto. No son un período gratis.
- No se reinicia el mejor global ni la población. La cita habla de parámetros y de ventana, no de borrar el incumbente.

**Abierto, y no implicado por la cita:** reiniciar `no_improve_len`, `trigger_streak` y los historiales de percentiles. Si no se reinician, los umbrales adaptativos arrastran el régimen viejo aunque X sea nueva. Hay que registrarlo antes de implementar.

## 8. Integración: población compartida o rotación

La variante de integración está **propuesta, no acordada**. Las ecuaciones de cada operador sí están observadas por separado.

**Población compartida, lectura sugerida de “integrado”.** Cada individuo tiene `x_real`, `x_bin` y fitness. Un paso aplica un operador al real, propone binarios, repara y selecciona. GWO usaría el promedio de tres líderes; DE, `DE/rand/1/bin` sobre el mismo real. No está acordado el calendario: solo GWO, solo DE, alternar, o que ambos propongan y compitan.

El emparejamiento real/binario tampoco está unificado. GWO híbrido escribe el real siempre y el binario solo con `>=`. DE de `mkp_common` escribe ambos solo si el trial gana. Mezclarlos en un bucle hace que el real signifique cosas distintas según el operador. Unificar es una decisión previa, no un arreglo ya justificado. LB2 usa el real como velocidades de volteo del binario actual; la desincronización es el comportamiento híbrido observado, no un defecto medido.

**Rotación.** Bloques sucesivos de GWO y de DE, pasando la población o solo el mejor. Es más cercano a ejecutar las clases ya separadas, o a seleccionar metaheurística, que a un estado compartido. Pasar solo el mejor pierde al resto. Pasar la población exige un mapa: el GWO de `mkp_common` no guarda continuo, el GWO híbrido inicia en `[-1, 1]` y luego clipea a `[-6, 6]`, y DE inicia en `[-6, 6]`. Sin ese mapa, la rotación no está especificada.

Otras mutaciones DE (`best/1`, `current-to-best`, `rand/2`) no están en el código citado y no se proponen como la variante. La variante explícitamente no acordada es: población dual compartida, GWO de tres líderes y `DE/rand/1/bin`, con el calendario de operador abierto.

## 9. Pseudocódigo conceptual

No existe esta función. `politica_operador`, `disparador` y `perfil_siguiente` son huecos.

```text
cargar instancia, semilla y presupuesto de candidatas reparadas
iniciar población dual, mejor global y perfil P
ventana = vacía

mientras quede presupuesto y no se alcance el límite de generaciones:
    op = politica_operador(estado)          # abierto
    propuestas = un paso de op bajo P
    reparar cada candidata; contar esa evaluación aunque se rechace
    actualizar individuo según la regla de aceptación aún abierta
    actualizar mejor global solo con mejora estricta
    agregar mejor_global.fitness a ventana

    si |ventana| < W:
        continuar
    señal = distancias(ventana, rampa, constante, DTW o DDTW)
    si disparador(señal):                   # abierto: no reutilizar fire sin decidirlo
        P = perfil_siguiente(P, señal)      # abierto
        ventana = vacía                     # no reutilizar las W que dispararon
        # no borrar mejor global
        # historiales de umbral: abierto

devolver mejor factible, configuración, contadores y series
```

Si el presupuesto se agota a mitad de un paso de población, no inventar individuos no evaluados ni seguir en silencio. La política de corte parcial está abierta.

## 10. Valores de referencia, no valores cerrados

| Tema | Referencia observada | Uso en el híbrido |
|---|---|---|
| W, DDTW | 20 y verdadero en `mkp_common`; 40 y falso en `dtw_stagnation.py` | Configurable. Ninguno adoptado. |
| Perfiles GWO/DE | 0.5/2.0 y F/CR 0.5/0.9 frente a 0.9/0.3 | Extremos de referencia, no óptimos. |
| `q` B1/B3 | Fórmulas de la sección 5 | Otro mecanismo. No es la ventana fresca. |
| G LB2 | 0.5/1.0, 0.5/7.2, 0.5/0.0, `v_max=6` | Si se usan, revisar G2 contra `v_max` antes de interpolar. |
| Población e iteraciones | 30×300×10 épocas, o 20×1000×31 | No son el presupuesto del experimento futuro. |
| Semilla | `SEMILLA=1` en config DTW; epochs usan `ep+1` | Registrar la que se fije. |

Todo lo anterior debe quedar en la configuración exportada. Cambiar W cambia el costo del warm-up y el retraso mínimo entre cambios.

## 11. Contabilidad de evaluaciones

**Sugerido, no implementado:** una evaluación es el beneficio devuelto por una candidata ya reparada.

- Inicialización: una por individuo.
- LB2: dos por individuo y por iteración, aunque la selección individual rechace ambas o una.
- Sigmoide de `mkp_common`: una por individuo y por paso.
- Reparaciones de inyección o mutación, si se reutilizan, cuentan aparte.
- Los intentos internos de apagar o encender ítems no son evaluaciones extra.
- DTW, ordenar líderes y precalcular densidades no cuentan.
- El real actualizado sin aceptar binario igual ya gastó las reparaciones LB2.

Comparar por número de iteraciones un LB2 (2N reparaciones) con una sigmoide (N) no es el mismo presupuesto. **Sugerido:** tope común de candidatas reparadas, semillas emparejadas e idéntica reparación. El tope concreto está abierto. No se afirma que el código actual lleve ese contador.

## 12. Diagnóstico a exportar

No hay un esquema de exportación del híbrido porque el híbrido no existe. `GWOEpochResult` tampoco guarda `dtw_info_hist`. Para una implementación futura, guardar al menos:

- instancia, semilla, presupuesto, configuración completa y versión de fuentes;
- mejor fitness, solución factible y brecha contra la referencia identificada, sin llamarla óptimo certificado;
- serie del mejor histórico y, por separado, fitness instantáneo de alpha o del trial;
- D1, D2, delta, umbrales, `ready`, disparo, DTW o DDTW, y tamaño de ventana;
- perfil activo, operador del paso, iteración de cada cambio y observaciones acumuladas desde el cambio;
- evaluaciones por fase, rechazos, empates y cambios que no alcanzaron a afectar un paso;
- corte por presupuesto: individuos no procesados, sin rellenarlos.

## 13. Validación, baselines y ablaciones

Esto es un plan de comprobación, no una campaña ejecutada. Una corrida de humo no valida el método. Los mejores de OR-Library son referencia, no certificado automático.

Antes de experimentar, pruebas del contrato que se cierre:

- no hay cambio de perfil antes de W observaciones;
- tras un cambio, las W que dispararon no vuelven a disparar solas;
- el paso que produjo el disparo todavía usó el perfil anterior;
- con menos de W pasos restantes no hay otro cambio y la corrida no aborta por eso;
- una decisión en la última iteración no altera la solución;
- el mejor global no empeora;
- toda candidata contada es factible tras reparar;
- el contador no omite rechazos ni duplica la reparación elegida;
- líderes copiados, solo si se acepta esa sugerencia;
- baselines viejos y corridas previas no se modifican.

Comparaciones sugeridas, una sola diferencia por contraste, mismas semillas y mismo tope:

1. GWO solo con perfil fijo de referencia.
2. DE solo con F/CR fijos de referencia.
3. El mismo integrado sin cambiar parámetros.
4. Rolling B1 o B3 ya existente, etiquetado como mecanismo distinto, no como este híbrido.
5. Abort de epoch, solo para contrastar contra control de parámetros, no como método propuesto.

Ablaciones posibles, no ordenadas ni aprobadas: DTW contra DDTW; rolling contra ventana fresca; población compartida contra rotación; LB2 contra sigmoide; `fire` de tres condiciones contra A3 o contra intensidad continua. No lanzarlas desde este documento.

## 14. Límites y decisiones abiertas

La ventana fresca retrasa la siguiente reacción al menos W pasos y puede gastar presupuesto en un mal perfil. Es un costo del diseño, no un defecto medido. La reparación cambia la propuesta del operador; el fitness no es el del vector crudo. No hubo revisión bibliográfica en este handoff. No hay resultados nuevos que citar.

Decisiones que bloquean una implementación fiel:

1. Población compartida o rotación, y el mapa de estado si hay rotación.
2. Calendario del operador: GWO, DE, alternancia o competencia de propuestas.
3. Regla única o separada para escribir real y binario.
4. LB2 o sigmoide, y una sola función de reparación.
5. Disparador: tres condiciones, A3, B1, B3 u otro. La cita no lo nombra.
6. Perfil destino: extremos fijos, interpolación `q`, parámetros G, o un subconjunto.
7. Reinicio o no de racha, meseta e historial de umbrales.
8. Contrato off-by-one: cuándo la decisión puede afectar la búsqueda.
9. Presupuesto, tamaño de población, W, DDTW, banda, paciencia y semillas.
10. Corte si el presupuesto termina a mitad de la población.
11. Si se copian líderes y si G2 puede superar `v_max`.

No promover ninguna celda de referencia a configuración definitiva por haberla documentado.

## 15. Fuentes verificadas

- `hybrid_mkp/mh/gwo.py`
- `hybrid_mkp/mkp_core/problem.py`
- `hybrid_mkp/mkp_core/repair.py`
- `mkp_core/repair.py`
- `lb2/binarization.py`
- `lb2/__init__.py`
- `dtw_stagnation.py`
- `DTW_optimization-main/mkp_common/config.py`
- `DTW_optimization-main/mkp_common/runner.py`
- `DTW_optimization-main/mkp_common/monitor.py`
- `DTW_optimization-main/mkp_common/base.py`
- `DTW_optimization-main/mkp_common/mh/gwo.py`
- `DTW_optimization-main/mkp_common/mh/de.py`
- `DTW_optimization-main/mkp_common/problem.py`
- `DTW_optimization-main/continuous_simple/runner.py`
- `DTW_optimization-main/continuous_complex/runner.py`
- `planes/futuro_mcdp/README.md`

Los comentarios de `lb2` mencionan notebooks `kirito/` que no están en este árbol. No se usaron como fuente.

## Siguiente paso

Cerrar por escrito las decisiones de la sección 14. Después, y en otro cambio, implementar contra esas pruebas. Commit y publicación no forman parte de este documento.
