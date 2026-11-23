# Selective harmonic elimination in PUC-5 multilevel inverter using hybrid IGWO-DE algorithm

**Clave de cita:** `Sifat_2024`  
**PDF:** [PDF](./Engineering%20Reports%20-%202024%20-%20Sifat%20-%20Selective%20harmonic%20elimination%20in%20PUC%E2%80%905%20multilevel%20inverter%20using%20hybrid%20IGWO%E2%80%90DE.pdf) · **Bibliografía:** [referencias.bib](../referencias.bib)

| Campo | Dato |
| --- | --- |
| Autores | Zaid Sifat, Md Tahmid Hussain, Mohd Anas Khan, Md Shahbaz Hussain, Adil Sarwar, Mohd Tariq, Mehedi Hasan |
| Año | 2024 (recibido 23-09-2023, aceptado 03-03-2024) |
| Venue | Engineering Reports, 6(10), e12883 (Wiley) |
| DOI | 10.1002/eng2.12883 |
| Sección primaria | `07-hibridos-gwo-de` |
| Etiquetas | IGWO-DE; cruce binomial entre candidato GWO y donante DE; horarios sigmoides; SHE-PWM; dos variables |

## Aporte para el estado del arte

Muestra una tercera forma de juntar GWO y DE en el mismo individuo: ni en serie ni en competencia, sino mezclando por dimensión el candidato GWO y el donante DE con un cruce binomial. Es útil para la taxonomía del acoplamiento. Su evidencia experimental es débil.

## Problema y método

Buscan dos ángulos de conmutación de un inversor PUC-5 que fijen la fundamental y anulen el tercer armónico (ecs. 4-5, p. 7). El fitness es el cociente entre el tercer armónico y la fundamental, con restricciones manejadas por las reglas de Deb (ecs. 27-32, p. 12).

Para cada individuo, IGWO-DE calcula una posición GWO `W` con un factor de convergencia sigmoide `a = 2 / (1 + e^{k(t/tmax − 1/2)})`, `k ∈ [8, 12]` (ec. 25, p. 11), y un donante DE/best/1 `V = X_alpha + F'·(X_R1 − X_R2)` con `F' = 2 / (1 + e^{k·t/tmax})`, `k ∈ [8, 14]` (ecs. 23-24, p. 11). La posición siguiente sale de un cruce binomial entre `V` y `W` (ec. 33, p. 13). Los pasos 1-6 (pp. 12-13) no incluyen selección greedy: el resultado del cruce reemplaza al individuo. La ec. 26 (p. 11) describe el cruce entre `V` y la posición previa `X`, no entre `V` y `W`; la ec. 33 es la que coincide con los pasos.

`a` y `F'` son horarios abiertos en función de `t`; no responden a la búsqueda. No hay detección de estancamiento, reinicio ni archivo; los líderes se recalculan desde la población (p. 11). El texto no informa tamaño de población, número de iteraciones, valor de `CR` ni el `k` usado.

## Qué sostiene el texto

La Tabla 2 (p. 17) compara GWO, DE, GA e IGWO-DE con un valor por método para dos índices de modulación. Para `M = 0.75`, IGWO-DE logra 0.78 % de tercer armónico y THD 31.29 %. Para `M = 0.85`, 0.2 % y 29.68 %. No hay corridas repetidas, dispersión ni pruebas estadísticas. Validan los ángulos en un prototipo con DSP, con THD medida de 33.8 % y 30.6 % (pp. 17-18).

## Límites

**Hallazgos de los autores.** La conclusión (p. 19) afirma que el método determina los mejores ángulos y que se aplica a otros niveles de inversor; no declara limitaciones.

**Lectura crítica.** El texto (p. 15) dice que IGWO-DE "consistently outperforms" en THD, pero la propia Tabla 2 muestra para `M = 0.85` una THD de GA (27.26 %) menor que la de IGWO-DE (29.68 %), y un quinto armónico de IGWO-DE (24.74 %) mayor que el de DE (16.06 %). El problema tiene dos variables y un solo resultado por método, así que no permite separar el efecto del híbrido del azar. Faltan los parámetros necesarios para reproducirlo. La p. 7 describe el caso como "5-level cascaded H-bridge" aunque la topología estudiada es PUC-5.

## Relación con el GWO-DE actual

Comparte la generación de un candidato GWO y uno DE por individuo en cada iteración. Difiere en cómo se resuelven: aquí se mezclan por dimensión sin evaluar cada uno y sin selección; en el GWO-DE actual cada candidato se evalúa y gana el mejor, con aceptación greedy. El control de `a` y `F'` es de horario fijo, sin señal de trayectoria. Ilustra la opción de recombinación, no aporta evidencia comparativa entre opciones.

## Evidencia

- p. 1: resumen.
- p. 4: motivación del factor sigmoide y del `F` dinámico.
- p. 7: ecuaciones SHE (ecs. 4-5).
- pp. 9-10: DE estándar con selección greedy (ecs. 18-20), descrito como fondo.
- p. 10: argumento de hibridación (sección 6).
- p. 11: donante (ec. 23), `F'` (ec. 24), `a` sigmoide (ec. 25), cruce con `X` (ec. 26).
- p. 12: fitness y reglas de Deb; pasos 1-4.
- p. 13: paso 5 con cruce entre `V` y `W` (ec. 33); terminación por iteraciones.
- p. 15: lectura de resultados.
- p. 17: Tabla 2 y montaje experimental.
- p. 18: THD medida.
- p. 19: conclusión.
