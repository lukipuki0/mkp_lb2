| Desarrollo | de Disen˜os | H´ıbridos | para |     |
| ---------- | ----------- | --------- | ---- | --- |
Optimizaci´on
| Comparacio´n      | estad´ıstica | de algoritmos | estoc´asticos | de  |
| ----------------- | ------------ | ------------- | ------------- | --- |
| optimizaci´on:    | gu´ıa paso   | a paso        |               |     |
| Prof. Dr. Emanuel | Vega         |               |               |     |
2025-OII464-1-Desarrollo de Disen˜os H´ıbridos para Optimizaci´on
EscueladeIngenier´ıaInform´atica
PontificiaUniversidadCato´licadeValpara´ıso

Motivaci´on y objetivo

Motivaci´on
• Los algoritmos de optimizaci´on estoc´asticos (p. ej., PSO, GA,
| GWO,          | WOA, etc.)     | producen     | resultados   | aleatorios por corrida. |
| ------------- | -------------- | ------------ | ------------ | ----------------------- |
| • Compararlos | requiere       | separar      | variabilidad | de diferencias          |
| reales        | de desempen˜o. |              |              |                         |
| • Esta clase  | entrega        | un protocolo | reproducible | para: disen˜o           |
experimental, verificaci´on de supuestos, elecci´on de pruebas,
| taman˜o | de efecto, | potencia y | reporte. |     |
| ------- | ---------- | ---------- | -------- | --- |
Prof.Dr.EmanuelVega 1

| Objetivo del | protocolo         |         |                |     |
| ------------ | ----------------- | ------- | -------------- | --- |
| Dado un      | par de algoritmos | A y B y | un conjunto de |     |
problemas/instancias:
| 1. Disen˜ar | corridas   | y recolectar m´etricas. |                |     |
| ----------- | ---------- | ----------------------- | -------------- | --- |
| 2. Plantear | hip´otesis | H vs. H (bilateral      | o unilateral). |     |
0 1
| 3. Verificar | normalidad, | homogeneidad | de varianzas | y   |
| ------------ | ----------- | ------------ | ------------ | --- |
apareamiento.
4. Elegir prueba (param´etrica o no param´etrica) acorde a los
supuestos.
| 5. Estimar   | taman˜o    | de efecto y ICs. |                     |     |
| ------------ | ---------- | ---------------- | ------------------- | --- |
| 6. Controlar | mu´ltiples | comparaciones    | cuando corresponda. |     |
| 7. Reportar  | resultados | de manera        | clara y replicable. |     |
Prof.Dr.EmanuelVega 2

Disen˜o experimental

| Disen˜o: | unidades | experimentales |     |     | y r´eplicas |     |     |
| -------- | -------- | -------------- | --- | --- | ----------- | --- | --- |
•
|     | Instancias | (o funciones/benchmarks) |     |     | son | las | unidades sobre |
| --- | ---------- | ------------------------ | --- | --- | --- | --- | -------------- |
|     | las que    | comparamos.              |     |     |     |     |                |
•
|     | Por instancia, | ejecutar  | cada | algoritmo   |                | con R | r´eplicas |
| --- | -------------- | --------- | ---- | ----------- | -------------- | ----- | --------- |
|     | independientes | (semillas |      | distintas). | Recomendaci´on |       | t´ıpica:  |
|     | R ∈ [20,30].   |           |      |             |                |       |           |
•
|     | Mantener      | condiciones | comparables |           | (presupuesto |          | de       |
| --- | ------------- | ----------- | ----------- | --------- | ------------ | -------- | -------- |
|     | evaluaciones, | tiempo,     | criterios   | de        | paro).       |          |          |
|     | • CRN (Common | Random      |             | Numbers): | usar         | semillas | pareadas |
|     | por instancia | para        | reducir     | varianza  | al comparar  |          | A vs B.  |
Prof.Dr.EmanuelVega 3

| M´etricas de evaluaci´on |     |     |     |
| ------------------------ | --- | --- | --- |
•
| Valor objetivo      | final (menor      | es mejor | / mayor es mejor). |
| ------------------- | ----------------- | -------- | ------------------ |
| • Best-so-far       | a un presupuesto  | fijo.    |                    |
| • Tiempo-a-objetivo | (time-to-target). |          |                    |
•
Tasa de ´exito (al alcanzar umbral).(binaria;verMcNemar/Fishersi
aplica)
•
| Medidas       | de robustez (varianza, | IQR) | y estabilidad |
| ------------- | ---------------------- | ---- | ------------- |
| (consistencia | entre corridas).       |      |               |
Prof.Dr.EmanuelVega 4

| Preparaci´on   | de            | datos    |           |              |     |            |      |
| -------------- | ------------- | -------- | --------- | ------------ | --- | ---------- | ---- |
| Para cada      | instancia     | i y      | algoritmo | k ∈ {A,B}:   |     |            |      |
| 1. Calcular    | descriptivos: |          | media,    | mediana,     | sd, | IQR, min,  | max. |
| 2. Visualizar: |               | boxplots | y Q–Q     | (inspecci´on | de  | normalidad | y    |
outliers).
| 3. Decidir | apareamiento: |     | comparar | sobre | las | mismas |     |
| ---------- | ------------- | --- | -------- | ----- | --- | ------ | --- |
r´eplicas/semillas (pareado) o sobre muestras independientes.
| 4. Elegir    | unidad   | de            | comparaci´on: | (i)              | corridas | crudas,        | (ii) |
| ------------ | -------- | ------------- | ------------- | ---------------- | -------- | -------------- | ---- |
| estad´ıstico |          | por instancia |               | (p. ej., mediana |          | por instancia) | y    |
| luego        | comparar | entre         | instancias.   |                  |          |                |      |
Prof.Dr.EmanuelVega 5

Hip´otesis

| Planteamiento | de hip´otesis |     |
| ------------- | ------------- | --- |
Sea X desempen˜o de A y Y de B (menor es mejor). Dos opciones
t´ıpicas:
Bilateral
| H : [X]    | = [Y] vs      | H : [X] ̸= [Y]. |
| ---------- | ------------- | --------------- |
| 0          |               | 1               |
| Unilateral | (superioridad | de A)           |
| H : [X]    | ≥ [Y] vs      | H : [X] < [Y].  |
| 0          |               | 1               |
Recomendaci´on: usar bilateral salvo que exista una justificaci´on
| previa s´olida | para unilateral. |     |
| -------------- | ---------------- | --- |
Prof.Dr.EmanuelVega 6

Verificaci´on de supuestos

Normalidad
• Para disen˜os pareados: evaluar normalidad de las diferencias
| d = X −Y | .   |     |     |
| -------- | --- | --- | --- |
| i i      | i   |     |     |
•
| Pruebas recomendadas: | Shapiro–Wilk        | (muestras |     |
| --------------------- | ------------------- | --------- | --- |
| pequen˜as/medianas)   | o Anderson–Darling. |           |     |
•
| Complementar          | con Q–Q plots. | Con n grande, | las pruebas |
| --------------------- | -------------- | ------------- | ----------- |
| detectan desviaciones | triviales      |               |             |
⇒ considerarrobustez/transformaciones.
Prof.Dr.EmanuelVega 7

| Homoscedasticidad | (igualdad | de varianzas) |     |     |
| ----------------- | --------- | ------------- | --- | --- |
•
| Disen˜os | independientes: | Levene o Brown–Forsythe |     | para |
| -------- | --------------- | ----------------------- | --- | ---- |
| (X) =    | (Y).            |                         |     |      |
•
Disen˜os pareados: no aplica directamente (la prueba t pareada
| usa varianza | de d ). |     |     |     |
| ------------ | ------- | --- | --- | --- |
i
| • Si no | hay homoscedasticidad: | usar Welch | (param´etrico) | o   |
| ------- | ---------------------- | ---------- | -------------- | --- |
| pruebas | no param´etricas.      |            |                |     |
Prof.Dr.EmanuelVega 8

| Disen˜o: | pareado | vs independiente |
| -------- | ------- | ---------------- |

| Disen˜o experimental: | pareado | vs independiente | (ejemplo | vi- |
| --------------------- | ------- | ---------------- | -------- | --- |
sual)
Pareado Independiente
| AlgoritmoA | AlgoritmoB | AlgoritmoA | AlgoritmoB |     |
| ---------- | ---------- | ---------- | ---------- | --- |
(s e m il la si, p r e- (semillass1,s2,...) (semillast1,t2,...)
| s u pu e sto fi jo ) | (mismasi) |     |     |     |
| -------------------- | --------- | --- | --- | --- |
|                      | s 1       | s 1 | t 1 |     |
s 1
|     | s 2 | s 2 | t 2 |     |
| --- | --- | --- | --- | --- |
s 2
|     | s 3 | s 3 | t 3 |     |
| --- | --- | --- | --- | --- |
s 3
Enpareadosecomparanpares(Ai,Bi)conmismasemilla;enindependienteson
muestrasseparadas.
| Prof.Dr.EmanuelVega |     |     |     | 9   |
| ------------------- | --- | --- | --- | --- |

| Caso    | 1 — Disen˜o | pareado | (paired) |     |     |
| ------- | ----------- | ------- | -------- | --- | --- |
| Cu´ando | usarlo      |         |          |     |     |
•
|     | Puedes | controlar semillas | o fuentes | de aleatoriedad | en ambos |
| --- | ------ | ------------------ | --------- | --------------- | -------- |
algoritmos.
• Quieres reducir varianza comparando bajo el mismo escenario
estoc´astico.
| C´omo | implementarlo |                     |     |     |     |
| ----- | ------------- | ------------------- | --- | --- | --- |
|       | 1. Elige R    | semillas: {s ,...,s | }.  |     |     |
|       |               | 1                   | R   |     |     |
2. Corre A y B con cada s (presupuesto/criterios id´enticos).
i
|     | 3. Para cada | instancia, calcula | d   | =A −B (o B | −A segu´n |
| --- | ------------ | ------------------ | --- | ---------- | --------- |
|     |              |                    | i   | i i i      | i         |
convenci´on).
Prof.Dr.EmanuelVega 10

| Caso | 1         | — Disen˜o    | pareado | (paired)           |     |                     |     |
| ---- | --------- | ------------ | ------- | ------------------ | --- | ------------------- | --- |
|      | An´alisis | estad´ıstico |         |                    |     |                     |     |
|      | •         | Normalidad   | de      | {d }: Shapiro–Wilk |     | / Anderson–Darling. |     |
i
• Si normal: t pareada + IC de la diferencia; taman˜o de efecto:
|     |     | Cohen’s | d pareado. |     |     |     |     |
| --- | --- | ------- | ---------- | --- | --- | --- | --- |
•
|     |     | Si no normal:  |     | Wilcoxon signed-rank; |                     | taman˜o | de efecto: r o |
| --- | --- | -------------- | --- | --------------------- | ------------------- | ------- | -------------- |
|     |     | Hodges–Lehmann |     | (mediana              | del desplazamiento) |         | con IC.        |
Reporte
|     | •   | Cita d¯o | mediana(d) | + IC95%, | p-valor, | taman˜o | de efecto, |
| --- | --- | -------- | ---------- | -------- | -------- | ------- | ---------- |
|     |     | nu´mero  | de pares   | R.       |          |         |            |
• Incluye gr´afico de diferencias emparejadas o Bland–Altman.
Prof.Dr.EmanuelVega 11

| Caso 2 — Disen˜o | independiente |     |     |
| ---------------- | ------------- | --- | --- |
Cu´ando usarlo
•
| No controlas | (o no coinciden) | las semillas | entre algoritmos. |
| ------------ | ---------------- | ------------ | ----------------- |
•
Los resultados provienen de implementaciones/entornos diferentes.
C´omo implementarlo
| 1. Corre A  | con semillas {s ,...,s | } y B con           | {t ,...,t }. |
| ----------- | ---------------------- | ------------------- | ------------ |
|             | 1                      | RA                  | 1 RB         |
| 2. Verifica | normalidad por grupo   | y homoscedasticidad |              |
(Levene/Brown–Forsythe).
Prof.Dr.EmanuelVega 12

| Caso 2 — Disen˜o | independiente |     |     |     |     |
| ---------------- | ------------- | --- | --- | --- | --- |
An´alisis estad´ıstico
•
| Normal       | + varianzas     | iguales:    | t cl´asica. |                  |          |
| ------------ | --------------- | ----------- | ----------- | ---------------- | -------- |
| • Normal     | + varianzas     | desiguales: | Welch.      |                  |          |
| • No normal: | Mann–Whitney    |             | U; si       | formas/varianzas | difieren |
| mucho:       | Brunner–Munzel. |             |             |                  |          |
•
| Taman˜o        | de efecto: | Hedges   | g/Cohen              | d (param´etrico), |     |
| -------------- | ---------- | -------- | -------------------- | ----------------- | --- |
| Vargha–Delaney |            | A /Cliff | δ (no param´etrico). |                   |     |
12
Reporte
| • Media/mediana |           | por grupo | + IC95%,   | p-valor (ajustado | si hay |
| --------------- | --------- | --------- | ---------- | ----------------- | ------ |
| mu´ltiples      | pruebas), | taman˜o   | de efecto. |                   |        |
•
Box/violin plots por grupo; indicar R ,R y criterios de paro.
A B
Prof.Dr.EmanuelVega 13

Selecci´on de la prueba

| Mapa | de decisi´on (dos | algoritmos, | una instancia) |     |
| ---- | ----------------- | ----------- | -------------- | --- |
No
|     | ¿Pareado? |     | Independiente |     |
| --- | --------- | --- | ------------- | --- |
S´ı
No
|     | Normalidadendi? |            | Normalidad?       | Mann–WhitneyU |
| --- | --------------- | ---------- | ----------------- | ------------- |
|     |                 | No         | S´ı               |               |
|     | tpareada        | WilcoxonSR | Varianzasiguales? | Welch         |
tcl´asica
Consejo:usarpruebasbilateralespordefecto;unilaterals´oloconjustificaci´on.
Prof.Dr.EmanuelVega 14

| Mapa | de decisi´on | — notas | r´apidas |     |
| ---- | ------------ | ------- | -------- | --- |
•
|     | Pareado          | ⇒ probar diferencias | d i (Shapiro–Wilk             | sobre d). i |
| --- | ---------------- | -------------------- | ----------------------------- | ----------- |
|     | • Independiente: | normalidad           | por grupo; homoscedasticidad: |             |
Levene/Brown–Forsythe.
|     | • No normal:   | Wilcoxon (pareado); | Mann–Whitney | (indep.); |
| --- | -------------- | ------------------- | ------------ | --------- |
|     | Brunner–Munzel | si formas           | difieren.    |           |
•
Normal (indep.): t cl´asica (varianzas iguales) o Welch (desiguales).
Prof.Dr.EmanuelVega 15

| Comparaci´on |     |     | en mu´ltiples | instancias |     | (2 algoritmos) |     |     |
| ------------ | --- | --- | ------------- | ---------- | --- | -------------- | --- | --- |
• Estrategia comu´n: resumir por instancia (p. ej., mediana de R
corridas) y obtener pares (A ,B ) para i = 1,...,n instancias.
i i
•
|     | Pruebas |     | recomendadas: |     |     |     |     |     |
| --- | ------- | --- | ------------- | --- | --- | --- | --- | --- |
•
|     |     | Normalidad |     | en diferencias | d   | ⇒ t pareadasobred. |     |     |
| --- | --- | ---------- | --- | -------------- | --- | ------------------ | --- | --- |
|     |     |            |     |                | i   |                    |     | i   |
•
|                     |                | No     | normal   | ⇒ Wilcoxon | signed-ranksobred. |                 | i         |     |
| ------------------- | -------------- | ------ | -------- | ---------- | ------------------ | --------------- | --------- | --- |
|                     | • Alternativa  |        | robusta: | estimador  | de                 | Hodges–Lehmann  |           | del |
|                     | desplazamiento |        |          | (mediana   | de todas           | las diferencias | pareadas) |     |
|                     | con            | IC por | rank.    |            |                    |                 |           |     |
| Prof.Dr.EmanuelVega |                |        |          |            |                    |                 |           | 16  |

¿2 algoritmos y mu´ltiples instancias
• No param´etrico y ampliamente usado: Friedman (o Aligned
Friedman) para comparar k algoritmos sobre n instancias.
• Mejora de potencia: estad´ıstico de Iman–Davenport.
• Alternativa con covariables: Quade.
• Post-hoc: Nemenyi (todas vs todas), Holm o Shaffer (ajuste
m´as potente) frente a un control.
• Reportar diagramas de rangos cr´ıticos y s ajustados.
Prof.Dr.EmanuelVega 17

F´ormulas clave

t pareada y Welch
| t pareada | sobre | d = X | −Y (pareado, | normal): |     |
| --------- | ----- | ----- | ------------ | -------- | --- |
|           |       | i i   | i            |          |     |
d¯
|       |                | t =          | √           | , gl = n−1 |     |
| ----- | -------------- | ------------ | ----------- | ---------- | --- |
|       |                |              | s d / n     |            |     |
|       | d¯= 1 (cid:80) |              |             |            |     |
| donde |                | d i y s d su | desviaci´on | t´ıpica.   |     |
n
| t de Welch | (independiente, |           | normal, | varianzas | desiguales): |
| ---------- | --------------- | --------- | ------- | --------- | ------------ |
|            |                 |           |         | (cid:16)  | (cid:17)2    |
|            |                 |           |         | s2        | s2           |
|            |                 | X¯ −Y¯    |         | X         | + Y          |
|            |                 |           |         | n X       | n Y          |
|            | t =             |           | , gl    | ≈         |              |
|            |                 | (cid:113) |         | 4         | 4            |
|            |                 | s 2 s     | 2       | s X       | + s Y        |
|            |                 | X +       | Y       | 2(n       | 2(n          |
|            |                 | n X n     | Y       | n X −1)   | n Y −1)      |
|            |                 |           |         | X         | Y            |
Prof.Dr.EmanuelVega 18

Wilcoxon SR, Mann–Whitney U y Brunner–Munzel
Wilcoxon signed-rank (pareado, no normal): ordenar |d |, asignar
i
rangos, sumar rangos con signo; el estad´ıstico W se compara con
su distribuci´on (o aproximaci´on normal para n grande).
Mann–Whitney U (independiente, no normal):
n (n +1)
U = m´ın(U ,U ), U = n n + X X −R
X Y X X Y 2 X
donde R es la suma de rangos del grupo X.
X
Brunner–Munzel: prueba robusta para dominancia estoc´astica
cuando las distribuciones difieren en forma/varianza.
Prof.Dr.EmanuelVega 19

| Taman˜o | de efecto | e IC |
| ------- | --------- | ---- |

| Taman˜o de | efecto (param´etrico) |     |     |     |     |
| ---------- | --------------------- | --- | --- | --- | --- |
X¯ −Y¯
| • Cohen | d (independiente, |     | varianzas | iguales): d | = , con |
| ------- | ----------------- | --- | --------- | ----------- | ------- |
s
p
| s la sd | combinada. |     |     |     |     |
| ------- | ---------- | --- | --- | --- | --- |
p
• Hedges g: correcci´on de sesgo de d para muestras pequen˜as.
| •       |                     |      |                   |               | d¯. |
| ------- | ------------------- | ---- | ----------------- | ------------- | --- |
| Cohen   | d pareado:          | usar | sd de diferencias | s d y         |     |
| • IC de | d/g: aproximaciones |      | basadas           | en noncentral | t o |
bootstrap.
Prof.Dr.EmanuelVega 20

| Taman˜o | de efecto |     | (no param´etrico) |     |     |     |
| ------- | --------- | --- | ----------------- | --- | --- | --- |
•
|     | Vargha–Delaney |     | : prob. de | que X supere | a Y (common |     |
| --- | -------------- | --- | ---------- | ------------ | ----------- | --- |
language).
|     | = 1 indica | dominancia | de X; | = 0.5 indica | indiferencia. |     |
| --- | ---------- | ---------- | ----- | ------------ | ------------- | --- |
•
|     | Cliff δ:   | diferencia | de probabilidades | P(X | > Y)−P(X | < Y). |
| --- | ---------- | ---------- | ----------------- | --- | -------- | ----- |
|     | Relaci´on: | δ =        | 2−1.              |     |          |       |
Z
•
|     | r para | Wilcoxon/Mann–Whitney: |     | r = | √ . |     |
| --- | ------ | ---------------------- | --- | --- | --- | --- |
N
|     | • Desplazamiento |       | de Hodges–Lehmann:   |     | mediana | de  |
| --- | ---------------- | ----- | -------------------- | --- | ------- | --- |
|     | {x −y            | } con | IC por ordenamiento. |     |         |     |
i j
| Prof.Dr.EmanuelVega |     |     |     |     |     | 21  |
| ------------------- | --- | --- | --- | --- | --- | --- |

| Intervalos de        | confianza  |         |             |
| -------------------- | ---------- | ------- | ----------- |
| • Medias/diferencias | de medias: | IC-t (o | Welch) para |
normalidad.
| • Medianas/desplazamiento: |                     | IC por rangos  | (Hodges–Lehmann) |
| -------------------------- | ------------------- | -------------- | ---------------- |
| o bootstrap                | percentil/BCa.      |                |                  |
| • Para o                   | δ: IC con bootstrap | re-muestreando | pares.           |
Prof.Dr.EmanuelVega 22

Mu´ltiples comparaciones

| Correcciones |     | por | mu´ltiples | comparaciones |     |     |     |     |
| ------------ | --- | --- | ---------- | ------------- | --- | --- | --- | --- |
• Si comparas muchos algoritmos/instancias, controla el error
|     | tipo              | I.  |     |              |     |              |                |     |
| --- | ----------------- | --- | --- | ------------ | --- | ------------ | -------------- | --- |
|     | • Holm–Bonferroni |     |     | (secuencial, |     | m´as potente | que Bonferroni |     |
cl´asico).
|     | • Benjamini–Hochberg |         |           | (control         |     | de FDR) | cuando interesa |     |
| --- | -------------------- | ------- | --------- | ---------------- | --- | ------- | --------------- | --- |
|     | limitar              | la tasa | de falsos | descubrimientos. |     |         |                 |     |
•
|                     | En        | Friedman: | usar      | post-hoc | (Nemenyi, | Holm,      | Shaffer) | con s |
| ------------------- | --------- | --------- | --------- | -------- | --------- | ---------- | -------- | ----- |
|                     | ajustados | y         | diagramas | de       | rangos    | cr´ıticos. |          |       |
| Prof.Dr.EmanuelVega |           |           |           |          |           |            |          | 23    |

| Potencia | y taman˜o | muestral |
| -------- | --------- | -------- |

| Potencia | (power)  |            | y taman˜o | muestral   |              |         |     |
| -------- | -------- | ---------- | --------- | ---------- | ------------ | ------- | --- |
|          | • Elegir | α (t´ıpico | 0.05)     | y potencia | deseada (0.8 | o 0.9). |     |
•
|     | Requiere      | suponer |              | un efecto m´ınimo | relevante     | (EMR) | — por     |
| --- | ------------- | ------- | ------------ | ----------------- | ------------- | ----- | --------- |
|     | ej., d        | = 0.5.  |              |                   |               |       |           |
|     |               |         |              |                   | (cid:18)(z +z | )s    | (cid:19)2 |
|     | •             |         |              |                   | 1−α/2         | 1−β   | d         |
|     | Aproximaci´on |         | (t pareada): | n                 | ≈             |       | .         |
EMR
• Pr´actica en optimizaci´on: R ∈ [20,30] r´eplicas por instancia
suele equilibrar costo y precisi´on; validar con an´alisis de
|                     | potencia | si  | es cr´ıtico. |     |     |     |     |
| ------------------- | -------- | --- | ------------ | --- | --- | --- | --- |
| Prof.Dr.EmanuelVega |          |     |              |     |     |     | 24  |

| Buenas | pr´acticas | y reporte |
| ------ | ---------- | --------- |

Buenas pr´acticas
| • Fijar y | publicar semillas | y   | scripts de | generaci´on | de  |
| --------- | ----------------- | --- | ---------- | ----------- | --- |
instancias.
| • Usar CRN | y pareo por | instancia | cuando | sea | posible. |
| ---------- | ----------- | --------- | ------ | --- | -------- |
•
| Mostrar       | tanto s como     | taman˜os       | de efecto   | e ICs.        |              |
| ------------- | ---------------- | -------------- | ----------- | ------------- | ------------ |
| • Acompan˜ar  | pruebas          | con gr´aficos: | diferencias |               | emparejadas, |
| bland-altman, | violin/boxplots, |                | curvas de   | convergencia. |              |
• Evitar p-hacking y reportar criterios de exclusi´on (outliers) a
priori.
Prof.Dr.EmanuelVega 25

| Plantilla de | reporte (ejemplo) |     |     |
| ------------ | ----------------- | --- | --- |
Descripci´on
Comparaci´on de A vs B en n instancias, R r´eplicas pareadas. M´etrica:
| valor objetivo | final (menor | es mejor). |     |
| -------------- | ------------ | ---------- | --- |
Hip´otesis
| H :[X]=[Y] | (bilateral). | Nivel α=0.05. |     |
| ---------- | ------------ | ------------- | --- |
0
Supuestos
Shapiro–Wilk sobre d: p =0.18 (no se rechaza normalidad). Se aplica t
i
pareada.
Resultados
| d¯=−0.92 | (IC95%: [−1.30,−0.54]), | t =−4.6, | p <0.001; Cohen |
| -------- | ----------------------- | -------- | --------------- |
d =−0.53.
p
Conclusi´on
A supera a B con una diferencia media de 0.92 unidades (efecto
Prof.mDro.dEmeraanudeloV)e,gaestad´ısticamente 26
significativa.

Ap´endice: casos especiales

Casos especiales
| • Tiempo-a-objetivo: |           |                | an´alisis | de supervivencia |                |
| -------------------- | --------- | -------------- | --------- | ---------------- | -------------- |
| (Kaplan–Meier),      |           | log-rank       | para      | comparar curvas. |                |
| • Tasa               | de ´exito | (binaria,      | pareado): | McNemar.         | Independiente: |
| Fisher               | exacta    | o chi-cuadrado |           | (si procede).    |                |
• Outliers: usar m´etricas robustas (mediana, IQR) y pruebas no
param´etricas; considerar transformaciones (log) si justificadas.
•
| Modelos | jer´arquicos/mixtos: |            |        | permiten modelar | variabilidad |
| ------- | -------------------- | ---------- | ------ | ---------------- | ------------ |
| entre   | instancias           | y corridas | (nivel | avanzado).       |              |
Prof.Dr.EmanuelVega 27

Referencias sugeridas
• Demˇsar, J. (2006). Statistical Comparisons of Classifiers over Multiple
Data Sets. JMLR.
• Garc´ıa, S., Fern´andez, A., Luengo, J., Herrera, F. (2010). A Study of
Statistical Techniques and Performance Measures for Genetics-Based
Machine Learning. Soft Computing.
• Derrac,J.,Garc´ıa,S.,Molina,D.,Herrera,F.(2011).APracticalTutorial
on the Use of Nonparametric Statistical Tests... Neurocomputing.
• McDonald, J.H. (2014). Handbook of Biological Statistics. (cap´ıtulos de
Wilcoxon/Mann–Whitney/ANOVA).
Prof.Dr.EmanuelVega 28

¿Preguntas?
