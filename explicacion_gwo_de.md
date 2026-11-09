# GWO–DE para el MKP binario

GWO y DE buscan juntos, sobre una sola población, un subconjunto factible de ítems que maximice el beneficio. La curva del mejor fitness decide si los parámetros explotan o exploran. No es un GWO que aborta la época, ni un DE aparte, ni WOA–ABC.

El paquete vigente es `gwo_de_dtw/`. `gwo_de_sequential/` repite la misma idea con otro orden de operadores y no importa el primero. El contrato largo, en inglés, está en `gwo_de_dtw/README.md`. Esta nota es la lectura corta.

## Qué problema resuelve

El MKP binario elige ítems. Cada ítem suma beneficio y consume capacidad en varias mochilas. La solución que se evalúa es siempre binaria y factible.

GWO y DE no operan sobre bits. Operan sobre un vector real. Por eso cada individuo guarda tres cosas a la vez:

| Pieza | Para qué |
|---|---|
| `x_real` | Ahí hacen la aritmética GWO y DE. |
| `x_bin` | La solución que se repara y se evalúa. |
| `fit` | Beneficio de esa solución factible. |

La reparación no optimiza. Solo deja la solución factible: saca ítems de menor densidad hasta entrar en capacidad, y después mete los de mayor densidad que todavía caben. La densidad es `min_i(p_j / r_ij)`.

## Una generación

Al empezar la generación se copian los tres líderes: alpha, beta y delta. Esos tres quedan congelados para todo el paso GWO. No se reordenan a mitad de la barrida.

Cada operador propone un candidato real y después se binariza:

- GWO tira al individuo hacia los tres líderes. El parámetro `a` decide si se acerca o se aleja. `a` chico explota; `a` grande explora.
- DE/rand/1/bin arma un mutante `x_r1 + F (x_r2 - x_r3)` y cruza con probabilidad `CR`.

El candidato real no se acepta crudo. Se pasa a bits, se repara y se compara con el individuo actual. Entra solo si el fitness reparado es `>=`. El vector real, los bits y el fitness se escriben juntos, o no se escribe nada.

Hay dos formas de combinar los operadores.

| Modo | Qué pasa | Dónde vive | Default |
|---|---|---|---|
| Competitivo | Cada individuo recibe un candidato GWO y uno DE. Gana el mejor de los dos. | `gwo_de_dtw`, `hybrid_mode=competitive` | Sí |
| Secuencial | Primero una barrida GWO completa. Después una barrida DE sobre la población que dejó GWO. No compiten. | Opt-in en `gwo_de_dtw` (`--hybrid-mode sequential`) y, aparte, todo `gwo_de_sequential/` | No |

En secuencial, DE trabaja in place: un movimiento tardío puede usar una fila que DE ya aceptó en esa misma barrida. GWO no ve eso; sus líderes siguen siendo los del inicio.

El mejor global solo mejora con un `>` estricto. Un empate no lo reemplaza.

## Cómo el real se vuelve binario

Por defecto es LB2. El vector real no inventa bits desde cero: da la probabilidad de voltear cada bit actual.

```text
L1 = clip(-G1 * v / (v_max - G2) + G3, 0, 1)
L2 = clip( G1 * v / (v_max - G2) + G3, 0, 1)
```

Se prueban los dos volteos, se reparan los dos y se queda el mejor. Por eso un candidato LB2 cuesta 2 evaluaciones. `G1` y `G2` quedan fijos en 0.5. `G3` es la probabilidad base de volteo del perfil activo. Hoy los dos perfiles usan `G3 = 0.1`, así que en la práctica G3 no se mueve.

La alternativa es sigmoide: cada bit sale con `P(1) = 1 / (1 + exp(-v))`, un solo reparado por candidato, espacio `[-6, 6]`. No usa `G1`, `G2` ni `G3`. LB2 usa `[-100, 100]`.

## Quién cambia los parámetros

Un perfil fijo puede explorar de más o quedarse en una meseta. La señal es barata: la curva del mejor fitness hasta ahora, no la población entera.

Un monitor (`dtw_stagnation.py`, el mismo de la raíz) mira las últimas `W` observaciones. El default es DDTW, ventana 40. Compara esa curva con una constante y con una rampa. De ahí salen `D2_vs_const`, `delta` y el disparo `fire`.

Esa lectura elige un perfil conjunto. GWO y DE cambian juntos. No se resetea la población ni el mejor global.

| | Exploit (arranque) | Explore |
|---|---|---|
| `a` (GWO) | 0.5 | 2.0 |
| `F` (DE) | 0.5 | 0.9 |
| `CR` (DE) | 0.9 | 0.3 |
| `G3` (LB2) | 0.1 | 0.1 |

Exploit: pasos chicos de GWO, mutación moderada, cruce alto. Explore: GWO se aleja de los líderes, mutación más grande, cruce bajo.

Hasta que la ventana no está llena no hay decisión. La corrida sigue en exploit. Un cambio decidido después de la generación `g` se aplica en `g+1`.

## Las cuatro políticas

Las cuatro leen el mismo monitor y los mismos perfiles. Difieren en cómo traducen la señal.

| Política | Cómo entra a explorar | Después de un cambio |
|---|---|---|
| `binary_simple` | `D2_vs_const <= theta_c` | Ventana nueva. No decide hasta llenar otras `W`. |
| `binary_complex` | El monitor dispara (`fire`: meseta, constante, rampa, y paciencia seguida). Es el controlador original. | Igual: ventana nueva. |
| `continuous_simple` | Intensidad `1 - clip(D2 / (theta_c * 2), 0, 1)` | No resetea el monitor. Mezcla los dos perfiles. |
| `continuous_complex` | Sigmoide sobre `delta / theta_delta` | Tampoco resetea. Mezcla. |

En las binarias la vuelta a exploit, con ventana fresca, ocurre cuando una observación lista muestra mejora reciente (`no_improve_len < plateau_max`). Si la curva sigue plana, se queda explorando.

En las continuas no hay salto fijo. La intensidad va de 0 (exploit puro) a 1 (explore puro) y los parámetros son la mezcla lineal. El nombre del perfil es `explore` solo si la intensidad pasa 0.5. `changes` anota cuándo cambia ese nombre, no un reset del monitor.

`--dtw-kirito` es opt-in y solo afecta a las binarias. Un solo monitor para toda la corrida: en cada observación lista explora si el disparador sigue activo y, si no, vuelve a explotar. El default es ventana fresca (`dtw_kirito` apagado). No es el control de `DTW_optimization-main`: allá la ventana rueda y la binarización de referencia es sigmoide, no LB2.

## Cuánto cuesta una corrida

Se frena por generaciones, no por evaluaciones. El default es `N = 30` y 1000 generaciones.

| Binarización | Por generación | Total default |
|---|---|---|
| LB2 | `4N` (2 operadores × 2 candidatos) | 120030, contando las `N` iniciales |
| Sigmoide | `2N` (2 operadores × 1 candidato) | 60030 |

El secuencial cuesta lo mismo por generación: también hace un GWO y un DE por individuo. Lo que cambia es el orden y si compiten, no el número de reparados.

`python -m gwo_de_dtw.run` corre las cuatro políticas, misma instancia y misma semilla, cada una con su propia población. No hay flag para elegir una sola.

## Los dos paquetes

| | `gwo_de_dtw` | `gwo_de_sequential` |
|---|---|---|
| Rol | Diseño vigente. | Experimento aparte. |
| Default | Competitivo. | Siempre GWO y después DE. |
| Código compartido | Importa el monitor de la raíz. | Importa solo ese monitor. Parser y reparación propios. |
| Campaña | `python -m gwo_de_dtw.campaign` | `python -m gwo_de_sequential.campaign` |

La campaña planificada es la misma cuenta: 9 archivos × 3 índices × 31 semillas × 4 políticas = 3348 corridas. Guarda traza y gráficos solo del mejor fitness por política. Un empate se queda con la semilla menor. Un destino que ya existe se rechaza; no pisa resultados.

## Qué no afirmar

Una semilla no prueba que una política, ni el modo secuencial, ni Kirito, sea mejor. Los pilotos de `mknapcb1` índice 0, semilla 1, están anotados en `.engram/GWO_DE.md` como números de una corrida, no como ranking.

WOA–ABC no es baseline de este diseño. Vive en otra rama (`feat/binary-woa-abc-dtw`) y este flujo lo trata como retirado.

## Dónde seguir

| Si querés | Leé |
|---|---|
| Flags, JSON y gráficos | `gwo_de_dtw/README.md` |
| El secuencial aislado | `gwo_de_sequential/README.md` |
| El contrato que el código tiene que cumplir | `odd/tasks/gwo-de-dtw-hybrid.md` |
| Por qué no es el GWO de `hybrid_mkp` ni el rolling de `DTW_optimization-main` | `planes/GWO_DE_DTW_DDTW.md` |
| Correr el default | `python -m gwo_de_dtw.run` desde la raíz del repo |
