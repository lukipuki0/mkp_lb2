# Parametros del DTW: Que miden y como los podes usar

## El DTW como termometro

Imaginate que la MH es un motor y el DTW es un tablero de instrumentos.
En cada iteracion, el DTW mira los ultimos W valores del fitness (la "ventana")
y te dice: "esta curva se parece a progreso o a estancamiento?"

Para responder, genera **6 numeros**. Los primeros 3 son **mediciones**
(lo que el sensor lee). Los otros 3 son **umbrales** (los limites que definen
"que es mucho" y "que es poco").

---

## Las 3 mediciones (el sensor)

### D1 — "Que tan lejos estas de progresar"

```
D1 = distancia_DTW(ventana_fitness, rampa_ideal)
```

La rampa ideal es una linea recta que sube. Si tu curva de fitness se parece
a esa rampa, D1 es bajo. Si no se parece, D1 es alto.

**Analogia**: Es como medir la distancia entre tu velocidad actual y la
velocidad crucero ideal. Si vas a 100 km/h y la ideal es 120, la distancia
es 20. Si vas a 30, la distancia es 90.

| D1 bajo | La MH ESTA mejorando (la curva sube como la rampa) |
|---------|-----------------------------------------------------|
| D1 alto | La MH NO esta mejorando (la curva no se parece a progreso) |

**Que lo afecta**:
- `window`: ventana mas grande = mide tendencia a largo plazo, mas suave
- `min_slope`: pendiente de la rampa. Si la rampa es muy empinada (min_slope alto),
  D1 siempre va a ser alto porque ninguna MH mejora TAN rapido. Si es muy
  baja, D1 siempre va a ser bajo porque cualquier mejora minima "cuenta"
- `use_ddtw`: si es True, compara PENDIENTES en vez de valores. Mas robusto
  porque no depende de la magnitud del fitness

**Para nuevas versiones**: D1 te dice "que tan bien le va a la MH". Un D1
que baja consistentemente = la MH esta convergiendo bien. Un D1 que sube =
se alejo del progreso. Podrias usar la TASA DE CAMBIO de D1 para anticipar
estancamiento antes de que ocurra.

---

### D2 — "Que tan estancado estas"

```
D2 = distancia_DTW(ventana_fitness, meseta)
```

La meseta es una linea plana. Si tu curva de fitness es plana, D2 es bajo.
Si tiene actividad (sube, baja, oscila), D2 es alto.

**Analogia**: Es como medir cuanto se mueve la aguja del velocimetro.
Si esta clavada en un numero, D2 es bajo. Si se mueve, D2 es alto.

| D2 bajo | La curva ES plana = la MH esta ESTANCADA |
|---------|-------------------------------------------|
| D2 alto | La curva NO es plana = hay actividad (puede ser mejora o ruido) |

**Que lo afecta**:
- `window`: ventana mas grande = necesita mas iteraciones planas para que D2 baje
- `band`: banda de warping. Con band alto, el DTW puede "estirar" la secuencia
  para alinearla con la meseta, haciendo que D2 baje mas facil

**Para nuevas versiones**: D2 es una medida DIRECTA de estancamiento. No
necesitas fire, no necesitas delta. Un D2 que se acerca a 0 = la MH se
freno. Podrias usar D2 directamente como input continuo: "mientras mas
bajo D2, mas agresiva la exploracion".

---

### delta — "Balance entre progreso y estancamiento"

```
delta = D1 - D2
```

Es la diferencia entre las dos distancias. Combina ambas mediciones en un
solo numero.

| delta < 0 | D1 < D2 = mas cerca de la rampa que de la meseta = PROGRESO |
|-----------|--------------------------------------------------------------|
| delta > 0 | D1 > D2 = mas cerca de la meseta que de la rampa = ESTANCAMIENTO |
| delta ~ 0 | Ambiguo = la curva no se parece ni a progreso ni a estancamiento |

**Analogia**: Si D1 mide "que lejos estas de ir bien" y D2 mide "que lejos
estas de estar parado", delta te dice "estas mas cerca de ir bien o de
estar parado?"

**Para nuevas versiones**: Delta es el candidato natural para normalizacion
continua. Podrias mapear delta a un score [0,1] con sigmoid o min-max y
usarlo para interpolar parametros de la MH en tiempo real. Tambien podrias
usar el HISTORIAL de delta para detectar patrones: si delta oscila mucho,
la MH esta al borde; si crece monotonamente, el estancamiento se profundiza.

---

## Los 3 umbrales (el termostato)

Los umbrales definen los limites para decidir. En la version actual (fire
binario) se usan para las 3 condiciones de fire. Pero para nuevas versiones,
pensalos como "puntos de referencia" que el sistema se auto-calibra.

### theta_c — "Que tan bajo es 'bajo' para D2"

```
theta_c = percentil(p_low, historial_D2)     # con adapt_thresholds=True
theta_c = 0.1 * window                       # con adapt_thresholds=False
```

Es el umbral que decide si D2 es "suficientemente bajo" para considerar que
la curva es plana. Se calcula como el percentil 30 del historial de D2.

**Que significa**: Si theta_c = 0.5, entonces D2 <= 0.5 implica "si, esto
es una meseta". Es adaptativo: se autocalibra al rango tipico de D2 que
genera esta MH en este problema.

**Para nuevas versiones**: theta_c te da una referencia de "que es normal"
para D2 en ESTE problema con ESTA MH. Podrias usar D2/theta_c como un
ratio normalizado: si D2/theta_c < 1, estas estancado respecto a tu propia
historia.

---

### theta_r — "Que tan alto es 'alto' para D1"

```
theta_r = percentil(p_high, historial_D1)    # con adapt_thresholds=True
theta_r = 0.5 * window                       # con adapt_thresholds=False
```

Es el umbral que decide si D1 es "suficientemente alto" para considerar que
la curva NO es una rampa. Se calcula como el percentil 70 del historial de D1.

**Que significa**: Si theta_r = 15.0, entonces D1 >= 15.0 implica "no, esto
no es progreso". Es el "estandar de progreso" minimo para esta corrida.

**Para nuevas versiones**: theta_r te da una referencia de "que tan lejos
del progreso ideal sueles estar". D1/theta_r > 1 = peor que lo habitual.
D1/theta_r < 1 = mejor que lo habitual. Otra vez, un ratio auto-normalizado.

---

### theta_delta — "Que tan alto es 'alto' para delta"

```
theta_delta = percentil(p_high, historial_delta)   # con adapt_thresholds=True
theta_delta = 0.3 * window                         # con adapt_thresholds=False
```

Es el umbral que decide si delta es "suficientemente positivo" para
considerar que hay estancamiento significativo. Percentil 70 del historial.

**Que significa**: Si theta_delta = 8.0, entonces delta >= 8.0 implica
"el estancamiento es peor que el 70% de las veces". Es relativo a la
historia de esta corrida.

**Para nuevas versiones**: theta_delta te da escala. delta/theta_delta es
un score normalizado: 0 = sin estancamiento, 1 = estancamiento tipico,
>1 = peor que lo habitual. Podrias usar esto directamente como intensidad
de exploracion.

---

## Los parametros de configuracion (las perillas)

Estos son los que se configuran en `StagnationConfig`. Afectan COMO se
calculan las 6 metricas de arriba.

### window (ventana de observacion)

**Que hace**: Cuantos valores del fitness mira el DTW en cada iteracion.
Siempre los ultimos W valores (ventana deslizante).

```
window=10  →  mira las ultimas 10 iteraciones (reactivo, ruidoso)
window=30  →  mira las ultimas 30 iteraciones (estable, lento)
window=50  →  mira las ultimas 50 iteraciones (muy estable, muy lento)
```

**Efecto en las metricas**: Window grande suaviza D1, D2 y delta. Window
chico los hace mas volatiles.

**Tradeoff clave**: Con `max_iter=100` y `window=20`, el DTW no arranca
hasta la iteracion 20 (warm-up). Solo 80 de 100 iteraciones son "activas".
Si window es demasiado grande relativo a max_iter, el DTW casi no actua.

**Para nuevas versiones**: Podrias usar MULTIPLES ventanas (corta + larga)
para tener dos perspectivas: una reactiva y una estable. O hacer window
adaptativo: empezar chico y crecer a medida que avanza la ejecucion.

---

### band (banda Sakoe-Chiba)

**Que hace**: Limita cuanto puede "estirarse" el alineamiento DTW.
Con band=2, la iteracion i solo se puede alinear con i-2..i+2.

```
band=1  →  alineamiento casi 1-a-1 (muy rigido, rapido)
band=3  →  algo de flexibilidad
band=0  →  sin restriccion (DTW completo, mas caro: O(W^2))
```

**Efecto en las metricas**: Band bajo hace que D1 y D2 capturen diferencias
mas locales. Band alto permite que el DTW "perdone" desfases temporales.

**Para nuevas versiones**: Band bajo da una medida mas "instantanea".
Band alto da una medida mas "tolerante". Para deteccion de estancamiento,
band bajo suele ser suficiente.

---

### min_slope (pendiente de la rampa ideal)

**Que hace**: Define que tan empinada es la rampa de referencia contra la
que se mide D1.

```
min_slope=0.0  →  se autocalcula como 1% del rango / window (muy suave)
min_slope=2.0  →  espera mejora de 2 unidades de fitness por iteracion
min_slope=10   →  espera mejora de 10 por iteracion (casi imposible)
```

**Efecto critico en D1**: Si min_slope es alto (ej: 2.0 con fitness ~24000),
la rampa ideal espera 2*20=40 unidades de mejora en la ventana. Esto es
MUCHO para optimizacion combinatoria. Resultado: D1 es siempre alto →
delta es siempre positivo → sin las 3 condiciones, el sistema veria
"estancamiento" todo el tiempo.

**Para nuevas versiones**: Un min_slope bien calibrado es clave. Si lo
auto-calculas (0.0), se adapta al problema. Si lo fijas, estas imponiendo
una expectativa de progreso. Podrias calcular min_slope DINAMICAMENTE
basandote en la tasa de mejora real de las ultimas iteraciones.

---

### use_ddtw (usar derivadas)

**Que hace**: Si True, calcula DTW sobre las PENDIENTES (primeras diferencias)
en vez de los valores absolutos del fitness.

```
use_ddtw=True   →  compara formas de curva (independiente de magnitud)
use_ddtw=False  →  compara valores absolutos (sensible a escala)
```

**Por que importa**: Si el fitness es 24000 y la rampa empieza en 24000
y sube a 24040, el DTW estandar mide distancias de 0 a 40. Si el fitness
es 500, mide distancias de 0 a algo mucho menor. El DDTW normaliza esto
comparando pendientes.

**Para nuevas versiones**: DDTW es casi siempre mejor para deteccion de
estancamiento. Pero si queres medir MAGNITUD del progreso (cuanto mejoro
en numeros absolutos), necesitas DTW estandar.

---

### plateau_max, patience (NO son del DTW — son de la logica de decision)

Estos parametros NO afectan las mediciones D1, D2, delta. Afectan la
DECISION de fire. Son especificos del enfoque fire binario.

- `plateau_max`: cuantas iteraciones sin mejora para considerar plateau
- `patience`: cuantas confirmaciones consecutivas para disparar fire

**Para nuevas versiones**: Si no usas fire binario, estos parametros NO
aplican. El DTW genera D1, D2, delta, y los thetas INDEPENDIENTEMENTE de
como decidas usarlos.

---

## Resumen: que metricas estan disponibles en cada iteracion

Despues del warm-up (iter >= window), en cada iteracion tenes:

| Metrica | Rango tipico | Que te dice |
|---------|-------------|-------------|
| D1 | 0 a ~inf | Distancia al progreso ideal |
| D2 | 0 a ~inf | Distancia al estancamiento total |
| delta | -inf a +inf | Balance progreso/estancamiento |
| theta_c | > 0 | Referencia adaptativa para D2 |
| theta_r | > 0 | Referencia adaptativa para D1 |
| theta_delta | cualquier | Referencia adaptativa para delta |

Ademas tenes:
- `no_improve_len`: iteraciones consecutivas sin mejora (contador simple)
- `trigger_streak`: racha de condiciones cumplidas (solo en fire binario)

---

## Ideas para combinar: ratios normalizados

Con las metricas y sus umbrales, podes construir scores normalizados:

```
score_estancamiento = D2 / (theta_c + eps)
  → < 1 = estancado respecto a tu historia
  → > 1 = activo respecto a tu historia

score_progreso = D1 / (theta_r + eps)
  → < 1 = mejor progreso que lo habitual
  → > 1 = peor progreso que lo habitual

score_balance = delta / (theta_delta + eps)
  → < 0 = progresando
  → 0-1 = estancamiento normal
  → > 1 = estancamiento severo (peor que el 70% de tu historia)
```

Estos scores son auto-normalizados (los thetas se adaptan al problema) y
van de negativo a positivo, lo que los hace ideales para interpolar
parametros de la MH de forma continua.
