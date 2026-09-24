# Explicacion del DTW como Monitor de Estancamiento

## Concepto

El **Dynamic Time Warping (DTW)** mide la **similitud entre dos secuencias** alineandolas de forma no lineal. En este proyecto, se usa para comparar la **curva de convergencia del fitness** (ventana de los ultimos W valores del best-so-far) contra dos patrones de referencia:

- **Rampa ideal**: progreso constante (linea recta con pendiente positiva)
- **Meseta (constante)**: estancamiento total (linea plana)

La intuicion es: si la curva real se parece mas a una rampa, la MH esta progresando. Si se parece mas a una meseta, esta estancada.

---

## Metricas

### D1 — Distancia a la rampa ideal
```
D1 = distancia_DTW(ventana_fitness, rampa_ideal)
```
- **D1 bajo** → la ventana tiene forma de rampa → la MH esta **mejorando**
- **D1 alto** → la ventana no se parece a una rampa → posible estancamiento

### D2 — Distancia a la meseta
```
D2 = distancia_DTW(ventana_fitness, meseta)
```
- **D2 bajo** → la ventana es plana → la MH esta **estancada**
- **D2 alto** → la ventana no es plana → hay actividad

### Delta
```
delta = D1 - D2
```
- **delta < 0** → D1 < D2 → mas cerca de rampa que de meseta → **progreso**
- **delta > 0** → D1 > D2 → mas cerca de meseta que de rampa → **estancamiento**

### No se usa delta directamente para decidir

A diferencia de lo que podria parecer intuitivo, `delta > 0` NO dispara el cambio de parametros por si solo. Se usan **3 condiciones simultaneas** + un mecanismo de confirmacion (abajo).

---

## DDTW — Derivative Dynamic Time Warping

El DDTW aplica DTW sobre la **primera derivada** de las secuencias en vez de los valores absolutos.

```
DDTW(s, t) = DTW(diff(s), diff(t))
```

donde `diff(x) = [x[0], x[1]-x[0], x[2]-x[1], ...]`

**Por que usarlo**: La derivada captura la **forma/tendencia** (pendiente) independientemente de la magnitud absoluta. Si el fitness pasa de 1000 a 50000, el DTW estandar da distancias enormes. El DDTW compara pendientes, que son comparables sin importar la escala. Para deteccion de estancamiento, DDTW es mas robusto.

---

## Parametros del StagnationConfig

### `window` (ventana)
- **Que es**: Cantidad de iteraciones que se analizan a la vez. La ventana se desliza con el tiempo (siempre los ultimos W valores).
- **Default**: 30
- **En la version "original"**: 20
- **Como afecta**:
  - `window` pequeno (5-10): muy reactivo, sensible a ruido, puede disparar falsos positivos. Bueno para MH que convergen rapido.
  - `window` mediano (15-30): balance entre reactividad y estabilidad.
  - `window` grande (50+): lento para detectar estancamiento (necesita mucha historia). Bueno para MH con convergencia lenta y muchas iteraciones.
- **Efecto practico**: Con `window=20` y `max_iter=100`, el DTW no empieza a funcionar hasta la iteracion 20 (warm-up). De las 100 iteraciones, solo 80 son "activas".

### `band` (banda Sakoe-Chiba)
- **Que es**: Ancho maximo de desvio permitido en el alineamiento DTW. Restringe cuanto puede "estirarse" la secuencia para alinearse con la otra.
- **Default**: 0 → se autocalcula como 10% de `window`
- **En la version "original"**: 2
- **Como afecta**:
  - `band` pequeno (1-2): DTW muy restrictivo, el alineamiento es casi uno-a-uno. Mas sensible a diferencias locales. Menos costo computacional.
  - `band` grande (5+): DTW mas flexible, permite mas warping. Menos sensible al ruido. Mayor costo O(window * band).
  - `band` = 0: sin restriccion (DTW completo, O(W^2)).
- **Efecto practico**: Con `band=2`, el costo de DTW es muy bajo y la comparacion es practicamente local (la iteracion i solo se alinea con i-2..i+2). Esto da una medida mas "instantanea" de la forma.

### `min_slope` (pendiente minima de la rampa)
- **Que es**: Cuanto debe crecer el fitness por iteracion en la rampa ideal. Define que tan "empinada" es la rampa de referencia.
- **Default**: 0.0 → se autocalcula como 1% del rango / window
- **En la version "original"**: 2.0
- **Como afecta**:
  - `min_slope` bajo (~auto, ~0.01-0.1): la rampa ideal es CASI PLANA → cualquier mejora minima hace que D1 < D2 → el sistema tiende a ver "progreso" constantemente → **falsos negativos** (no detecta estancamiento real).
  - `min_slope` alto (2.0+): la rampa ideal es MUY EMPINADA → D1 siempre es alto → el sistema tiende a ver "estancamiento" constantemente → **falsos positivos**.
  - `min_slope` calibrado: debe ser cercano a la tasa de mejora real de la MH en fase de progreso.
- **Efecto practico**: Con `min_slope=2.0` en fitness de ~24000, la rampa ideal espera que el fitness mejore 2 unidades POR ITERACION. En 20 iteraciones son 40 unidades. Esto es mucho para un problema de optimizacion combinatoria → D1 es grande → delta es positivo casi siempre. Por eso las otras condiciones (plateau_max, cond_constant, cond_ramp) filtran los falsos positivos.

### `plateau_max`
- **Que es**: Cuantas iteraciones consecutivas sin mejora deben pasar para considerar que entramos en una meseta.
- **Default**: 15
- **En la version "original"**: 10
- **Como afecta**:
  - `plateau_max` bajo (3-5): el sistema considera estancamiento muy rapido. Puede reaccionar a pausas breves.
  - `plateau_max` alto (15-20): necesita evidencia fuerte de estancamiento. Mas conservador.
- **Relacion con max_iter**: Con `max_iter=100`, `plateau_max=10` significa que necesita el 10% de las iteraciones sin mejora. Con `max_iter=500`, puede ser mas alto sin problema.
- **Efecto practico**: Es la primera barrera anti-ruido. Si la MH mejora cada 3-4 iteraciones, `plateau_max=10` evita que se dispare.

### `patience`
- **Que es**: Cuantas veces consecutivas deben cumplirse las 3 condiciones para que `fire = True`.
- **Default**: 3
- **En la version "original"**: 2
- **Como afecta**:
  - `patience=1`: dispara en la primera confirmacion → mas reactivo, mas ruidoso.
  - `patience=2-3`: filtro anti-oscilacion. Un pico aislado de D1 no dispara.
  - `patience=5+`: muy conservador, puede tardar en reaccionar.
- **Efecto practico**: Es la segunda barrera anti-ruido. Junto con `plateau_max`, define la inercia del sistema.

### `use_ddtw`
- **True**: usa Derivative DTW (compara pendientes, no valores)
- **False**: usa DTW estandar (compara valores absolutos del fitness)
- **Recomendacion**: `True` para problemas donde el fitness puede variar mucho en magnitud entre ejecuciones.

### `adapt_thresholds`
- **True**: los umbrales `theta_c`, `theta_r`, `theta_delta` se calculan como **percentiles moviles** del historial de D1, D2, delta. Se autoadaptan a la escala del problema.
- **False**: usa umbrales fijos: `theta_c = 0.1*W`, `theta_r = 0.5*W`, `theta_delta = 0.3*W`.
- **Efecto practico**: Con `True`, el sistema es mas robusto a diferentes tamanos de instancia y escalas de fitness. Con `False`, los umbrales son mas predecibles pero pueden no ser adecuados para todos los problemas.

### `p_low` y `p_high`
- **Que son**: Percentiles usados cuando `adapt_thresholds=True`.
  - `theta_c = percentil(p_low, historial_D2)` → umbral bajo de D2
  - `theta_r = percentil(p_high, historial_D1)` → umbral alto de D1
  - `theta_delta = percentil(p_high, historial_delta)` → umbral alto de delta
- **Default**: p_low=30, p_high=70
- **Como afecta**: p_low mas bajo hace que `theta_c` sea mas estricto (D2 debe ser muy bajo para disparar). p_high mas alto hace que `theta_r` y `theta_delta` sean mas permisivos.

---

## Logica de Decision (las 3 condiciones)

El `fire = True` SOLO se dispara cuando las 3 condiciones siguientes se cumplen SIMULTANEAMENTE durante `patience` iteraciones consecutivas:

```
1. cond_plateau:  no_improve_len >= plateau_max
   "La MH lleva N iteraciones sin mejorar nada"

2. cond_constant: D2 <= theta_c
   "La curva de fitness se parece a una meseta (D2 bajo)"

3. cond_ramp:     D1 >= theta_r  OR  delta >= theta_delta
   "La curva NO se parece a una rampa (D1 alto) o el delta es alto"
```

Si las 3 se cumplen → `trigger_streak += 1`
Si alguna falla → `trigger_streak = 0` (reset)
Si `trigger_streak >= patience` → `fire = True`

**Por que 3 condiciones y no solo delta**: Si solo usaras `delta > 0`, el sistema oscilaria constantemente entre EXPLOIT y EXPLORE porque el ruido en la curva de fitness genera fluctuaciones. Las condiciones extra (plateau + D2 bajo) filtran el ruido y exigen evidencia acumulada.

---

## Flujo de la Auto-Adaptacion

```
iteracion t:
  fitness = mh.step()
  status = monitor.update(fitness)
  
  if status["fire"] and modo_actual == "exploit":
      mh.adapt(True)   → cambiar a EXPLORE
      
  if not status["fire"] and modo_actual == "explore":
      mh.adapt(False)  → volver a EXPLOIT
```

- **EXPLOIT**: parametros estandar de la MH (w=0.729, c1=1.49, c2=1.49 para PSO)
- **EXPLORE**: parametros que fuerzan exploracion (w=0.9, c1=2.5, c2=0.5)
- El cambio es por **transiciones** (no por estado). Si fire=True durante 30 iteraciones seguidas, los parametros cambian UNA vez (al entrar) y se quedan en EXPLORE.

---

## Ejemplo Concreto

Con los parametros de la version "original" (window=20, plateau_max=10, patience=2, min_slope=2.0, use_ddtw=True, adapt_thresholds=False, band=2):

```
iter 0-19:  warm-up. No hay datos para DTW.
iter 20:    D1=150, D2=90, delta=+60
            no_improve=3  (< plateau_max=10)
            → cond_plateau: NO → trigger_streak=0
iter 21-29: similar. La MH mejora esporadicamente, no_improve se resetea.
iter 30:    no_improve=11 (>= 10)  → cond_plateau: SI
            D2=85 (<= theta_c=2)   → cond_constant: SI
            D1=200 (>= theta_r=10) → cond_ramp: SI
            → trigger_streak=1
iter 31:    mismas 3 condiciones → trigger_streak=2 (>= patience=2)
            → FIRE = True → cambiar a EXPLORE
iter 32-50: en EXPLORE, la MH empieza a encontrar mejores soluciones
            no_improve se resetea a 0
            → cond_plateau: NO → trigger_streak=0 → fire=False
            → volver a EXPLOIT
```

---

## Relacion con los Parametros PSO

| Modo | w (inercia) | c1 (cognitivo) | c2 (social) | Efecto |
|------|-------------|----------------|-------------|--------|
| EXPLOIT | 0.729 | 1.49445 | 1.49445 | Particulas siguen al enjambre, convergen al gbest |
| EXPLORE | 0.9 | 2.5 | 0.5 | Particulas ignoran al enjambre estancado, exploran su propio pbest con mas inercia |

- **w alto (0.9)**: las particulas mantienen su direccion, "saltan mas lejos"
- **c1 alto (2.5)**: mas peso a la experiencia individual → diversidad
- **c2 bajo (0.5)**: menos peso al gbest estancado → escapar del optimo local
