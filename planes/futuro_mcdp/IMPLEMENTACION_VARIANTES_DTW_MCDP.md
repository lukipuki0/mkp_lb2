# Plan de implementación: variantes DTW para WOA--ABC en MCDP

## 1. Alcance correcto

El objetivo inmediato es conservar el solver MCDP existente en
`woa_abc/cooperativo_mcdp_dtw.py` y crear varias versiones de **cómo DTW cambia
sus parámetros**.

No se propone por ahora:

- convertir `woa_abc` en un solver MKP;
- sustituir la representación máquina→celda;
- crear una población WOA y otra ABC;
- cambiar la función objetivo del MCDP;
- copiar las metaheurísticas binarias descritas en `ALGORITMO_CENTRAL.md`.

Todas las versiones deben compartir exactamente:

- la misma representación latente continua;
- la misma reparación de asignaciones MCDP;
- la misma población única;
- las mismas fases WOA, ABC empleada, ABC observadora y scout;
- la misma evaluación y el mismo criterio de aceptación;
- el mismo presupuesto y las mismas semillas.

Solo deben variar dos cosas:

1. la política que interpreta las métricas DTW;
2. el conjunto de parámetros que esa política puede modificar.

La idea reutilizable más importante de `ALGORITMO_CENTRAL.md` es separar:

```text
sensor DTW -> política -> actuador de parámetros
```

## 2. Comportamiento actual de `woa_abc`

El MCDP minimiza costo. Como `StagnationMonitor` espera una señal creciente, el
solver entrega:

```python
status = monitor.update(-best_cost)
```

Cuando `status["fire"]` es verdadero y terminó el cooldown, la implementación
actual ejecuta un pulso de exploración:

| Parámetro | Cambio actual ante `fire` |
|---|---|
| `woa_exploration` | Multiplica por `1.35`, con máximo `2.50`. |
| `abc_phi_scale` | Multiplica por `1.30`, con máximo `2.50`. |
| `abc_guide_strength` | Multiplica por `0.60`, con mínimo `0.02`. |
| `abc_limit` | Multiplica por `0.70`, con mínimo `2`. |
| `random_reassignment_rate` | Multiplica por `1.8`, con máximo `0.35`. |
| población | Reinicia `rescue_fraction=0.25` de los peores individuos. |

Después de cada iteración, `_relax_state()` acerca gradualmente los parámetros
a sus valores base usando `adaptation_decay=0.90`.

Esta implementación corresponde a una primera versión que puede llamarse:

```text
legacy_fire_full
```

porque usa la alarma compuesta `fire` y cambia al mismo tiempo WOA, ABC,
diversificación y población.

### Ciclo actual

```text
1. Calcular momentum.
2. Consultar DTW con el mejor global obtenido hasta la iteración anterior.
3. Si fire=True, modificar parámetros y rescatar peores individuos.
4. Ejecutar WOA.
5. Ejecutar ABC empleada.
6. Ejecutar ABC observadora.
7. Ejecutar scout.
8. Registrar resultados.
9. Relajar los parámetros hacia la base.
```

### Aspectos que deben corregirse antes de comparar versiones

1. `_relax_state()` lleva `woa_exploration` y `abc_phi_scale` hacia `1.0`, no
   hacia `params.woa_exploration` y `params.abc_phi_scale`. Si el usuario cambia
   esos parámetros iniciales, la relajación converge a otro valor.
2. El factor WOA se aplica como multiplicador del parámetro temporal `a`. Al
   final de la corrida, `a` llega a cero y ningún multiplicador puede reactivar
   exploración: `0 * factor = 0`.
3. Un `fire` cambia cinco parámetros y además rescata población. Si mejora el
   resultado, actualmente no se puede atribuir la mejora a un mecanismo.
4. El estado DTW se consulta antes de las fases de búsqueda. El evento guardado
   con una iteración describe en realidad la curva producida hasta la iteración
   anterior.
5. El CLI permite configurar DTW, pero no elegir políticas ni desactivar el
   controlador como un tratamiento experimental explícito.
6. `stagnation_fires` cuenta pulsos ejecutados; no distingue una alarma DTW, una
   entrada de modo o el número de iteraciones en exploración.

Antes del refactor se debe conservar `legacy_fire_full` para poder reproducir
los resultados actuales. Las versiones nuevas usarán un ciclo con semántica de
eventos más clara.

## 3. Dos ejes para construir versiones

No conviene crear un archivo completo por cada variante. El solver debe recibir
dos opciones independientes.

### Eje A — Cuándo y con qué intensidad cambiar

```text
control_policy =
    no_dtw
    fire_pulse
    binary_simple
    binary_hysteresis
    four_state
    proportional
```

### Eje B — Qué componentes cambiar

```text
adapt_target =
    none
    woa
    abc
    woa_abc
    diversity
    full
```

La combinación de ambos ejes permite responder preguntas distintas:

- ¿DTW aporta algo frente al híbrido sin control?
- ¿Conviene reaccionar con pulsos o mantener modos?
- ¿La mejora viene de WOA o de ABC?
- ¿Cambiar parámetros basta o hace falta rescatar población?
- ¿Dos estados son suficientes o se necesitan cuatro intensidades?

## 4. Políticas DTW propuestas

### P0 — `no_dtw`

No crea monitor y no cambia parámetros. Es el baseline principal del híbrido.
El paso exponencial y el programa temporal original de WOA se mantienen porque
son parte del algoritmo, no del controlador DTW.

### P1 — `fire_pulse`

Usa la salida compuesta que ya calcula el monitor:

```text
fire = plateau
       AND cercanía a constante
       AND alejamiento de rampa
       durante patience observaciones
```

Cuando `fire=True`, aplica un pulso; después los parámetros se relajan hacia la
base. Debe existir en dos formas:

- `legacy_fire_full`: reproducción exacta del código actual;
- `fire_params_only`: mismo disparo, pero sin rescate ni reasignación aleatoria.

Esta pareja permite medir cuánto del comportamiento actual proviene del cambio
de parámetros y cuánto del reinicio de población.

### P2 — `binary_simple`

Usa la regla directa descrita en `ALGORITMO_CENTRAL.md`:

$$
mode_t=
\begin{cases}
explore,&D_{2,t}\leq\theta_{c,t},\\
exploit,&D_{2,t}>\theta_{c,t}.
\end{cases}
$$

No usa `status.fire`, `patience`, `D1`, `delta` ni `no_improve_len`. Puede
cambiar de modo en observaciones consecutivas, por lo que sirve como versión
reactiva simple.

### P3 — `binary_hysteresis` — política principal recomendada

Mantiene su propio modo, inicialmente `exploit`:

$$
mode_t=
\begin{cases}
explore,&mode_{t-1}=exploit \land \Delta_t\geq\theta_{\Delta,t},\\
exploit,&mode_{t-1}=explore \land \Delta_t\leq0,\\
mode_{t-1},&\text{en otro caso}.
\end{cases}
$$

La banda `0 < delta < theta_delta` evita alternancias rápidas. Esta política
ignora `status.fire`; `plateau_max` y `patience` quedan disponibles solo como
telemetría.

Es la mejor candidata inicial porque produce transiciones interpretables y
mantiene los parámetros en un régimen el tiempo suficiente para observar su
efecto.

### P4 — `four_state`

Clasifica la búsqueda en cuatro intensidades:

| Condición | Estado |
|---|---|
| `delta > theta_delta` | explorar mucho |
| `0 <= delta <= theta_delta` | explorar poco |
| `-theta_delta <= delta < 0` | explotar poco |
| `delta < -theta_delta` | explotar mucho |

Los parámetros se toman de cuatro perfiles predefinidos. Es una extensión de
P3; no debe implementarse primero porque duplica el número de regímenes a
calibrar.

### P5 — `proportional`

Convierte `delta` en una intensidad continua:

$$
s_t=clip\left(\frac{\Delta_t}{\max(|\theta_{\Delta,t}|,\varepsilon)},-1,1\right).
$$

- `s` cercano a `1`: máxima exploración;
- `s` cercano a `-1`: máxima explotación;
- `s` cercano a `0`: parámetros próximos a la base.

Para evitar ruido se recomienda suavizar:

$$
\bar s_t=\beta\bar s_{t-1}+(1-\beta)s_t,
\qquad \beta=0.8.
$$

Esta política elimina saltos discretos, pero es la versión más difícil de
explicar y calibrar. Debe quedar para una segunda etapa.

## 5. Actuadores: parámetros que puede cambiar cada versión

### A0 — `none`

No modifica nada. Se usa con P0.

### A1 — `woa`

Modifica solamente:

- escala de `a`;
- piso exploratorio de `a`;
- máximo permitido de `a`.

No se recomienda cambiar `exploration_probability` en la primera prueba. En el
código actual esa probabilidad elige la familia encircling/individuo aleatorio
frente a la espiral, pero la exploración real de la primera familia aún depende
de `abs(A)`.

### A2 — `abc`

Modifica solamente:

- `abc_phi_scale`;
- `abc_guide_strength`;
- `abc_limit` mediante un factor sobre el límite base.

Esto permite saber si DTW mejora principalmente la búsqueda vecinal y el uso de
scouts, sin alterar WOA.

### A3 — `woa_abc`

Combina A1 y A2, sin modificar la reparación ni reiniciar individuos. Es el
actuador principal recomendado.

### A4 — `diversity`

Modifica solamente:

- `random_reassignment_rate` usado durante reparación;
- rescate de una fracción de los peores al entrar en exploración.

Sirve para aislar el efecto de la diversificación explícita.

### A5 — `full`

Combina WOA, ABC y diversidad. Es el comportamiento más cercano a la versión
actual, pero no debe ser la única alternativa evaluada.

## 6. Perfiles de parámetros para MCDP

La implementación debe usar perfiles completos, no multiplicadores acumulados
sin un objetivo explícito.

Los valores siguientes son puntos iniciales de experimentación; todavía no son
hiperparámetros validados:

| Parámetro | `neutral` actual | `exploit` | `explore` |
|---|---:|---:|---:|
| `woa_a_scale` | 1.00 | 0.75 | 1.50 |
| `woa_a_floor` | 0.00 | 0.00 | 1.25 |
| `woa_a_cap` | 2.00 | 0.99 | 2.50 |
| `abc_phi_scale` | 1.00 | 0.60 | 1.50 |
| `abc_guide_strength` | 0.20 | 0.35 | 0.05 |
| `abc_limit_factor` | 1.00 | 1.25 | 0.50 |
| `random_reassignment_rate` | 0.04 | 0.02 | 0.12 |
| `rescue_fraction` | 0.25 en `fire` | 0.00 | 0.25 al entrar |

El coeficiente WOA debe pasar de:

```python
a = a_base * state.woa_exploration
```

a:

```python
a = min(profile.woa_a_cap, max(
    profile.woa_a_floor,
    a_base * profile.woa_a_scale,
))
```

El `woa_a_floor` permite que una entrada tardía a exploración tenga un efecto
real, aunque `a_base` ya esté próximo a cero.

El límite ABC debe calcularse siempre desde el valor base, no desde el valor
redondeado de la iteración anterior:

```python
effective_limit = max(
    params.min_abc_limit,
    round(base_limit * profile.abc_limit_factor),
)
```

Esto evita deriva acumulativa.

### Parámetros que deben permanecer fijos al principio

Para atribuir resultados al controlador, inicialmente no se deben adaptar:

- `b_spiral`;
- `exploration_probability`;
- `step_initial` y `step_final`;
- `momentum_factor`;
- tamaño de población;
- función de reparación;
- criterio codicioso de aceptación.

Después se puede estudiar una versión adicional donde DTW también module
momentum o el paso, pero no debe mezclarse con la primera comparación.

## 7. Versiones experimentales concretas

Estas son las versiones mínimas recomendadas:

| Versión | Política | Actuador | Rescate | Pregunta que responde |
|---|---|---|---|---|
| `M0_no_dtw` | P0 | A0 | no | ¿Qué logra WOA--ABC sin DTW? |
| `M1_legacy_fire_full` | P1 legacy | A5 | sí | ¿Qué hace exactamente la implementación actual? |
| `M2_fire_params_only` | P1 | A3 | no | ¿El pulso de parámetros basta sin reiniciar población? |
| `M3_simple_woa_abc` | P2 | A3 | no | ¿Una regla directa de meseta es suficiente? |
| `M4_hysteresis_woa` | P3 | A1 | no | ¿El efecto viene de WOA? |
| `M5_hysteresis_abc` | P3 | A2 | no | ¿El efecto viene de ABC? |
| `M6_hysteresis_woa_abc` | P3 | A3 | no | ¿Funciona el cambio coordinado de parámetros? |
| `M7_hysteresis_full` | P3 | A5 | solo al entrar | ¿El rescate agrega valor al cambio coordinado? |
| `M8_four_state_woa_abc` | P4 | A3 | no | ¿Cuatro intensidades superan a dos modos? |
| `M9_proportional_woa_abc` | P5 | A3 | no | ¿Una adaptación suave supera los cambios discretos? |

### Paquete mínimo para la primera corrida

No hace falta ejecutar las diez versiones desde el comienzo. Primero:

```text
M0, M1, M2, M4, M5 y M6
```

Con ese conjunto se puede separar:

- sin DTW frente al método actual;
- pulso actual con y sin rescate;
- adaptación de WOA solamente;
- adaptación de ABC solamente;
- adaptación coordinada WOA + ABC.

Después se agregan M3 y M7. M8/M9 quedan para una segunda etapa una vez que los
perfiles binarios estén calibrados.

## 8. Arquitectura de código

No se deben mantener diez copias de `cooperativo_mcdp_dtw.py`. Las variantes
deben compartir un único motor.

### Archivos propuestos

```text
woa_abc/
├── cooperativo_mcdp_dtw.py       # motor compartido MCDP
├── mcdp_dtw_control.py           # políticas, perfiles y actuadores
├── run_cooperative_mcdp.py       # ejecución de una configuración
├── run_cooperative_mcdp_batch.py # comparación de tratamientos
└── tests/
    ├── test_mcdp_dtw_control.py
    └── test_cooperative_mcdp_variants.py
```

### Tipos sugeridos

```python
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

class SearchMode(str, Enum):
    NEUTRAL = "neutral"
    EXPLOIT = "exploit"
    EXPLORE = "explore"

@dataclass(frozen=True)
class MCDPParameterProfile:
    woa_a_scale: float
    woa_a_floor: float
    woa_a_cap: float
    abc_phi_scale: float
    abc_guide_strength: float
    abc_limit_factor: float
    random_reassignment_rate: float
    rescue_fraction: float = 0.0

@dataclass(frozen=True)
class ControlDecision:
    mode: SearchMode
    intensity: float
    rescue: bool
    reason: str

class MCDPControlPolicy(Protocol):
    def reset(self) -> None: ...
    def decide(self, status: dict) -> ControlDecision: ...
```

El motor recibe una decisión y un `adapt_target`. Las ecuaciones de WOA y ABC
no deben contener condicionales específicos para `binary_simple`, histéresis o
cuatro estados.

### Configuración pública

Agregar a `CooperativeMCDPParams`:

```python
control_policy: str = "legacy_fire_full"
adapt_target: str = "full"
profile_neutral: MCDPParameterProfile = ...
profile_exploit: MCDPParameterProfile = ...
profile_explore: MCDPParameterProfile = ...
rescue_on_explore_entry: bool = True
standardize_control_timing: bool = False
```

Los defaults anteriores preservan compatibilidad. Para experimentos nuevos se
usará `standardize_control_timing=True` y una variante con nombre explícito.

## 9. Ciclo estándar para las versiones nuevas

El ciclo recomendado observa el resultado completo de WOA--ABC y decide los
parámetros de la siguiente iteración:

```text
inicializar población, mejor global y parámetros
crear monitor y política nuevos para el epoch
decision_actual = exploit

para cada iteración t:
    construir perfil efectivo(decision_actual, adapt_target)

    ejecutar fase WOA
    actualizar mejor global
    ejecutar ABC empleada
    ejecutar ABC observadora
    ejecutar scout
    actualizar mejor global

    registrar costo y parámetros usados en t
    status = monitor.update(-best_cost)

    si status.ready:
        decision_siguiente = policy.decide(status)
        registrar transición y sus métricas

        si se entra a explore y el actuador permite rescate:
            rescatar peores individuos, protegiendo el mejor

    decision_actual = decision_siguiente
```

Así, una decisión registrada en `t` se aplica en `t+1`. `M1_legacy_fire_full`
debe conservar el orden anterior para reproducir los resultados ya obtenidos;
no debe mezclarse silenciosamente con las versiones de tiempo estandarizado.

## 10. Semántica de eventos y resultados

Los contadores deben separarse:

```text
dtw_alarm_count       # veces que status.fire fue verdadero
adaptation_count      # acciones de cambio realmente ejecutadas
mode_entry_count      # entradas exploit -> explore
mode_exit_count       # salidas explore -> exploit
rescue_count          # eventos de rescate
rescued_individuals   # total de individuos reiniciados
```

Cada iteración debe registrar:

```text
iteracion
best_cost
instant_cost
woa_best
abc_best
dtw_ready
D1, D2, delta
theta_c, theta_r, theta_delta
mode_used
mode_decided_for_next
adapt_target
effective_parameters
objective_evaluations
```

Cada transición debe guardar:

```text
iteracion
modo_anterior
modo_nuevo
motivo
métricas DTW
perfil anterior
perfil nuevo
si hubo rescate
cantidad rescatada
```

Esto permite explicar si una adaptación fue seguida por una mejora real.

## 11. CLI propuesto

Una ejecución debería poder declarar completamente el tratamiento:

```bash
python -m woa_abc.run_cooperative_mcdp \
  --instance 1 \
  --cells 3 \
  --capacity 6 \
  --iterations 500 \
  --epochs 31 \
  --pop-size 30 \
  --control-policy binary_hysteresis \
  --adapt-target woa_abc \
  --no-rescue \
  --window 75 \
  --ddtw \
  --seed 42
```

Opciones mínimas nuevas:

```text
--control-policy
--adapt-target
--rescue / --no-rescue
--standard-control-timing / --legacy-control-timing
```

El nombre completo del tratamiento debe guardarse en JSON/CSV y formar parte
de la carpeta de resultados, por ejemplo:

```text
M6_hysteresis_woa_abc/
```

## 12. Uso del monitor DTW

Se puede conservar el preset MCDP actual:

```python
StagnationConfig(
    window=75,
    band=0,
    min_slope=0.1,
    plateau_max=15,
    patience=25,
    use_ddtw=True,
    adapt_thresholds=True,
    p_low=30.0,
    p_high=70.0,
)
```

También puede existir un preset compatible con `ALGORITMO_CENTRAL.md`:

```python
StagnationConfig(
    window=20,
    band=2,
    min_slope=2.0,
    plateau_max=4,
    patience=2,
    use_ddtw=True,
    adapt_thresholds=True,
    p_low=30.0,
    p_high=70.0,
)
```

Ese preset sirve para replicar la metodología, no implica que sea mejor para
MCDP.

Reglas comunes:

1. El monitor siempre recibe `-best_cost`.
2. Cada epoch crea un monitor y una política nuevos.
3. Ninguna política DTW cambia parámetros antes de `ready=True`.
4. Durante warm-up se usa el perfil `neutral` o `exploit`, pero la elección debe
   ser igual para todas las variantes comparadas y quedar registrada.
5. Debe haber muchas más iteraciones que `window`; se recomienda al menos
   `max(300, 4*window)`.
6. Los umbrales adaptativos comienzan después de diez observaciones válidas.
7. `moving_percentile()` utiliza todo el historial acumulado, no una segunda
   ventana móvil.
8. DDTW elimina el desplazamiento vertical, pero no toda dependencia de escala.

## 13. Pruebas necesarias

### Compatibilidad

- `M1_legacy_fire_full` reproduce el resultado actual con la misma semilla.
- Los defaults antiguos siguen siendo aceptados.
- MCDP continúa usando asignaciones máquina→celda, sin LB2.

### Políticas

- `binary_simple` explora con `D2 == theta_c`.
- `binary_simple` explota con `D2 > theta_c`.
- La histéresis entra con `delta == theta_delta`.
- La histéresis sale con `delta == 0`.
- La banda `0 < delta < theta_delta` conserva el modo.
- `reset()` devuelve cada política con estado al modo inicial.
- No hay transición durante warm-up.

### Perfiles y actuadores

- `adapt_target="woa"` no cambia parámetros ABC ni diversidad.
- `adapt_target="abc"` no cambia parámetros WOA ni diversidad.
- `adapt_target="woa_abc"` no rescata ni cambia reasignación aleatoria.
- `adapt_target="full"` aplica únicamente los campos autorizados.
- El perfil de explotación mantiene `abs(A) < 1` mediante `woa_a_cap < 1`.
- El piso exploratorio funciona cuando `a_base == 0`.
- El límite ABC siempre se deriva de `base_limit` y respeta el mínimo.
- La relajación converge a los valores configurados, no a constantes internas.

### Solver MCDP

- Toda asignación evaluada es válida y respeta capacidad.
- `historial` del mejor costo es monótono no creciente.
- El mejor global nunca se pierde durante scout o rescate.
- El mejor costo coincide con `inst.objective(mejor_solucion)`.
- Una decisión de `t` se refleja en `parametros_historial[t+1]`.
- La misma semilla reproduce costo, trazas y eventos.
- Monitor, política y perfiles se reinician entre epochs.

## 14. Diseño experimental

### Piloto

Primero ejecutar una selección pequeña de casos MCDP que incluya:

- dos y tres celdas;
- capacidades ajustadas y holgadas;
- instancias pequeñas y grandes;
- casos donde la versión actual tiene gap alto y bajo.

Usar el piloto para detectar configuraciones inválidas y fijar los perfiles. No
seguir ajustando parámetros después de observar el conjunto final.

### Comparación principal

Ejecutar el paquete mínimo:

```text
M0, M1, M2, M4, M5, M6
```

Después ejecutar M3 y M7. Solo si M6 muestra un efecto consistente, agregar M8
y M9.

### Protocolo

- mismas semillas emparejadas;
- mismo tamaño de población;
- mismo número de epochs;
- mismo presupuesto de evaluaciones;
- monitor y política reiniciados por epoch;
- configuración congelada antes del benchmark completo;
- evaluación final sobre los 90 casos del batch actual;
- óptimos de `mcdp_optimos.csv` usados solo para calcular gap.

Los rescates añaden evaluaciones de objetivo. Para comparar M1/M7 con versiones
sin rescate se debe limitar por evaluaciones o, al menos, reportar el costo
adicional; comparar solo por iteraciones favorece a las variantes que generan
más candidatos.

### Métricas

- mejor costo;
- gap porcentual al óptimo;
- media, mediana y desviación entre seeds;
- tiempo y evaluaciones;
- entradas y duración del modo exploratorio;
- cantidad de rescates e individuos rescatados;
- tasa de recuperación después de entrar en exploración;
- mejora producida dentro de las siguientes `k` iteraciones;
- factibilidad de todas las soluciones;
- diversidad de asignaciones de la población.

Para la comparación estadística se recomienda Wilcoxon emparejado por seed y
corrección Holm--Bonferroni. Deben persistirse los resultados individuales, no
solamente sus promedios.

## 15. Fases de implementación

### Fase 1 — Congelar y describir la versión actual

1. Nombrar el tratamiento actual `M1_legacy_fire_full`.
2. Crear una prueba de reproducción por seed.
3. Separar contadores de alarma, adaptación y rescate.

### Fase 2 — Extraer el controlador

1. Crear `mcdp_dtw_control.py`.
2. Definir perfiles, decisiones, políticas y actuadores.
3. Corregir los objetivos de relajación.
4. Mantener una ruta legacy sin cambios funcionales.

### Fase 3 — Crear las primeras variantes

1. Implementar M0 y M2.
2. Implementar histéresis.
3. Implementar M4, M5 y M6 mediante `adapt_target`.
4. Añadir tests de aislamiento de parámetros.

### Fase 4 — Diversificación y reglas adicionales

1. Implementar M3 y M7.
2. Ejecutar rescate solo al entrar en exploración.
3. Implementar M8 de cuatro estados.
4. Implementar M9 proporcional solo si los resultados lo justifican.

### Fase 5 — Benchmark

1. Ampliar runner y batch con tratamiento explícito.
2. Registrar evaluaciones, parámetros y transiciones.
3. Ejecutar piloto.
4. Congelar configuración.
5. Ejecutar benchmark completo y análisis estadístico.

## 16. Recomendación final

La comparación más informativa no es crear muchas combinaciones arbitrarias,
sino conservar el método actual y agregar tres versiones de histéresis:

```text
M4_hysteresis_woa
M5_hysteresis_abc
M6_hysteresis_woa_abc
```

Comparadas contra:

```text
M0_no_dtw
M1_legacy_fire_full
M2_fire_params_only
```

Con esas seis versiones se podrá determinar si DTW realmente aporta al MCDP,
si el rescate actual es necesario y si la respuesta al estancamiento debe
aplicarse sobre WOA, ABC o ambos. Solo después conviene probar cuatro estados o
adaptación proporcional.
