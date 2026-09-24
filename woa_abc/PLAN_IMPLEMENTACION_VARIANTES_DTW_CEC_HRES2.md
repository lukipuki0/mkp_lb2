# Plan maestro: variantes WOA--ABC controladas por DTW para CEC2022 y HRES2

## Estado de implementación

Implementado:

- las siete variantes M0, M1, M2, M4, M5, M6 y M8;
- controlador de parámetros `woa_abc/adaptive.py`;
- motor CEC separado y ejecutor CEC;
- motor HRES2 separado y ejecutor HRES2;
- persistencia de configuración, resultados por seed, convergencia, DTW y
  parámetros bajo `woa_abc/resultados/`.
- una única traza continua por variante; la interfaz activa no usa epochs ni
  crea carpetas `epoch_*`.

Pendiente para una etapa posterior: análisis estadístico inferencial y gráficos
comparativos a partir de los CSV guardados. No forma parte de la implementación
actual.

## 1. Objetivo inmediato

Implementar exactamente siete variantes de control DTW, inspiradas en el
trabajo previo de MCDP pero aplicadas ahora a problemas continuos:

- las 12 funciones CEC2022;
- el modelo energético HRES2-H2/WPEB.

La implementación tendrá un motor separado por dominio. CEC usará su propio
WOA--ABC para funciones CEC y HRES2 tendrá su propio WOA--ABC adaptado al
modelo energético. Solo se compartirán el sensor DTW, las políticas y los
perfiles de parámetros. El código, los ejecutores de CEC/HRES2 y todos sus
resultados se concentran en `woa_abc/`.

MCDP queda fuera de esta etapa. Su plan se conserva en:

```text
planes/futuro_mcdp/IMPLEMENTACION_VARIANTES_DTW_MCDP.md
```

El material MCDP queda archivado y fuera de los imports activos durante esta
fase.

## 2. Decisión de arquitectura

Se implementará una sola familia de MH: WOA--ABC adaptativa. WOA realiza la
fase exploratoria y ABC ejecuta sus fases empleada, observadora y scout sobre
una población compartida. CEC2022 y HRES2 tendrán motores independientes para
no mezclar contratos, límites ni evaluaciones de los problemas.

El pipeline rotativo existente de HRES2 (PSO, GWO, WOA, EHO, ACO, ABC, ILS, SA,
TS y VNS) queda fuera del alcance. No se integrarán MH de trayectoria ni se
compararán turnos de algoritmos en este plan.

```text
CEC2022Function ──> motor WOA--ABC CEC ──> runner CEC ──> woa_abc/resultados/cec

HRES2Function ────> motor WOA--ABC HRES2 ─> runner HRES2 ─> woa_abc/resultados/hres2
                         ▲                  ▲
                         └── DTW/políticas/perfiles compartidos ──┘
```

Esto evita comparar, dentro de una misma tabla, una modificación de parámetros
de WOA--ABC con un cambio de arquitectura completo.

### Regla de uso de DTW

DTW será únicamente un sensor para seleccionar parámetros de WOA y/o ABC. No
podrá:

- abortar turnos o iteraciones;
- cambiar de algoritmo o introducir MH de trayectoria;
- resembrar, rescatar o reemplazar individuos por decisión del monitor;
- alterar la función objetivo, los límites o el presupuesto de evaluaciones.

La población solo cambia por las operaciones normales de WOA--ABC y por su
aceptación codiciosa.

## 3. Estado actual que se reutiliza

| Componente existente | Uso en esta etapa |
|---|---|
| `dtw_stagnation.py` | Sensor DTW/DDTW común. |
| `woa_abc/cooperativo_cec_dtw.py` | Motor WOA--ABC exclusivo para CEC2022. |
| `woa_abc/cooperativo_hres2_dtw.py` | Motor WOA--ABC exclusivo para HRES2; se implementa por separado. |
| `continuous_benchmark/mh/woa_abc.py` | Fachada de compatibilidad; no contiene la lógica del algoritmo. |
| `continuous_benchmark/funciones_cec2022.py` | 12 funciones oficiales y sus datos de desplazamiento/rotación. |
| `HRES2-H2/wpeb_model.py` | `HRES2Function`, simulación WPEB, decodificación y métricas energéticas; no contiene el motor MH. |
| `continuous_benchmark/orchestrator.py` | Código existente; no es parte del runner de variantes. |
| `HRES2-H2/orchestrator.py` | Pipeline histórico fuera del alcance de esta implementación. |
| `continuous_benchmark/benchmark_continuo.py` | Formato de resultados CEC que debe ampliarse para tratamientos. |
| `HRES2-H2/benchmark_hres2.py` | Formato de resultados HRES2 que debe ampliarse para tratamientos. |

## 4. Límites separados por dominio

### CEC2022

`woa_abc/cooperativo_cec_dtw.py` es exclusivamente el motor CEC y puede
mantener el contrato escalar actual:

```python
lb = float(func.lb)
ub = float(func.ub)
```

No se agregará lógica HRES2 a este archivo. Las 12 funciones CEC comparten el
mismo intervalo escalar por dimensión y el motor CEC seguirá usando ese
contrato.

### HRES2

`HRES2Function` tiene límites por variable:

```text
lb_vector = [0, 10, 0, 0]
ub_vector = [200, 20, 50, 2]
```

Las variables representan, después de decodificar:

```text
x[0] -> potencia eólica
x[1] -> número de unidades del electrolizador
x[2] -> potencia de batería
x[3] -> índice de duración de batería
```

El motor separado `woa_abc/cooperativo_hres2_dtw.py` trabajará con vectores:

```python
lower = np.asarray(func.lb_vector, dtype=float)
upper = np.asarray(func.ub_vector, dtype=float)
```

Todos los `clip`, la inicialización y el scout del motor HRES2 recibirán esos
vectores. La decodificación y las validaciones físicas permanecerán dentro del
flujo HRES2.

CEC y HRES2 no compartirán una clase genérica de problema ni una función de
límite. Compartirán únicamente el controlador DTW y los perfiles de parámetros.

## 5. Variantes que se implementarán (solo estas siete)

Las variantes mantienen la misma población y las mismas fases WOA--ABC. Se
identifican con nombres comunes para CEC y HRES2. No se implementarán variantes
adicionales en esta etapa.

| ID | Variante | Política DTW | Parámetros modificados | Operación DTW |
|---|---|---|---|---|
| `M0_no_dtw` | baseline continuo | ninguno | ninguno | no hay monitor |
| `M1_legacy_fire_full` | regla legacy adaptada | `fire` compuesto | todos los parámetros legacy de WOA + ABC | solo parámetros |
| `M2_fire_params_only` | pulso aislado | `fire` compuesto | amplitud WOA + intensidad ABC | solo parámetros |
| `M4_hysteresis_woa` | histéresis WOA | `delta` con entrada/salida | solo parámetros WOA | solo parámetros |
| `M5_hysteresis_abc` | histéresis ABC | `delta` con entrada/salida | solo parámetros ABC | solo parámetros |
| `M6_hysteresis_woa_abc` | histéresis coordinada | `delta` con entrada/salida | parámetros WOA + ABC | solo parámetros |
| `M8_four_state_woa_abc` | cuatro intensidades | cuatro estados de `delta` | parámetros WOA + ABC | solo parámetros |

### M0 — `no_dtw`

No se crea un monitor. WOA y ABC usan los parámetros base y su calendario
temporal normal. Es el baseline que permite medir el costo y el beneficio de
añadir control DTW.

### M1 — `legacy_fire_full`

Reproduce la regla de parámetros del código CEC previo:

```text
fire=True
  -> aumentar exploración WOA
  -> aumentar phi de ABC
  -> reducir guía al mejor
  -> reducir límite scout
  -> relajar parámetros con decay
```

Debe conservarse como control de compatibilidad, pero todos sus efectos deben
ser valores de parámetros calculados desde el perfil base. No puede rescatar,
resembrar ni reemplazar individuos por decisión de DTW.

### M2 — `fire_params_only`

Usa exactamente la misma alarma y cooldown que M1, pero limita la intervención
a dos parámetros: amplitud de WOA y escala de `phi` de ABC. No modifica guía,
límite scout ni ningún estado de la población. Responde si el cambio mínimo de
parámetros es suficiente.

### M4 — `hysteresis_woa`

DTW decide el modo, pero únicamente modifica la amplitud de WOA. ABC usa siempre
sus parámetros base. Sirve para medir si la recuperación depende de los
movimientos globales de las ballenas.

### M5 — `hysteresis_abc`

DTW decide el modo, pero solo modifica ABC:

- `abc_phi_scale`;
- `abc_guide_strength`;
- `abc_limit`.

WOA conserva su calendario y parámetros base.

### M6 — `hysteresis_woa_abc`

Es la versión principal. La política es:

```text
exploit -> explore cuando delta >= theta_delta
explore -> exploit cuando delta <= 0
```

El modo se mantiene en la banda intermedia. Al entrar en `explore`, se aplican
perfiles de WOA y ABC; al salir, se vuelve al perfil de explotación.

### M8 — `four_state_woa_abc`

En lugar de dos perfiles usa cuatro:

| Condición | Perfil |
|---|---|
| `delta > theta_delta` | explorar mucho |
| `0 <= delta <= theta_delta` | explorar poco |
| `-theta_delta <= delta < 0` | explotar poco |
| `delta < -theta_delta` | explotar mucho |

Es una extensión de M6, no una implementación alternativa del motor.

## 6. Perfiles continuos compartidos

El controlador debe trabajar con perfiles, no con multiplicadores acumulados.

| Parámetro | Base | Exploit | Explore |
|---|---:|---:|---:|
| `woa_a_scale` | 1.00 | 0.75 | 1.50 |
| `woa_a_floor` | 0.00 | 0.00 | 1.25 |
| `woa_a_cap` | 2.00 | 0.99 | 2.50 |
| `abc_phi_scale` | 1.00 | 0.60 | 1.50 |
| `abc_guide_strength` | 0.20 | 0.35 | 0.05 |
| `abc_limit_factor` | 1.00 | 1.25 | 0.50 |
| `step_scale` | 1.00 | 0.85 | 1.00 |
| `momentum_scale` | 1.00 | 1.00 | 1.00 |

Los valores son puntos iniciales. `b_spiral`, `exploration_probability`, el
tamaño de población y el calendario base de `step_size` permanecen fijos hasta
que se complete la comparación principal.

El coeficiente WOA debe calcularse con un piso y un techo:

```python
a_base = 2.0 - 2.0 * iteration / max(1, iterations - 1)
a = min(profile.woa_a_cap, max(
    profile.woa_a_floor,
    a_base * profile.woa_a_scale,
))
```

Así una detección tardía de estancamiento todavía puede aumentar la exploración.

El límite ABC debe derivarse siempre del límite base:

```python
effective_limit = max(
    min_abc_limit,
    round(base_limit * profile.abc_limit_factor),
)
```

No se debe multiplicar repetidamente el límite ya adaptado porque produciría
deriva entre activaciones.

### Ajustes experimentales activos

Para reducir alarmas producidas por ruido numérico y evitar perturbaciones
repetidas, la configuración manual activa usa `window=20`, `band=2`,
`min_slope=2.0`, `plateau_max=4`, `patience=2` e `improvement_tol=1e-6`.
La tolerancia convierte mejoras relativas menores que ese umbral en no-mejoras.
El controlador acepta como máximo un `fire` por episodio de estancamiento; la
alarma vuelve a aceptarse únicamente después de que la condición desaparece.

Como punto de partida de la búsqueda se usan `woa_exploration=1.20`,
`exploration_probability=0.65`, `step_initial=2.0`, `step_final=0.10` y un
límite ABC base `pop_size * n_dim // 4` (respetando `min_abc_limit`). Estos
valores son configurables por separado para CEC y HRES2 en
`config_experimentos.py` y quedan registrados en `configuracion.json`.

## 7. Controlador común

El controlador compartido queda dentro del módulo activo WOA--ABC:

```text
woa_abc/adaptive.py   # catálogo, perfiles, políticas, decisiones y contadores
```

El paquete no define una clase genérica de problema. Cada motor recibe el tipo
de función de su dominio y conserva sus propias reglas de evaluación y límites.

### Interfaz del controlador

```python
class ControlMode(str, Enum):
    BASE = "base"
    EXPLOIT = "exploit"
    EXPLORE = "explore"
    EXPLORE_LOW = "explore_low"
    EXPLORE_HIGH = "explore_high"
    EXPLOIT_LOW = "exploit_low"
    EXPLOIT_HIGH = "exploit_high"

@dataclass(frozen=True)
class ControlDecision:
    mode: ControlMode
    intensity: float
    reason: str

class DTWController(Protocol):
    def reset(self) -> None: ...
    def observe_minimization(self, best_cost: float) -> dict[str, Any]: ...
    def decide(self, status: dict[str, Any]) -> ControlDecision: ...
```

`observe_minimization()` centraliza el signo y llama internamente a:

```python
monitor.update(-best_cost)
```

CEC y HRES2 no deben repetir esa conversión en sus propios runners.

## 8. Motores WOA--ABC separados

### Motor CEC

`woa_abc/cooperativo_cec_dtw.py` permanece dedicado a CEC2022. Mantiene su
contrato `func`, `n_dim`, `lb`, `ub` y sus resultados específicos de benchmark.
`continuous_benchmark/mh/woa_abc.py` solo lo reexporta por compatibilidad.

### Motor HRES2

`woa_abc/cooperativo_hres2_dtw.py` es el motor independiente. Recibe
`HRES2Function`, usará `lb_vector`/`ub_vector`, llamará a la evaluación WPEB y
conservará la decodificación y las métricas energéticas de HRES2.

### Ciclo compartido conceptualmente

Ambos motores siguen el mismo orden, pero con implementaciones separadas:

```text
crear función del dominio y sus límites propios
crear población y evaluar costos del dominio
crear controlador nuevo por ejecución
modo_actual = exploit/base

para cada iteración:
    calcular perfil efectivo del dominio
    ejecutar WOA y aceptación codiciosa
    ejecutar ABC empleada
    ejecutar ABC observadora
    ejecutar scout
    actualizar mejor global
    registrar historia y evaluaciones
    observar -best_cost con DTW
    decidir el perfil de parámetros para la siguiente iteración
```

Todas las variantes adaptativas, incluida `legacy_fire_full`, usan una semántica
común y reproducible: la decisión tomada al observar `t` se aplica a `t+1`.

### Invariantes del motor

- Cada motor recorta con los límites definidos por su propio dominio.
- Los candidatos rechazados no reemplazan su individuo.
- El mejor global histórico nunca aumenta en costo.
- Un scout no puede borrar el mejor global externo.
- El estado de la política se reinicia al inicio de cada ejecución.
- CEC y HRES2 usan el mismo conteo de evaluaciones.
- Momentum y `step_size` se calculan con el mismo calendario en todas las
  variantes principales.

## 9. CEC2022

### Cobertura

Ejecutar las 12 funciones oficiales en las dimensiones actualmente soportadas:

```text
F1 ... F12
D = 10 y 20
```

Los archivos oficiales de desplazamiento, rotación y permutación se mantienen
en `continuous_benchmark/input_data/`. No se generan datos sustitutos.

### Runner nuevo

```text
woa_abc/run_cec_variants.py
```

Responsabilidades:

1. cargar la función CEC desde `continuous_benchmark/funciones_cec2022.py`;
2. elegir una variante de `M0_no_dtw`, `M1_legacy_fire_full`,
   `M2_fire_params_only`, `M4_hysteresis_woa`, `M5_hysteresis_abc`,
   `M6_hysteresis_woa_abc` o `M8_four_state_woa_abc`;
3. ejecutar las seeds emparejadas;
4. guardar configuración, historial y eventos;
5. generar resumen por función y variante dentro de un experimento de una sola dimensión.

Uso previsto:

```bash
python -m woa_abc.run_cec_variants \
  --functions all \
  --dimension 10 \
  --variants all \
  --runs 1 \
  --iterations 1000 \
  --window 20 \
  --band 2 \
  --min-slope 2.0 \
  --plateau-max 4 \
  --patience 2 \
  --ddtw
```

`D=20` se ejecuta por separado con `--dimension 20`. Las 31 corridas se
habilitarán en la etapa posterior de análisis estadístico.

### Métricas CEC

- mejor valor final;
- media, mediana y desviación por seed;
- error respecto al bias oficial;
- evaluaciones para alcanzar umbrales de error;
- tiempo;
- entradas y salidas de exploración;
- cantidad de actualizaciones de parámetros;
- área bajo la convergencia;
- distancias `D1`, `D2` y `delta`.

El benchmark debe conservar los vectores por seed. Los promedios no sustituyen
los datos necesarios para Wilcoxon y Holm--Bonferroni.

## 10. HRES2-H2

### Contrato propio del motor HRES2

`cooperativo_hres2_dtw.py` recibe directamente `HRES2Function` y usa:

```text
n_dim = 4
lower = [0, 10, 0, 0]
upper = [200, 20, 50, 2]
evaluate = HRES2Function._evaluate
```

El motor trabaja en continuo y `decode_solution()` conserva la proyección a:

- potencia eólica;
- unidades enteras de electrolizador;
- potencia discreta de batería;
- duración `{1, 2, 4}` horas.

No se debe rediseñar el modelo físico. La evaluación seguirá devolviendo LCOE o
la penalización de infeasibilidad que ya utiliza HRES2.

### Baseline HRES2

`M0_no_dtw` será la referencia: el motor WOA--ABC propio de HRES2 sin
controlador DTW. Las seis variantes con DTW (M1, M2, M4, M5, M6 y M8) se
comparan contra M0. El pipeline rotativo histórico no se ejecuta ni se incluye
en esta ablación.

### Runner nuevo

```text
woa_abc/run_hres2_variants.py
```

Responsabilidades:

1. construir `HRES2Function` desde `HRES2-H2/wpeb_model.py` con el mismo TMY para todas las seeds;
2. ejecutar las siete variantes: M0, M1, M2, M4, M5, M6 y M8;
3. registrar LCOE y la solución de capacidades;
4. decodificar la mejor solución con `decode_solution()`;
5. guardar LCOH, AGSR, producción de H2, factor de capacidad y factibilidad;
6. generar reportes por variante.

Uso previsto:

```bash
python -m woa_abc.run_hres2_variants \
  --variants all \
  --runs 31 \
  --iterations 1000 \
  --window 20 \
  --band 2 \
  --min-slope 2.0 \
  --plateau-max 4 \
  --patience 2 \
  --ddtw
```

### Métricas HRES2

- LCOE final;
- LCOH;
- AGSR;
- producción anual de hidrógeno;
- factor de capacidad del electrolizador;
- factibilidad;
- capacidades decodificadas;
- tiempo y evaluaciones;
- número de transiciones y actualizaciones de parámetros;
- estabilidad entre seeds.

`HRES2Function.optimum=0.30` debe tratarse como referencia declarada, no como
óptimo demostrado, hasta validar que corresponde al modelo/configuración
actual. La comparación principal debe usar LCOE, factibilidad y métricas
físicas, no solo un gap respecto a ese valor.

## 11. Estructura final propuesta

```text
mkp_lb2/
├── dtw_stagnation.py                 # sensor DTW/DDTW existente
├── continuous_benchmark/
│   ├── funciones_cec2022.py
│   ├── mh/
│   │   └── woa_abc.py                # fachada compatible
│   └── input_data/
├── HRES2-H2/
│   ├── wpeb_model.py
│   └── orchestrator.py               # histórico, fuera del runner nuevo
├── woa_abc/
│   ├── adaptive.py                   # siete variantes y perfiles DTW
│   ├── cooperativo_cec_dtw.py        # motor WOA--ABC exclusivo CEC
│   ├── cooperativo_hres2_dtw.py      # motor WOA--ABC HRES2 separado
│   ├── run_cec_variants.py           # ejecutor CEC2022
│   ├── run_hres2_variants.py         # ejecutor HRES2
│   ├── result_io.py                  # persistencia común
│   ├── PLAN_IMPLEMENTACION_VARIANTES_DTW_CEC_HRES2.md
│   ├── README.md
│   └── resultados/
│       ├── cec/
│       │   └── run_TIMESTAMP/
│       └── hres2/
│           └── run_TIMESTAMP/
├── planes/
│   └── futuro_mcdp/
│       ├── IMPLEMENTACION_VARIANTES_DTW_MCDP.md
│       ├── codigo/
│       │   ├── mcdp_core/
│       │   └── woa_abc/
│       └── resultados/
└── tests/
    └── test_woa_abc_variants.py
```

`woa_abc/` es el módulo canónico desde el inicio. `run_cec_variants.py` importa
las funciones CEC desde `continuous_benchmark/` y `run_hres2_variants.py` importa
el modelo HRES2 desde `HRES2-H2/`; cada uno llama únicamente a su motor y ambos
guardan configuración, historial y resultados bajo `woa_abc/resultados/`.

## 12. Formato de resultados

Cada ejecución debe guardar, dentro de `woa_abc/resultados/cec/` o
`woa_abc/resultados/hres2/` según el dominio:

```text
configuracion.json
configuracion.txt
results_variants.csv
results_variants.json
resumen_global.md
```

Para cada variante, función, dimensión y seed:

```text
domain
problem
dimension
variant
seed
best_value
optimum_reference
error
gap_pct
iterations
objective_evaluations
time_seconds
dtw_alarm_count
mode_transition_count
parameter_update_count
```

Cada run incluye actualmente:

```text
historial_convergencia.csv
historial_dtw.csv
historial_parametros.csv
eventos_control.json
```

Los gráficos comparativos se producirán junto con el análisis estadístico
posterior, sin volver a ejecutar los motores.

En HRES2 se añaden:

```text
solucion_decodificada.json
metricas_hres2.csv
```

## 13. Protocolo experimental

### Fase piloto

CEC:

- F1, F4, F7 y F10;
- dimensiones 10 y 20;
- 5 seeds;
- 300 iteraciones.

HRES2:

- 5 seeds;
- 300 iteraciones;
- TMY sintético o TMY fijado, pero igual para todas las variantes.

El piloto verifica límites, tiempos, número de evaluaciones y frecuencia de
activaciones. No se deben elegir perfiles usando el conjunto completo después
de mirar sus resultados finales.

### Benchmark final

CEC:

- F1--F12;
- D=10,20;
- 31 seeds emparejadas;
- mismo presupuesto por variante.

HRES2:

- 31 seeds emparejadas;
- misma instancia meteorológica;
- mismo presupuesto de evaluaciones;
- M0, M1, M2, M4, M5, M6 y M8.

### Análisis estadístico posterior (no implementado en esta entrega)

Los CSV conservan cada seed para que posteriormente se implemente:

- Wilcoxon pareado por seed;
- corrección Holm--Bonferroni;
- Friedman para comparar todas las variantes en conjunto;
- tamaño del efecto, no solo `p-value`;
- análisis separado CEC y HRES2.

En esa fase se creará `woa_abc/analyze_statistics.py`, que leerá los
`results_variants.csv` ya generados y escribirá tablas de rangos, valores p,
correcciones, tamaños de efecto y gráficos comparativos. Los motores no se
volverán a ejecutar.

No se deben mezclar escalas ni tratar las 24 combinaciones CEC como si fueran
una sola función sin incluir función y dimensión en el análisis.

## 14. Pruebas de aceptación

### Control DTW

- ninguna política cambia antes de `ready=True`;
- histéresis entra con `delta >= theta_delta`;
- histéresis sale con `delta <= 0`;
- cuatro estados respetan sus fronteras;
- el estado se reinicia por ejecución;
- los parámetros usados en `t+1` corresponden a la decisión de `t`;
- DTW no puede abortar iteraciones ni modificar la población;
- M1 conserva la regla legacy de parámetros con aplicación reproducible en `t+1`.

### Motor continuo

- CEC usa directamente sus límites escalares;
- HRES2 usa `lb_vector` y `ub_vector`;
- ninguna coordenada sale de sus límites;
- el costo global es monotónico no creciente;
- un candidato rechazado no reemplaza la población;
- DTW no crea, rescata ni resembra individuos;
- el mismo seed reproduce resultado y eventos;
- se contabilizan evaluaciones de objetivo;
- CEC y HRES2 no duplican la lógica de control.

### Integración

- `continuous_benchmark.mh.woa_abc` conserva sus imports públicos como fachada;
- los ejecutores CEC/HRES2 y sus resultados viven bajo `woa_abc/`;
- HRES2 usa únicamente el motor WOA--ABC continuo y su baseline M0;
- los runners generan carpetas independientes por variante;
- las configuraciones quedan guardadas junto a los resultados;
- todos los smoke tests pasan antes del benchmark completo.

## 15. Orden de implementación

### Fase 0 — Organización

1. Mantener todo el código, datos, resultados y documentación MCDP bajo
   `planes/futuro_mcdp/`.
2. Crear el plan maestro actual para CEC2022 y HRES2.
3. No importar ni ejecutar el código MCDP desde los runners activos.
4. Ejecutar smoke tests del CEC y HRES2 existentes.

### Fase 1 — Contratos separados por dominio

1. Documentar el contrato escalar que ya usa el motor CEC.
2. Definir el contrato propio del motor HRES2 con `lb_vector` y `ub_vector`.
3. Añadir pruebas de clipping CEC y HRES2 por separado.
4. Añadir pruebas de decodificación y factibilidad física de HRES2.

### Fase 2 — Controlador DTW compartido

1. Crear `woa_abc/adaptive.py`.
2. Implementar perfiles base, exploit y explore.
3. Implementar las cuatro políticas necesarias: `no_dtw`, `fire`,
   `hysteresis` y `four_state`.
4. Implementar contadores y eventos.
5. Añadir prueba de fronteras y reset.

### Fase 3 — Motor WOA--ABC CEC

1. Refactorizar `cooperativo_cec_dtw.py` sin mezclar HRES2.
2. Agregar selección de política y actuador DTW.
3. Registrar parámetros y evaluaciones.
4. Ejecutar M0 y M1 sobre una función pequeña.
5. Crear `woa_abc/run_cec_variants.py`.
6. Escribir las salidas exclusivamente en `woa_abc/resultados/`.

### Fase 4 — Todas las variantes CEC

1. Añadir M2, M4, M5, M6 y M8.
2. Ejecutar piloto CEC.
3. Validar convergencia, errores y eventos.
4. Ejecutar benchmark final CEC.

### Fase 5 — Motor WOA--ABC HRES2 separado

1. Crear `woa_abc/cooperativo_hres2_dtw.py` sin modificar el motor CEC.
2. Integrar `HRES2Function`, `lb_vector`, `ub_vector` y `decode_solution()`.
3. Crear `woa_abc/run_hres2_variants.py`.
4. Ejecutar M0 sin DTW.
5. Ejecutar M1, M2, M4, M5, M6 y M8 con el motor HRES2.
6. Validar límites por dimensión y métricas físicas.
7. Ejecutar benchmark final HRES2.

### Fase 6 — Validación separada

1. Verificar CEC contra su propio motor y sus propias métricas.
2. Verificar HRES2 contra su propio motor y sus métricas físicas.
3. Confirmar que ambos usan solo el controlador DTW compartido.
4. Mantener cualquier pipeline rotativo histórico fuera de los resultados.

## 16. Recomendación operativa

La primera entrega de código debe incluir la infraestructura, el motor CEC y M0/M1:

```text
woa_abc/adaptive.py
motor WOA--ABC CEC
M0_no_dtw
M1_legacy_fire_full
```

Cuando esa base reproduzca el comportamiento actual, se incorporan M2, M4, M5,
M6 y M8 al motor CEC. Después se crea y valida el motor HRES2 por separado,
manteniendo MCDP aislado para una etapa posterior.
