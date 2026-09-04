# WOA--ABC híbrido con DTW para MCDP

Esta carpeta contiene la mezcla activa para resolver el **Machine Cell Design
Problem (MCDP)**. La implementación adapta la comunicación del enfoque
MSE-ABC a WOA + ABC: existe una sola población y las fases del algoritmo
trabajan sobre ella en secuencia.

## Idea principal

No se ejecutan dos algoritmos completos por separado ni se mantienen dos
poblaciones aisladas. El flujo es:

```text
Población única
      │
      ├── WOA: exploración global
      │       └── actualiza el mejor global
      │
      ├── ABC empleada: búsqueda vecinal
      │       └── usa las mejoras de WOA inmediatamente
      │
      ├── ABC observadora: selección probabilística
      │
      ├── ABC scout: reinicio de una fuente estancada
      │
      └── DTW: detecta estancamiento y adapta parámetros
```

Esta es la diferencia fundamental respecto a una arquitectura cooperativa de
dos islas. En una arquitectura de islas habría una población WOA y otra ABC
con intercambio de élites. Aquí hay una población común, como en MSE-ABC, y la
comunicación se produce por el estado actualizado de esa población y por el
mejor global.

El algoritmo conserva WOA como operador exploratorio. El documento MSE-ABC
utiliza SFOA como operador exploratorio; por eso la estructura de comunicación
es equivalente, pero la ecuación exploratoria no es idéntica a la del artículo.

## Representación del MCDP

Una instancia contiene una matriz máquina--pieza. Una solución asigna cada
máquina a una celda:

```text
máquina:  1  2  3  4  5  6
celda:    0  0  1  2  1  2
```

Las celdas se numeran desde `0` hasta `K-1`. Cada celda tiene una capacidad
máxima. El costo es la cantidad de elementos excepcionales y se minimiza.

El algoritmo mantiene para cada individuo:

- una posición latente continua, necesaria para aplicar WOA y ABC;
- una asignación discreta máquina→celda;
- su costo;
- un contador de intentos sin mejora para la fase scout.

La posición latente se redondea antes de evaluar. Si una celda está llena, se
asigna la máquina a la celda disponible más cercana. Así, los candidatos de
WOA, ABC, scouts y momentum siempre se reparan antes de evaluarse.

No se utiliza LB2, porque cada máquina puede pertenecer a una de varias celdas
y no solamente a una variable binaria.

## Ciclo de una iteración

Para cada iteración `t` se ejecutan estas etapas:

```text
1. Calcular momentum a partir de los mejores globales consecutivos.
2. DTW observa -mejor_costo.
3. Si hay estancamiento, adapta parámetros y rescata individuos.
4. Ejecutar la fase WOA sobre todos los individuos.
5. Actualizar el mejor global con los cambios de WOA.
6. Ejecutar la fase ABC empleada sobre la misma población.
7. Actualizar el mejor global.
8. Ejecutar la fase ABC observadora mediante selección probabilística.
9. Ejecutar la fase scout sobre la fuente más estancada.
10. Actualizar el mejor global, historiales y parámetros.
```

La fase ABC comienza con la población que WOA acaba de modificar. No recibe
una copia retrasada ni espera a una iteración posterior. Esta actualización
inmediata es la comunicación entre las dos MH.

### Ejemplo de comunicación inmediata

Supongamos que al inicio de una iteración el mejor costo es `40`.

```text
Inicio:       mejor global = 40
Después WOA:  mejor global = 37
Después ABC:  mejor global = 35
```

ABC utiliza el líder de costo `37` durante su fase empleada y puede terminar
en `35`. La siguiente iteración comienza con ese mejor global y con los
individuos modificados por ambas fases.

## Fase WOA

WOA recorre la población común usando:

- encircling alrededor del mejor global;
- exploración usando individuos aleatorios de la población;
- movimiento espiral;
- un factor de exploración que puede aumentar cuando DTW detecta una meseta.

Además, se suma el término de momentum y un paso que disminuye
exponencialmente:

```text
step(t) = step_initial · (step_final / step_initial)^(t / MaxIt)
```

La probabilidad `exploration_probability` controla la elección de la rama
exploratoria de WOA.

## Fases ABC

### Abejas empleadas

Cada individuo selecciona otro individuo como vecino y modifica una coordenada
con una diferencia ponderada por `phi`. También recibe orientación del mejor
global compartido y del momentum.

La aceptación es codiciosa: el candidato reemplaza al actual solamente si su
costo es menor o igual.

### Abejas observadoras

Se calcula una calidad para cada fuente de alimento. Las soluciones de menor
costo tienen mayor probabilidad de ser seleccionadas. Se generan nuevos
vecinos sobre la misma población y se aplica nuevamente la aceptación
codiciosa.

### Abejas scout

Cada individuo acumula intentos fallidos. Si la fuente más estancada alcanza
`limit`, se reemplaza por una solución aleatoria reparada. El mejor global se
mantiene fuera de peligro y nunca se elimina durante este reinicio.

## Momentum y paso adaptativo

El momentum usa la diferencia entre los mejores latentes de iteraciones
consecutivas:

```text
momentum(t) = best_latent(t-1) - best_latent(t-2)
```

El vector se escala mediante `momentum_factor` y se incorpora a WOA y ABC. Su
propósito es conservar una dirección útil de búsqueda y reducir movimientos
erráticos.

El paso exponencial comienza grande para favorecer exploración y termina pequeño
para permitir ajustes finos. Esto reproduce la transición progresiva de
exploración a explotación de la metodología MSE-ABC.

## DTW como controlador

MCDP es de minimización, pero el monitor DTW está diseñado para observar una
curva creciente. Por eso se le entrega `-mejor_costo`.

DTW compara la ventana reciente contra:

- una rampa de progreso;
- una línea constante de estancamiento.

Si detecta una meseta durante la cantidad de iteraciones y paciencia
configuradas, dispara una adaptación que:

- aumenta la exploración de WOA;
- aumenta el movimiento diferencial de ABC;
- reduce temporalmente el límite scout;
- aumenta las reasignaciones aleatorias;
- rescata una fracción de los peores individuos.

Después de la perturbación, los parámetros regresan gradualmente a sus valores
base. Se puede usar DTW estándar o DDTW, que compara las derivadas de las
series.

## Ejecución

Desde la raíz del repositorio:

```bash
python -m mezclas_mh.woa_abc.run_cooperative_mcdp --iterations 300
```

También se puede ejecutar el archivo directamente, incluso usando una ruta
absoluta en Windows:

```powershell
python .\mezclas_mh\woa_abc\run_cooperative_mcdp.py --iterations 300
```

El ejecutor localiza automáticamente la raíz del repositorio para encontrar
`dtw_stagnation.py` y `mcdp_core/instances/instancias.txt`.

Ejemplo con la configuración DTW indicada para el proyecto:

```bash
python -m mezclas_mh.woa_abc.run_cooperative_mcdp --file mcdp_core/instances/instancias.txt --instance 1 --cells 3 --capacity 6 --iterations 500 --epochs 3 --pop-size 30 --window 75 --band 0 --min-slope 0.1 --plateau-max 15 --patience 25 --ddtw --adapt --p-low 30 --p-high 70 --momentum-factor 0.2 --step-initial 1.0 --step-final 0.05 --seed 42
```

Parámetros relevantes:

| Opción | Función |
|---|---|
| `--pop-size` | Tamaño de la única población compartida. |
| `--iterations` | Iteraciones por epoch. |
| `--epochs` | Ejecuciones independientes. |
| `--exploration-probability` | Probabilidad de exploración WOA. |
| `--step-initial` / `--step-final` | Paso inicial y final de la reducción exponencial. |
| `--momentum-factor` | Intensidad del momentum. |
| `--limit` | Límite de intentos antes del scout. |
| `--window` | `STAG_WINDOW`. |
| `--band` | `STAG_BAND`; `0` calcula la banda automáticamente. |
| `--min-slope` | `STAG_MIN_SLOPE`. |
| `--plateau-max` | `STAG_PLATEAU_MAX`. |
| `--patience` | `STAG_PATIENCE`. |
| `--ddtw` / `--no-ddtw` | `STAG_USE_DDTW`. DDTW está activo por defecto. |
| `--adapt` / `--no-adapt` | `STAG_ADAPT`, umbrales adaptativos. |
| `--p-low` / `--p-high` | `STAG_P_LOW` y `STAG_P_HIGH`. |

El ejecutor informa el costo, elementos excepcionales, factibilidad, cantidad
de handoffs WOA→ABC y activaciones de DTW.

## Uso desde Python

```python
from dtw_stagnation import StagnationConfig
from mcdp_core.data import load_mcdp_instances
from mezclas_mh.woa_abc import (
    CooperativeMCDPParams,
    ejecutar_mcdp_cooperativo,
)

instances = load_mcdp_instances(
    "mcdp_core/instances/instancias.txt",
    max_cells=3,
    max_machines_per_cell=6,
)

params = CooperativeMCDPParams(
    pop_size=30,
    iterations=300,
    exploration_probability=0.50,
    step_initial=1.0,
    step_final=0.05,
    momentum_factor=0.20,
    stag_cfg=StagnationConfig(
        window=75,
        band=0,
        min_slope=0.1,
        plateau_max=15,
        patience=25,
        use_ddtw=True,
        adapt_thresholds=True,
        p_low=30.0,
        p_high=70.0,
    ),
    seed=42,
)

result = ejecutar_mcdp_cooperativo(instances[0], params, verbose=True)
print(result.mejor_costo_global)
print(result.mejor_sol_global)
print(result.evaluacion_global.feasible)
```

Cada epoch guarda `historial_woa` y `historial_abc`, pero representan el mejor
costo observado después de cada fase sobre la población común; no son dos
poblaciones independientes. También se registran `eventos_cooperacion`,
`eventos_adaptacion`, `dtw_info_hist` y `parametros_historial`.

## Archivos

- `cooperativo_mcdp_dtw.py`: implementación del híbrido.
- `run_cooperative_mcdp.py`: ejecutor de línea de comandos.
- `__init__.py`: interfaz pública del paquete.
