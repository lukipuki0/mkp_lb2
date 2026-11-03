# Sensibilidad OAT del monitor DTW

Copia exacta del paquete de la rama `OAT` (commits `01c5a0e` y `a291028`)
para llevarlo a otro proyecto. El código está en `sensibilidad/`, al lado
de este archivo. En `dtw_discreto` el runner y el config todavía no leen
estas variables: hay que hacer los dos enganches de abajo en el proyecto
destino antes de lanzar la campaña.

## Qué hace

Un barrido one-factor-at-a-time del monitor. Hay una configuración base
con los valores del paper y, para cada parámetro, tres valores distintos
del de la base. El resto queda fijo. Salen 19 configuraciones.

`min_slope` queda en 2.0 en todas. El barrido cubre solo lo que citó el
revisor R1.4: ventana, banda de Sakoe-Chiba, percentiles, `plateau_max`
(`π_max`) y `patience` (`τ_pat`).

| Parámetro | Valores (el del paper lleva `*`) | Variable de entorno |
|---|---|---|
| `window` | 50, 100, 200*, 400 | `MKP_WINDOW` |
| `band` | 1, 2*, 4, 8 | `MKP_BAND` |
| `p_low` | 20, 30, 40*, 50 | `MKP_P_LOW` |
| `p_high` | 50, 60*, 70, 80 | `MKP_P_HIGH` |
| `plateau_max` | 3, 5*, 8, 12 | `MKP_PLATEAU_MAX` |
| `patience` | 1, 2, 3*, 5 | `MKP_PATIENCE` |
| `min_slope` | 2.0* fijo | `MKP_MIN_SLOPE` |

La base del paper es `W=200`, banda `2`, percentiles `40/60`,
`π_max=5`, `τ_pat=3`. Cada corrida recibe las siete variables con esos
valores y después pisa una sola. El resultado no depende de los defaults
que tenga el config local.

Cada configuración corre Binary-Complex en PSO, GA, GWO y DE.
El protocolo de la rama `OAT` usa `mknapcb1` en los índices 0, 15 y 29,
con 31 épocas.

## Los tres archivos

- `grid.py` arma la lista. La primera entrada es `base` (sin overrides).
  Las demás tienen `config_id` del estilo `window=50` y un dict
  `overrides` con un solo parámetro. `python grid.py` imprime
  `OAT grid OK: 19 configurations` si la lista cierra.
- `run_sensitivity.py` escribe el manifiesto y lanza el runner del
  proyecto, una vez por configuración, instancia e índice.
- `analizar.py` lee los JSON de fitness y escribe una tabla markdown
  por parámetro, con la media por metaheurística y el salto contra la base.

## Dónde ponerlo

El paquete tiene que quedar en la raíz del proyecto destino, como
hermano del runner:

```text
<proyecto>/
  sensibilidad/
    __init__.py
    grid.py
    run_sensitivity.py
    analizar.py
  run_all_hpc.py
  mkp_common/config.py
```

`PROJECT_ROOT` es el padre de la carpeta `sensibilidad`. Desde ahí se
llama a `run_all_hpc.py` y ahí se leen
`results/sensibilidad/` y `results/binary_complex/`. Si el paquete se
queda dentro de `lucas copiar/`, el padre pasa a ser esa carpeta y el
runner no aparece.

## Enganche 1 — el config lee el entorno al importar

En el config del monitor, cada perilla sale de `os.environ` y el default
es el valor local del proyecto. Hay que leerlas en el momento del import:
el runner hijo hereda el entorno y después importa el config.

```python
_DTW_BASE = dict(
    window=int(os.environ.get("MKP_WINDOW", 100)),
    band=int(os.environ.get("MKP_BAND", 2)),
    min_slope=float(os.environ.get("MKP_MIN_SLOPE", 2.0)),
    use_ddtw=True,
    adapt_thresholds=True,
    p_low=float(os.environ.get("MKP_P_LOW", 20.0)),
    p_high=float(os.environ.get("MKP_P_HIGH", 80.0)),
)

DTW_FIRE_D2 = StagnationConfig(
    **_DTW_BASE,
    plateau_max=int(os.environ.get("MKP_PLATEAU_MAX", 5)),
    patience=int(os.environ.get("MKP_PATIENCE", 3)),
)
```

Los números de arriba son los defaults de `dtw_discreto` cuando no hay
variable. La campaña OAT los pisa siempre con la base del paper.

## Enganche 2 — el runner respeta el id y puede saltar el estadístico

`run_sensitivity.py` llama, con el cwd en la raíz del proyecto:

```text
python run_all_hpc.py
  --instancia <archivo>
  --indice <n>
  --epochs <n>
  --skip vanilla_explotacion vanilla_exploracion binary_simple binary_patient
  --no-stats
  [--cpus <n>]
```

El entorno del proceso ya trae las siete `MKP_*` y `MKP_CAMPAIGN_ID`
con un id propio de esa configuración (`<campaña>_<posición de dos
dígitos>`, por ejemplo `20260918_120000_00` para la base).

En `run_all_hpc.py` hacen falta dos cambios.

El id de la corrida sale del entorno. Si el runner inventa siempre un
timestamp, `analizar.py` no encuentra los JSON y un resume pisa otra
carpeta.

```python
run_id = os.environ.get("MKP_CAMPAIGN_ID") or datetime.now().strftime("%Y%m%d_%H%M%S")
os.environ["MKP_CAMPAIGN_ID"] = run_id
```

Y un flag para no lanzar el análisis estadístico al final de cada
configuración:

```python
p.add_argument(
    "--no-stats",
    action="store_true",
    help="Skip the statistical analysis subprocess at the end",
)

# donde hoy corre el estadístico:
if args.no_stats:
    print("\n  [SKIP] Statistical analysis skipped (--no-stats).")
elif completed >= 1:
    ...
```

Los JSON tienen que quedar en:

```text
results/binary_complex/todos/<instancia>_<indice>/comparacion_mhs_<MKP_CAMPAIGN_ID>/<MH>_<instancia>_<indice>.json
```

`<MH>` es `PSO`, `GA`, `GWO` o `DE`. El JSON lleva una lista `fitness`
(o `mejor_fitness`) con el mejor fitness de cada época. Eso ya lo escribe
`mkp_common.results.save_results` cuando lo llama `_save_strategy_results`.

## Qué editar si el otro proyecto no se llama igual

Estos nombres están fijos en el paquete. Cámbialos juntos:

- En `run_sensitivity.py`, `_build_command`: la lista `--skip` deja
  activo solo `binary_complex`. Si la estrategia destino tiene otro
  nombre, esa es la que no se salta.
- En `run_sensitivity.py`, `_config_is_complete`, y en `analizar.py`,
  `_config_metrics`: la carpeta de resultados es `results/binary_complex`.
  Tiene que ser la misma carpeta que usa el runner (`STRATEGIES[...]["folder"]`).
- En `analizar.py`, la tupla `MHS`. Tiene que coincidir con el prefijo
  de los JSON.
- En `grid.py`, `PAPER_BASE`, `GRID` y `ENV_MAP`, si el otro monitor
  usa otros parámetros u otros nombres de variable.

## Cómo se corre

Desde la raíz del proyecto destino:

```bash
python -m sensibilidad.grid
python -m sensibilidad.run_sensitivity --cpus 40 --epochs 31
python -m sensibilidad.analizar --campaign <id>
```

`<id>` es el timestamp que imprime la campaña, sin el prefijo
`campaign_`. El manifiesto queda en
`results/sensibilidad/campaign_<id>/configs.json`. El análisis escribe
ahí `tabla_oat.md` y también lo imprime.

Para retomar la misma campaña (mismos ids por configuración, así la base
sigue siendo comparable):

```bash
python -m sensibilidad.run_sensitivity --campaign <id> --cpus 40 --epochs 31
```

Sin más filtros, el resume salta las configuraciones que ya tienen los
cuatro JSON en cada índice. `--desde-config window=50` arranca en esa
configuración, inclusive. `--solo band=4` corre solo esa. Esos dos
filtros se pueden combinar con `--campaign`.

Una configuración se da por completa cuando, para cada instancia e índice
del manifiesto, existe el directorio y dentro están `PSO_*.json`,
`GA_*.json`, `GWO_*.json` y `DE_*.json`.
