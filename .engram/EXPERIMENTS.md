# Experimentos y resultados recuperados

Son corridas individuales, no evidencia estadística de superioridad. Instancia oficial mknapcb1, índice 0, semilla 60, selector adaptativo y 1000 iteraciones solicitadas salvo indicación contraria. El exceso de iteraciones corresponde al cierre del último bloque nativo.

| Corrida | Ventana / modo | Fitness | Iteraciones reales | Brecha vs 24381 |
|---|---|---:|---:|---:|
| Vista inicial adaptive | 70 / DTW | 24115 | 1041 | No registrada entonces; referencia agregada después |
| adaptive | 20 / DTW | 24280 | 1034 | 0,414257 % |
| adaptive | 20 / DDTW | 24280 | 1034 | 0,414257 % |
| adaptive_exploratory | 20 / DDTW | 24280 | 1034 | 0,414257 % |
| exploratory, tres MH por pool | 20 / DDTW | 24280 | 1034 | 0,414257 % |

## Rutas locales

Base: `resultados/adaptive_mh_selection/`.

- Inicial: `preview_mknapcb1_inst_00_20261023_084346/best_run_01/` (ejecutada con campaign).
- DTW W20: `preview_mknapcb1_inst_00_seed60_window20/adaptive_dtw/best_run_01/`.
- DDTW W20: `preview_mknapcb1_inst_00_seed60_window20_ddtw/adaptive_ddtw/best_run_01/`.
- Exploratory: `preview_mknapcb1_inst_00_seed60_window20_ddtw_exploratory/adaptive_exploratory_ddtw/best_run_01/`.
- Tres por pool: `preview_mknapcb1_inst_00_seed60_window20_ddtw_exploratory_3perpool/adaptive_exploratory_ddtw/best_run_01/`.

Cada ganador tiene `summary.json`, `resumen_pipeline_best_run.txt`, diagnóstico temporal y probabilidades. En las corridas de `run`, buscar `selection_trace.csv` en el hermano `run_01/`, no asumir que está bajo `best_run_01/`. Archivos históricos no garantizados en otros clones: no se subieron los resultados.

## Comparación exploratory

- Adaptive DDTW: VNS elegido 10 veces, probabilidad final registrada 85 %; tiempo de solver registrado 89,407 s.
- Exploratory completo: VNS elegido 10 veces, probabilidad final 82,3163 %; TS apareció una vez en lugar de ILS; tiempo 626,312 s.
- Exploratory reducido: población GA/PSO/ACO; trayectoria SA/ILS/VNS; tiempo 162,574 s.

| MH reducida | Veces elegida | Última probabilidad registrada dentro del pool |
|---|---:|---:|
| GA | 5 | 32,0715 % |
| PSO | 4 | 32,0715 % |
| ACO | 3 | 35,8570 % |
| SA | 1 | 8,17256 % |
| ILS | 1 | 8,17256 % |
| VNS | 10 | 83,6549 % |

Estas probabilidades se registran antes de elegir, no son necesariamente el estado posterior a la última actualización. Mayor movimiento de las curvas no produjo mejora de fitness en este piloto. Las diferencias de tiempo no están diagnosticadas; no equivalen a una comparación controlada de rendimiento.

## Redibujar sin optimizar (rama de selección adaptativa)

```python
from pathlib import Path
from adaptive_mh_selection.plotting import load_selection_trace, plot_selection_probabilities

records = load_selection_trace(Path("ruta/al/run_01/selection_trace.csv"))
plot_selection_probabilities(records, Path("ruta/a/una/vista_nueva"))
```

Genera PNG/PDF según la implementación publicada. Verificar la rama antes de usar este API. No sobrescribir históricos sin autorización.
