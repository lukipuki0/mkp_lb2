# Metaheurísticas colaborativas y control DTW

Repositorio de investigación para optimización **MKP**, funciones continuas **CEC2022** y dimensionamiento energético **HRES2–H2**. Incluye pipelines que alternan metaheurísticas mediante detección de estancamiento DTW/DDTW y un estudio independiente de adaptación de parámetros.

## Empezar por MKP: selección adaptativa

La implementación `adaptive_mh_selection/` usa el mismo orquestador por **iteraciones** de `hybrid_mkp/`, con su inicialización, alternancia y cambio de MH al terminar cada época. Solo cambia la selección de la siguiente MH: uniforme o según su mejora reciente por iteración.

```bash
# Desde la raíz: validar sin ejecutar ni crear resultados.
python -B -m adaptive_mh_selection.run --dry-run

# Primera familia OR-Library, primera instancia (índice 0), una corrida adaptativa.
python -B -m adaptive_mh_selection.run --instance instancias/mknapcb1.txt --index 0 --seed 42 --runs 1 --selectors adaptive --modes dtw --iterations 1000

# Comparación funcional uniforme/adaptativa con ambos detectores.
python -B -m adaptive_mh_selection.run --index 0 --seed 42 --runs 1 --selectors uniform adaptive --modes dtw ddtw --iterations 1000
```

Los resultados se guardan en una carpeta nueva dentro de `resultados/adaptive_mh_selection/`, con reportes y gráficos originales de `hybrid_mkp`: convergencia, delta DTW, Gantt y fitness instantáneo. No se usan presupuestos por evaluaciones. Se retiró `instantaneous_convergence.png` y se agregó una figura temporal de cuatro paneles con las MH y métricas registradas, sin imputar datos faltantes; ver la guía del paquete. Igual que en el pipeline original, la última época puede sobrepasar las iteraciones solicitadas; el reporte registra el total real. Las corridas anteriores por evaluaciones quedan como históricas, no se renombran como iteraciones.

Configuración, métricas, limitaciones y análisis: **[adaptive_mh_selection/README.md](adaptive_mh_selection/README.md)**.

## Campaña de 27 casos para análisis estadístico posterior

El ejecutor agrupado prepara las nueve familias, índices 0/1/2, con 31 semillas por caso: **837 corridas** para `adaptive_dtw`. Las agrupa por **100, 250 y 500 ítems**, conserva las 31 mediciones numéricas y guarda solución, historiales y gráficos completos **solo del mejor run** de cada caso.

```bash
# Validar el protocolo completo sin ejecutar ni crear resultados.
python -B -m adaptive_mh_selection.campaign --dry-run

# Ejecución manual completa; revisar costo y protocolo antes de lanzarla.
python -B -m adaptive_mh_selection.campaign --iterations 1000 --runs 31 --seed 42
```

Estructura, política de retención, estado y límites: [guía de la campaña](adaptive_mh_selection/README.md#campaña-agrupada-27-casos-y-31-corridas-por-caso). Los fallos y sus semillas se registran; no se sustituyen por ceros ni se esconden para completar artificialmente 31 observaciones. El modo individual `run` sigue disponible para depuración, pero la campaña best-only se ejecuta con `campaign`.

## Estructura y experimentos

| Carpeta / archivo | Responsabilidad |
|---|---|
| `adaptive_mh_selection/` | Selección adaptativa y uniforme sobre el pipeline MKP original por iteraciones. |
| `hybrid_mkp/` | Pipeline MKP colaborativo original, solvers, reparación y gráficos. |
| `mkp_core/`, `lb2/`, `mh/` | Modelo MKP, binarización LB2 y metaheurísticas discretas. |
| `dtw_stagnation.py` | Monitor compartido de estancamiento DTW/DDTW. |
| `continuous_benchmark/` | Funciones oficiales CEC2022, MH continuas, pipeline y estadísticas. |
| `HRES2-H2/` | Modelo energético y pipeline con MH poblacionales y de trayectoria. |
| `DTW_optimization-main/` | Estudio independiente de adaptación de parámetros con PSO, GA, GWO y DE binarios. |
| `sensitivity/` | Protocolos y campañas de sensibilidad de los pipelines rotativos. |
| `instancias/` | Instancias MKP OR-Library locales. |
| `resultados/`, `resultados finales/` | Salidas locales y resultados históricos; no mezclar campañas de protocolos diferentes. |
| `paper/`, `informes/`, `planes/` | Artículo, explicaciones, propuestas y material histórico. |

**Dos mecanismos distintos:** el pipeline colaborativo cambia de algoritmo; el estudio `DTW_optimization-main/` cambia parámetros de un mismo algoritmo. `ALGORITMO_CENTRAL.md` es documentación de otra etapa y no debe tomarse como inventario exhaustivo del código actual.

### Algoritmos individuales conservados

- MKP colaborativo: pool poblacional GA, PSO, GWO, WOA, EHO y ACO; pool de trayectoria SA, TS, ILS y VNS.
- Solvers continuos disponibles: GA, PSO, GWO, WOA, EHO, ACO y ABC.
- El pool continuo por defecto conserva WOA, PSO, EHO, GWO y ACO. HRES2 configura su propio pool y alternancia.

## Retiro de WOA–ABC

Se retiró la integración activa del híbrido **WOA–ABC** que dependía de un paquete ausente y bloqueaba los imports de CEC/HRES. También se retiraron el puente continuo, los dos lanzadores específicos y las pruebas exclusivas de esa función inexistente; se agregaron regresiones de los solvers independientes vigentes. Los nombres cooperativos ya no son opciones válidas del dispatcher continuo.

**WOA y ABC individuales se conservan.** Los resultados y documentos históricos no se borraron para hacer pasar tests.

El material MCDP en [planes/futuro_mcdp/](planes/futuro_mcdp/) es un archivo inactivo, no un paquete instalable ni un punto de entrada del flujo actual. Sus ejemplos cooperativos son históricos y requieren una integración explícita antes de ejecutarse.

## Dependencias y comprobaciones

Usar Python 3.11+ y un entorno configurado con NumPy, SciPy, Matplotlib y Requests. Las instancias y los datos oficiales de CEC deben existir localmente. No se descargan sustitutos de esos datos para simular una corrida oficial.

```bash
# Suite de los pipelines, análisis y herramientas existentes.
python -B -m unittest discover -s tests -v

# Suite de selección adaptativa MKP.
python -B -m unittest discover -s adaptive_mh_selection/tests -v
```

Los tests nuevos verifican imports sin el paquete cooperativo, ejecución de los siete solvers continuos, rechazo de los nombres retirados y preservación de WOA/ABC independientes. No se usan skips para esconder la dependencia eliminada.

Para CEC, HRES2, sensibilidad y adaptación de parámetros, consultar las guías de cada subproyecto y revisar sus presupuestos antes de lanzar los scripts HPC. Las comparaciones científicas requieren múltiples semillas, protocolos congelados y controles de costo equivalentes.
