# Time-Series-Driven Coordination of Collaborative Metaheuristics via Dynamic Time Warping

Código de investigación asociado al manuscrito **“Time-Series-Driven Coordination of Collaborative Metaheuristics via Dynamic Time Warping”**. El framework utiliza el historial del mejor fitness como una serie temporal: compara ventanas mediante **Dynamic Time Warping (DTW)** y **Derivative Dynamic Time Warping (DDTW)** para detectar estancamiento persistente y coordinar cambios entre metaheurísticas, conservando la mejor solución encontrada.

El método analiza trayectorias para tomar decisiones durante la búsqueda; **no realiza predicción de series temporales**. Combina el análisis temporal con contadores de falta de mejora y filtros de persistencia, sin una política de control aprendida.

> Este README no anuncia la aceptación o publicación del artículo. La referencia bibliográfica definitiva y el DOI se incorporarán cuando estén disponibles. Las extensiones experimentales del repositorio no deben confundirse con los experimentos del manuscrito.

## Dominios de evaluación

| Dominio | Problema | Implementación |
|---|---|---|
| MKP | Problema de la mochila multidimensional, con instancias OR-Library | [`hybrid_mkp/`](hybrid_mkp/), [`mkp_core/`](mkp_core/) |
| CEC2022 | Optimización continua, funciones F1–F12 con dimensión 10 en el paper | [`continuous_benchmark/`](continuous_benchmark/) |
| HRES2–H2 | Dimensionamiento de un sistema eólico, fotovoltaico, electrolizador y batería, con simulación de 8.760 horas | [`HRES2-H2/`](HRES2-H2/) |

El monitor compartido se encuentra en [`dtw_stagnation.py`](dtw_stagnation.py). El pipeline colaborativo alterna algoritmos poblacionales y de trayectoria; no es lo mismo que adaptar los parámetros de un único algoritmo.

## Organización del repositorio

| Ruta | Contenido |
|---|---|
| `hybrid_mkp/` | Orquestación MKP, reparación, solvers y gráficos. |
| `mkp_core/`, `lb2/`, `mh/` | Modelo MKP, binarización y metaheurísticas discretas. |
| `continuous_benchmark/` | Funciones CEC2022, solvers continuos y análisis. |
| `HRES2-H2/` | Modelo energético, benchmark y registro de resultados. |
| `instancias/` | Instancias MKP locales. |
| `sensitivity/` | Protocolos de sensibilidad del monitor. |
| `adaptive_mh_selection/` | Extensión de selección adaptativa de la siguiente metaheurística; no es el protocolo base del paper. |
| `DTW_optimization-main/` | Estudio independiente de adaptación de parámetros. |
| `tests/` | Pruebas de los pipelines y herramientas. |
| `resultados/`, `resultados finales/` | Salidas locales y campañas históricas con protocolos distintos. |
| `paper/`, `informes/`, `planes/` | Borradores, documentación de trabajo y material histórico. |

El manuscrito activo se mantiene en `asoc/`, un **repositorio Git independiente sincronizado con Overleaf**. No debe asumirse que un clon de este repositorio incluya ese manuscrito ni todas las salidas experimentales locales. Los borradores de `paper/` no sustituyen automáticamente la versión enviada a la revista.

## Entorno y datos

Usar **Python 3.11+** en un entorno virtual. Las bibliotecas utilizadas incluyen NumPy, SciPy, Matplotlib, Requests y, según el subproyecto, Pandas y tqdm. Estas indicaciones son una base de instalación, no un entorno de dependencias congeladas.

```bash
python -m venv .venv
# Activar el entorno antes de instalar:
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install numpy scipy matplotlib requests pandas tqdm
```

Antes de ejecutar:

- **MKP:** comprobar la disponibilidad de las instancias en `instancias/` y seleccionar explícitamente familia e índice.
- **CEC2022:** disponer de los datos oficiales de desplazamiento, rotación y permutación en `continuous_benchmark/input_data/`. La [guía CEC2022](continuous_benchmark/README_CEC2022.md) identifica su procedencia. No se generan sustitutos de los datos oficiales.
- **HRES2–H2:** revisar la [guía del benchmark](HRES2-H2/README.md#benchmark-reproducible-benchmark_hres2py) y la procedencia climática registrada en cada campaña. La evaluación del artículo corresponde a su escenario anual; no demuestra robustez frente a todos los años o emplazamientos.

## Reproducibilidad de los experimentos

Las campañas usan múltiples semillas y pueden requerir tiempos de ejecución considerables. **No ejecutar los lanzadores completos como prueba de instalación.** Los valores por defecto actuales no garantizan reproducir campañas históricas: deben contrastarse con el protocolo, la configuración y la procedencia de cada resultado.

### MKP y CEC2022

El artículo evalúa configuraciones DTW/DDTW con 31 corridas. MKP incluye presupuestos nominales de 1.000 y 3.000 iteraciones; CEC2022 utiliza 1.000 iteraciones y dimensión 10. Algunas campañas históricas emplearon ventanas y paciencia distintas de la configuración nominal: no deben agruparse ni renombrarse como si fueran el mismo experimento.

Consultar la implementación en `hybrid_mkp/`, la [guía CEC2022](continuous_benchmark/README_CEC2022.md) y los protocolos del manuscrito antes de configurar una campaña. Los scripts HPC de la raíz son lanzadores de trabajo: revisar sus rutas, recursos y parámetros antes de usarlos en otro entorno. Este README no presenta un comando único de reproducción integral de todas las tablas.

### HRES2–H2

Desde la raíz del repositorio:

```bash
# Consultar opciones sin ejecutar simulaciones.
python HRES2-H2/benchmark_hres2.py --help

# Prueba pequeña, explícitamente ajena al protocolo del paper.
python HRES2-H2/benchmark_hres2.py --nonpaper

# Campaña completa: ejecución costosa, no una prueba de instalación.
python HRES2-H2/benchmark_hres2.py --modes dtw,ddtw
```

La campaña completa ejecuta **31 corridas por método**: DTW, DDTW y nueve baselines compartidos (PSO, GWO, WOA, EHO, ACO, ILS, SA, TS y VNS), para **341 registros**. Los baselines se ejecutan una sola vez. ABC permanece disponible en el portafolio híbrido, pero no integra esta comparación de nueve baselines.

La configuración nominal es: ventana 40, banda 4, paciencia 8, pendiente mínima 0,1, contador de estancamiento 15, percentiles adaptativos 30/70 y 1.000 iteraciones. Las épocas pueden sobrepasar ese presupuesto: se guardan las iteraciones reales, sin equiparar iteraciones nominales con igual número de evaluaciones o tiempo.

**Ejecutar `benchmark_hres2.py` sin argumentos, incluido el botón Run del editor, inicia la campaña completa.** Para una prueba pequeña se requiere `--nonpaper`; sus resultados no respaldan inferencia estadística.

Cada ejecución crea un directorio nuevo en `HRES2-H2/resultados/`. También se puede indicar `--output RUTA_NUEVA`; un destino existente se rechaza para evitar sobrescrituras.

| Archivo o directorio | Evidencia registrada |
|---|---|
| `runs.jsonl` | Método, semilla, objetivo, métricas físicas, factibilidad, capacidades, iteraciones y tiempo cuando está disponible. |
| `manifest.json` | Configuración efectiva, procedencia climática, entorno y evidencia de fuente. |
| `completion.json` | Finalización de la campaña; no reemplaza la inspección de los registros. |
| `analisis_dtw/`, `analisis_ddtw/` | Resúmenes y comparaciones separados por variante. |

La campaña utilizada para la actualización del manuscrito se identifica localmente como `HRES2-H2/resultados/run_dtw_ddtw_20261027_143517_660355/`. Su presencia en una copia de trabajo **no garantiza su distribución en un clon**; comprobar los archivos disponibles antes de intentar reconstruir tablas o figuras.

### Interpretación estadística

- HRES2 utiliza Mann–Whitney bilateral y corrección de Holm, con una familia separada de nueve comparaciones por variante híbrida.
- Compartir semillas no demuestra condiciones iniciales pareadas. No se presenta una comparación pareada DTW–DDTW en HRES2.
- La inferencia se omite cuando faltan registros esperados o existen resultados no factibles o métricas físicas no finitas; no se imputan resultados para completar muestras.
- La desviación estándar reportada es muestral. El mínimo, la media y la mediana son medidas distintas.
- En la campaña HRES2 del manuscrito todos los métodos fueron factibles y varios compartieron el mínimo. No se afirma optimalidad global ni una ventaja exclusiva de factibilidad de los híbridos.
- No trasladar el protocolo estadístico de HRES2 a MKP o CEC2022 sin revisar el diseño correspondiente.

## Comprobaciones

Desde la raíz:

```bash
# Pruebas de los pipelines y herramientas.
python -B -m unittest discover -s tests -v

# Comprobación de las funciones oficiales CEC2022.
python continuous_benchmark/test_cec2022.py

# Pruebas de la extensión adaptativa MKP.
python -B -m unittest discover -s adaptive_mh_selection/tests -v
```

Las pruebas funcionales no sustituyen las campañas científicas ni garantizan reproducir exactamente sus resultados en otro entorno.

## Extensiones y material histórico

La [selección adaptativa MKP](adaptive_mh_selection/README.md) compara selección uniforme y selección según mejora reciente, conservando el pipeline por iteraciones. Dispone de una validación que no ejecuta optimización ni crea resultados:

```bash
python -B -m adaptive_mh_selection.run --dry-run
```

Su campaña agrupada de 27 casos y 837 corridas es un estudio separado; consultar su guía antes de ejecutarla. `DTW_optimization-main/` estudia adaptación de parámetros de un mismo algoritmo, no la coordinación del paper.

La integración cooperativa WOA–ABC fue retirada; **WOA y ABC individuales se conservan**. Los documentos y resultados históricos pueden mencionar esa integración, pero no constituyen puntos de entrada vigentes. El material de `planes/futuro_mcdp/` es igualmente inactivo.

El [README anterior archivado](docs/README_origin_9a11921.md) y `ALGORITMO_CENTRAL.md` describen etapas anteriores. Para interpretar un resultado, priorizar siempre la configuración y los registros de su propia campaña.

## Referencia del artículo

Título del manuscrito: **Time-Series-Driven Coordination of Collaborative Metaheuristics via Dynamic Time Warping**.

La cita definitiva, el DOI y los enlaces a los materiales publicados se añadirán cuando exista una referencia verificable. No usar una cita provisional inventada ni interpretar la disponibilidad del código como una declaración de licencia de uso.
