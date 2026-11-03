# Documentación del Modelo WPEB Extendido y Metaheurísticas

Este directorio (`HRES2-H2`) contiene la implementación del **Modelo Extendido de Optimización de Sistemas Híbridos Energía Eólica - Fotovoltaica - Electrolizador - Batería (WPEB)** resuelto mediante metaheurísticas híbridas.

El notebook principal de este desarrollo es:
- [`WPEB_extendido_metaheuristicas.ipynb`](file:///c:/Users/Abduzcan0/Desktop/mkp_lb2/HRES2-H2/WPEB_extendido_metaheuristicas.ipynb)

---

## 📌 Contexto y Referencia

El desarrollo se basa en el modelo planteado en el paper científico:
> **Li et al. (2024)**, *Capacity optimization of a wind-photovoltaic-electrolysis-battery (WPEB) hybrid energy system for power and hydrogen generation*. International Journal of Hydrogen Energy. [DOI / ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0360319923039812)

### Resumen del Modelo Base (Paper)
El paper original formula un método de optimización de capacidad de objetivo único con múltiples restricciones para un sistema WPEB a escala de utilidad (**>10 MW**) en Damao Banner, Mongolia Interior, China. El objetivo es minimizar el **LCOE** (*Levelized Cost of Electricity*) mediante una simulación de producción de 8760 horas anuales combinada con un barrido de cuadrícula (Grid Search) y descenso de gradiente.

---

## 🚀 Extensiones del Modelo (Este Trabajo)

Este trabajo introduce cuatro modificaciones clave que transforman el espacio de búsqueda en un **problema de optimización mixto (variables continuas y discretas simultáneamente)**:

| Aspecto | Modelo Original (Paper) | Modelo Extendido (Este Trabajo) |
|---|---|---|
| **Capacidad del Electrolizador** | Variable continua en MW | Variable entera discreta: N° de módulos de 5 MW (entre 10 y 20 módulos; 50 a 100 MW) |
| **Potencia de la Batería** | Fija en 30% de la capacidad del electrolizador | Variable independiente discreta, múltiplos de 5 MW (0 a 50 MW) |
| **Duración de la Batería** | Fija en 1 hora | Variable discreta en el conjunto `{1, 2, 4}` horas |
| **Método de Optimización** | Grid Search + Descenso de gradiente | Metaheurísticas Híbridas: **GWO**, **PSO**, **DE** |

---

## ⚙️ Estructura del Notebook (`WPEB_extendido_metaheuristicas.ipynb`)

El notebook está organizado en **5 secciones principales**:

### 1. Importación de Librerías y Definición de Parámetros
- **Librerías principales**: `numpy`, `pandas`, `matplotlib`, `scipy.stats.pearsonr`, `dataclasses`, `tqdm`.
- **Diccionario `CONFIG`**: Parámetros físicos del sistema, eficiencias, coeficientes térmicos PV, curva eólica y parámetros operacionales.
  - Capacidad total de generación: `200.0 MW` ($P_{Wind} + P_{PV} = 200\text{ MW}$).
  - Restricción AGSR (Annual Grid Supply Ratio): $\le 20\%$.
  - Electrolizador: Mínimo $30\%$ de carga operativa, capacidad máxima $\le 50\%$ de la generación total ($100\text{ MW}$).
- **Diccionario `COSTS`**: CAPEX, reemplazo y O&M por tecnología expresados en CNY (yuan chino) por kW/kW-año (tomados de la Tabla 4 del paper).

### 2. Definición del Modelo Físico y Económico
- **Modelo Fotovoltaico (`pv_power_mw`)**: Calcula la temperatura de celda basada en NOCT, ajusta la eficiencia por coeficiente térmico y aplica el derating factor.
- **Modelo Eólico (`wind_turbine_power_mw` / `aggregate_wind_power_mw`)**: Implementa una curva de potencia cúbica con $v_{\text{cut-in}}=2.5\text{ m/s}$, $v_{\text{rated}}=10.5\text{ m/s}$ y $v_{\text{cut-out}}=25.0\text{ m/s}$.
- **Cálculos Económicos (`crf`, `npc_from_capacities`)**: Determina el Factor de Recuperación de Capital (CRF), el Valor Presente Neto (NPC), los costos anualizados, LCOE (CNY/kWh) y LCOH (CNY/kg).
- **Simulador WPEB Extendido (`simulate_wpeb_extended`)**: Ejecuta el despacho horario (8760 h) aplicando reglas de prioridad:
  1. La generación renovable abastece directamente el electrolizador si supera el mínimo (30%).
  2. Si no alcanza el mínimo, se descarga la batería si es viablemente posible.
  3. De no ser posible, el electrolizador se apaga en esa hora.
  4. Los excedentes cargan la batería.
  5. El remanente sobrante se vende a la red eléctrica.

### 3. Obtención de Datos Climáticos
- Descarga de series meteorológicas de la API de **NASA POWER** para Damao Banner (lat: 41.70°N, lon: 110.43°E), abarcando el período 2001–2021.
- Selección del Año TípicoMeteorológico (TMY) mediante la distribución del coeficiente de correlación de Pearson diario entre viento e irradiancia solar, o mediante selección directa del año de referencia del paper (**2008**).

### 4. Declaración de Metaheurísticas Híbridas
Dado que el espacio de búsqueda es mixto, los algoritmos operan internamente en un espacio continuo 4D:
$$x = [\text{wind\_mw}, \text{n\_el\_units\_cont}, \text{battery\_mw\_cont}, \text{duration\_index\_cont}]$$

Y cada solución se proyecta al espacio discreto mediante la función `decode_solution`.

Se implementan las siguientes tres metaheurísticas:
1. **Grey Wolf Optimizer (GWO) Híbrido (`gwo_optimize_extended`)**: Modelo basado en la jerarquía de caza de lobos ($\alpha, \beta, \delta$).
2. **Particle Swarm Optimization (PSO) Híbrido (`pso_optimize_extend`)**: Modelo de enjambre con inercia lineal decreciente ($w_{\max}=0.9 \to w_{\min}=0.4$), componentes cognitivo/social ($c_1, c_2$) y velocidad máxima por dimensión.
3. **Differential Evolution (DE) Híbrido (`de_optimize_extend`)**: Algoritmo evolutivo con esquema **DE/rand/1**, cruzamiento binomial y selección greedy.

### 5. Resolución y Comparación de Resultados
- **Parámetros de ejecución**: Población = 20 individuos, Iteraciones = 50, Semilla aleatoria = 42.
- **Indicadores evaluados**:
  - LCOE (CNY/kWh) y LCOH (CNY/kg).
  - Capacidades óptimas: Eólica (MW), Solar PV (MW), Electrolizador (MW), Batería (MW y horas de almacenamiento).
  - Indicador de ventas a la red (AGSR) y Factor de Capacidad del Electrolizador (CF).
  - Gráficos de convergencia individual y comparación cruzada de curvas de aprendizaje (**GWO vs PSO vs DE**).

---

## 📊 Resumen de Variables de Decisión

| Variable | Tipo | Rango / Valores Permitidos |
|---|---|---|
| `wind_mw` | Continua | $[0.0, 200.0]\text{ MW}$ |
| `pv_mw` | Derivada | $200.0 - \text{wind\_mw}$ |
| `n_el_units` | Entera | $\{10, 11, \dots, 20\}$ unidades (5 MW c/u $\to 50$ a $100\text{ MW}$) |
| `battery_mw` | Discreta | $\{0, 5, 10, \dots, 50\}\text{ MW}$ |
| `battery_duration_h` | Discreta | $\{1.0, 2.0, 4.0\}\text{ horas}$ |

---

## 🛠️ Requisitos e Instalación

Para ejecutar el notebook `WPEB_extendido_metaheuristicas.ipynb`:

```bash
pip install pandas numpy matplotlib scipy requests tqdm
```

O dentro de un entorno Jupyter/Colab:

```python
!pip -q install requests pandas numpy matplotlib scipy tqdm
```

---

## Benchmark reproducible (`benchmark_hres2.py`)

No hay un wrapper `reproduce_campaign.py`. Los defaults del monitor nominal son W=40, band=4, patience=8, min_slope=0.1, plateau=15, adaptativo y percentiles 30/70. El protocolo de paper es 31 runs y 1000 iteraciones nominales, en modos `dtw` y `ddtw`, con nueve baselines ejecutados una sola vez: PSO, GWO, WOA, EHO, ACO, ILS, SA, TS y VNS. ABC permanece en el portafolio híbrido y no es un baseline del paper.

Para usar el botón **Run** del editor, abrí `HRES2-H2/benchmark_hres2.py` y ejecutalo sin argumentos. Inicia la campaña completa: DTW y DDTW, 31 runs por método, 1000 iteraciones nominales y los nueve baselines una sola vez. Crea automáticamente una carpeta nueva bajo `HRES2-H2/resultados/`; no hace falta escribir comandos. Es una ejecución completa, no una prueba pequeña.

```bash
python HRES2-H2/benchmark_hres2.py --help
python HRES2-H2/benchmark_hres2.py --nonpaper --output HRES2-H2/resultados/nonpaper_smoke_t4
python HRES2-H2/benchmark_hres2.py --modes dtw,ddtw --output HRES2-H2/resultados/paper_campaign_fresh
```

`--help` no lanza simulaciones. `--nonpaper` es un smoke explícito de 1 run y 1 iteración nominal, modo `dtw` y baseline `PSO`, salvo que `--modes`, `--baselines`, `--runs` o `--iterations` lo acoten. Esos overrides se honran. La salida estadística de un smoke n=1 no es significativa (`not meaningful`). No es un framework genérico. Un destino existente se rechaza con `refusing to overwrite existing output destination` y no se sobrescribe. Si no se pasa `--output`, se elige un directorio nuevo bajo `HRES2-H2/resultados/`.

`runs.jsonl` se vuelca y se hace flush después de cada run. Guarda run, método, modo y semilla; el objetivo de optimización aparte del LCOE/LCOH/AGSR físico y de `feasible`; el vector final y las capacidades/H2 decodificadas; iteraciones nominales, iteraciones reales y overshoot; y el tiempo cuando está disponible. Una métrica física no finita queda en `null`. Una penalización del objetivo no se reporta como LCOE físico. Repetir la semilla no demuestra por sí sola condiciones iniciales pareadas: cada solver consume el RNG de forma distinta y el clima se construye antes de sembrar. `manifest.json` guarda monitor efectivo, configuración, modelo, procedencia del clima, runtime y evidencia de fuente. `completion.json` solo aparece si la campaña termina.

La inferencia promovida es Mann-Whitney bilateral de muestras independientes con Holm. Solo se calcula si la referencia y el comparador tienen todos los registros esperados con LCOE físico finito y factible. Si falta un run o hay registros no factibles, se conservan todos los registros, los conteos y la media descriptiva de los factibles, pero esa comparación se omite con motivo explícito: no hay p ni W/T/L, no se imputa y no se promueve superioridad del subconjunto factible. Si la referencia está incompleta, se omiten todas las comparaciones. Wilcoxon y Friedman no se usan: no hay pareo justificado, tampoco entre DTW y DDTW. Con `dtw,ddtw`, cada híbrido se compara solo con los baselines de paper elegidos; el otro híbrido no entra en esa familia Holm. Los baselines se optimizan una vez y `runs.jsonl` los guarda una vez. Los resúmenes de cada modo van a `analisis_dtw/` y `analisis_ddtw/` para no sobrescribirse. En un solo modo se mantienen las claves de retorno anteriores y `stat_results` apunta a ese híbrido; `stat_results_by_hybrid` mapea cada variante. En el resumen, `best` es el mínimo y va separado de la mediana. Un resultado DDTW se etiqueta `Hybrid DDTW`, no `Hybrid DTW`.

El presupuesto nominal no cambia la semántica de las épocas: `max_epoch_iters` no se fuerza, así que una iteración nominal puede registrar un overshoot real. El smoke `--nonpaper` de 1 iteración nominal sigue pudiendo ejecutar una época híbrida completa.
