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
