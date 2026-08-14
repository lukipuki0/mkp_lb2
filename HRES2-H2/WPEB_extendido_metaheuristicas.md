# Documentación del Notebook: WPEB_extendido_metaheuristicas.ipynb

Este archivo resume el contenido, modelo físico-económico, formulación y algoritmos implementados en el notebook [`WPEB_extendido_metaheuristicas.ipynb`](file:///c:/Users/Abduzcan0/Desktop/mkp_lb2/HRES2-H2/WPEB_extendido_metaheuristicas.ipynb).

---

## 📋 Resumen Ejecutivo

- **Nombre del Proyecto**: Optimización de Capacidad en Sistema Híbrido Viento-PV-Electrolizador-Batería (WPEB) mediante Metaheurísticas Híbridas.
- **Ubicación de Estudio**: Damao Banner, Baotou, Mongolia Interior, China (Lat 41.70°N, Lon 110.43°E).
- **Paper de Referencia**: *Li et al. (2024)*, International Journal of Hydrogen Energy.
- **Función Objetivo**: Minimizar el **LCOE** (Levelized Cost of Electricity) y calcular el **LCOH** (Levelized Cost of Hydrogen).
- **Capacidad Total del Sistema**: 200 MW ($P_{\text{Wind}} + P_{\text{PV}} = 200\text{ MW}$).

---

## 💡 Novedades del Modelo Extendido

| Parámetro / Componente | Paper Base (Li et al., 2024) | Modelo Extendido (Este Notebook) |
|---|---|---|
| **Electrolizador** | Variable continua | Variable entera discreta ($N \times 5\text{ MW}$, $N \in [10, 20]$) |
| **Batería (Potencia)** | Fija ($30\%$ de capacidad de electrolizador) | Variable discreta en pasos de $5\text{ MW}$ (0 a $50\text{ MW}$) |
| **Batería (Duración)** | Fija (1 hora) | Variable discreta $\{1, 2, 4\}\text{ horas}$ |
| **Algoritmo de Optimización** | Grid Search + Descenso de gradiente | Metaheurísticas híbridas: **GWO**, **PSO**, **DE** |

---

## 🧱 Estructura y Secciones del Notebook

### 1. Importación de Librerías y Configuración (`CONFIG` y `COSTS`)
- Definición de constantes físicas (curva eólica, eficiencias, NOCT).
- Diccionario de costos (`COSTS`) con CAPEX, reemplazo y O&M por tecnología (CNY/kW).
- Restricciones: $AGSR \le 20\%$, Electrolizador $\le 100\text{ MW}$.

### 2. Definición del Modelo Físico-Económico
- **`pv_power_mw`**: Generación PV considerando corrección por temperatura de celda (NOCT) y factor de degradación.
- **`wind_turbine_power_mw`**: Generación eólica con curva cúbica ($v_{\text{cut-in}}=2.5$, $v_{\text{rated}}=10.5$, $v_{\text{cut-out}}=25.0$ m/s).
- **`npc_from_capacities`**: Valor Presente Neto (NPC) y Costo Anualizado.
- **`simulate_wpeb_extended`**: Simulador horario (8760 horas) con despacho priorizado (electrolizador $\to$ batería $\to$ red).

### 3. Datos Climáticos de NASA POWER
- Descarga de datos horarios de irradiancia (`ALLSKY_SFC_SW_DWN`), viento a 50 m (`WS50M`) y temperatura (`T2M`) para 2001-2021.
- Selección del Año TípicoMeteorológico (TMY) mediante coeficientes de correlación de Pearson o fijado en **2008**.

### 4. Metaheurísticas Híbridas
Decodificación de vectores continuos a espacio mixto discreto-continuo (`decode_solution`):
- **GWO (Grey Wolf Optimizer)**: `gwo_optimize_extended` (20 lobos, 50 iteraciones).
- **PSO (Particle Swarm Optimization)**: `pso_optimize_extend` (20 partículas, 50 iteraciones, $w_{\max}=0.9 \to w_{\min}=0.4$).
- **DE (Differential Evolution)**: `de_optimize_extend` (20 candidatos, 50 iteraciones, mutación $F=0.8$, $CR=0.9$).

### 5. Comparación y Resultados
- Generación de gráficos de convergencia individuales (`GWO`, `PSO`, `DE`) y gráfico comparativo consolidado.
- DataFrame de resultados finales listando LCOE, LCOH, capacidades óptimas, AGSR, CF y factibilidad.
