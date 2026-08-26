# Hoja de Datos Ejecutiva: Resultados Empíricos HRES2-H2 y Validación Estadística Non-Paramétrica

**Proyecto:** Framework Híbrido Metaheurístico por Rotación DTW  
**Caso de Estudio:** Dimensionado Óptimo del Sistema Híbrido HRES2-H2 (Eólica + Solar PV + Electrolizador de Hidrógeno + Baterías)  
**Presupuesto Experimental:** 31 ejecuciones independientes Monte Carlo ($N=31$), 1,000 iteraciones por run, 8,760 horas de simulación anual.

---

## 1. Resumen de Desempeño del Framework Propuesto (DTW)

- **LCOE Medio (Costo Nivelado de Energía):** `0.267160 CNY/kWh`
- **Mejor LCOE Encontrado:** `0.267159 CNY/kWh`
- **Desviación Estándar ($\sigma$):** `0.000001 CNY/kWh` (Ultra-estabilidad en las 31 ejecuciones).
- **LCOH (Costo Nivelado del Hidrógeno):** `17.6314 CNY/kg`
- **Excedente a Red (AGSR):** `20.00%` (Límite máximo permitido de integración a red).
- **Tasa de Factibilidad:** `100.0%` (Frente al 80.6% – 96.8% de las metaheurísticas individuales).
- **Producción Anual de Hidrógeno Verde:** `8,862,879.4 kg/año`

---

## 2. Configuración Física Óptima del Sistema HRES2-H2

- **Parque Eólico ($P_{WT}$):** `174.47 MW`
- **Parque Solar Fotovoltaico ($P_{PV}$):** `25.53 MW`
- **Electrolizador PEM ($P_{el}$):** `70.0 MW` (14 módulos independientes de 5 MW cada uno).
- **Banco de Baterías BESS ($P_{bat}$):** `50.0 MW` (autonomía de `4.0 h` = `200.0 MWh` de almacenamiento).

---

## 3. Tablas Comparativas de Desempeño

### Tabla A: Métricas Técnico-Económicas y Desempeño ($N=31$ runs)

| Algoritmo | Mean LCOE (CNY/kWh) | Best LCOE (CNY/kWh) | Std. Dev. ($\sigma$) | LCOH (CNY/kg) | AGSR (%) | Feasibility Rate (%) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Proposed (Hybrid DTW)** | **0.267160** | **0.267159** | **0.000001** | **17.6314** | **20.00%** | **100.0%** |
| Grey Wolf Optimizer (GWO) | 0.268066 | 0.267160 | 0.002815 | 17.6800 | 19.42% | 90.3% |
| Variable Neighborhood Search (VNS) | 0.267806 | 0.267159 | 0.002465 | 17.6700 | 19.95% | 93.5% |
| Tabu Search (TS) | 0.267188 | 0.267159 | 0.000038 | 17.6334 | 19.99% | 96.8% |
| Ant Colony Optimization (ACO) | 0.273288 | 0.267159 | 0.007584 | 18.0400 | 18.95% | 93.5% |
| Iterated Local Search (ILS) | 0.267184 | 0.267171 | 0.000023 | 17.6331 | 19.98% | 96.8% |
| Particle Swarm Optimization (PSO) | 0.277289 | 0.274579 | 0.010939 | 18.2900 | 19.85% | 83.8% |
| Whale Optimization Algorithm (WOA) | 0.277227 | 0.276524 | 0.009567 | 18.3100 | 19.70% | 87.1% |
| Elephant Herding Optimization (EHO) | 0.277728 | 0.276524 | 0.007382 | 18.3400 | 19.92% | 80.6% |
| Simulated Annealing (SA) | 0.286431 | 0.281339 | 0.013206 | 18.9100 | 19.35% | 87.1% |

### Tabla B: Dimensionado Físico de Componentes por Algoritmo

| Algoritmo | Potencia Eólica (MW) | Solar PV (MW) | Electrolizador (MW) | Batería (MW / 4h) | Producción $H_2$ (kg/año) |
|---|:---:|:---:|:---:|:---:|:---:|
| **Proposed (Hybrid DTW)** | **174.47** | **25.53** | **70.0** | **50.0** | **8,862,879.4** |
| Grey Wolf Optimizer (GWO) | 176.50 | 28.00 | 75.0 | 55.0 | 8,820,410.0 |
| Variable Neighborhood Search (VNS) | 175.20 | 25.80 | 70.0 | 50.0 | 8,845,120.0 |
| Tabu Search (TS) | 174.50 | 25.60 | 70.0 | 50.0 | 8,860,250.0 |
| Ant Colony Optimization (ACO) | 178.00 | 27.50 | 75.0 | 55.0 | 8,790,500.0 |
| Iterated Local Search (ILS) | 175.00 | 26.00 | 70.0 | 50.0 | 8,855,100.0 |
| Particle Swarm Optimization (PSO) | 185.00 | 35.00 | 85.0 | 65.0 | 8,650,200.0 |
| Whale Optimization Algorithm (WOA) | 182.00 | 32.00 | 80.0 | 60.0 | 8,690,400.0 |
| Elephant Herding Optimization (EHO) | 190.00 | 40.00 | 90.0 | 70.0 | 8,610,000.0 |
| Simulated Annealing (SA) | 195.00 | 42.00 | 95.0 | 75.0 | 8,510,300.0 |

---

## 4. Resultados de Inferencia Estadística No Paramétrica

### Prueba de Wilcoxon Signed-Rank (Proposed vs Baseline Metaheuristics)
- Nivel de significancia $\alpha = 0.05$.
- En las 9 comparaciones por pares frente a los algoritmos baseline, el Framework Propuesto (DTW) rechazó la hipótesis nula $H_0$ con un $p$-valor $p \le 3.63 \times 10^{-4} < 0.001$, demostrando superioridad estadística absoluta.

### Prueba Global de Ranking de Friedman
- **Estadístico de Friedman:** $\chi_F^2 = 129.17$
- **$p$-valor global:** $p = 6.91 \times 10^{-23}$
- **Ranking Medio:**
  1. **Proposed (Hybrid DTW):** `1.71` (1º Lugar)
  2. **Tabu Search (TS):** `2.95`
  3. **Iterated Local Search (ILS):** `3.10`
  4. **Variable Neighborhood Search (VNS):** `4.05`
  5. **Grey Wolf Optimizer (GWO):** `4.80`
  6. **Ant Colony Optimization (ACO):** `5.90`
  7. **Whale Optimization Algorithm (WOA):** `7.10`
  8. **Particle Swarm Optimization (PSO):** `7.30`
  9. **Elephant Herding Optimization (EHO):** `8.20`
  10. **Simulated Annealing (SA):** `9.89`
