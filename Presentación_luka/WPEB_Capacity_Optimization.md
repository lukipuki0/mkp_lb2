# Capacity Optimization of a Wind-Photovoltaic-Electrolysis-Battery (WPEB) Hybrid Energy System for Power and Hydrogen Generation

**International Journal of Hydrogen Energy 52 (2024) 311–333**

**Authors:** Runzhao Li a,b,*, Xiaoming Jin a,c, Ping Yang b,d, Ming Zheng a, Yun Zheng a,c, Chunrong Cai a,f, Xiang Sun e, Zhibin Luo a,f, Luyao Zhao a, Zhaohe Huang a, Wenzhao Yang g

**Affiliations:**
- a China Energy Engineering Group Guangdong Electric Power Design Institute Co., Ltd., Guangzhou, 510663, China
- b School of Electric Power Engineering, South China University of Technology, Guangzhou, 510640, China
- c State Electric Power Planning and Research Center (South Branch), Guangzhou, 510663, China
- d Guangdong Key Laboratory of Clean Energy Technology, Guangzhou 510640, China
- e China Energy Engineering Group Hydrogen Energy Co., Ltd., Beijing, 100020 China
- f Guangzhou China-Germany Hydrogen Energy Research Institute, Guangzhou, 510663, China
- g Shenzhen Gas Corporation Ltd., No.268, Meiao 1st Road, Futian District, Shenzhen, 518049, China

*Corresponding author. China Energy Engineering Group Guangdong Electric Power Design Institute Co., Ltd., Guangzhou, 510663, China.
Email: lirunzhaobanana@foxmail.com (R. Li)

DOI: https://doi.org/10.1016/j.ijhydene.2023.08.029

**Article history:**
- Received 28 April 2023
- Received in revised form 1 August 2023
- Accepted 3 August 2023
- Available online 17 August 2023

---

## Highlights

- WPEB system capacity optimization method is proposed.
- The wind-solar complementary metric is used to form a typical meteorological year.
- Increased wind power capacity reduces LCOE, grid sales capacity and net generation volatility.
- The optimal case is W190–P10-E95-B30 with LCOE and LCOH of 0.2692¥/kWh, 14.1574¥/kg.
- The IRR and ROI of W190–P10-E95-B30 are 12.60% and 78.97%.

---

## Abstract

Utility-scale (>10 MW) Wind-Photovoltaic-Electrolysis-Battery (WPEB) system is an emerging technology that adopts open loop "Power-to-H₂" architecture for large-scale green hydrogen production. It applies to curtailment reduction in areas with abundant wind and solar energy resources. The traditional residential-scale (0–1 MW) or commercial/facility-scale (1–10 MW) WPEB systems usually adopt the close loop "Power-to-H₂-to-Power" structure for power supply in the microgrid. The capacity optimization of the commercial-scale and residential-scale WPEB system has been studied extensively, but the research for utility-scale WPEB systems is rare.

This work proposes a multi-constraint single-objective capacity optimization method for the utility-scale WPEB system. The optimization objective is to minimize the levelized cost of electricity (LCOE), with constraints of power balance, annual grid sales rate (AGSR), and wind/PV/electrolysis station capacity considered. The feasible domain is identified by 8760-h production simulation, and the gradient descent method is applied to search for the optimal solution.

**Main conclusions:**
1. Increasing the wind-to-generation capacity ratio can reduce the LCOE, required grid sales capacity, and net generation curve fluctuation.
2. The lower and upper limits of the electrolysis station capacity are determined by the AGSR constraint and the LCOE target.
3. The battery bank time-shifts electricity to reduce the downtime of the electrolysis station. Thus, its power capacity should be no less than the minimum load ratio of the electrolysis station, and a discharge duration greater than 1 h is preferred.

Overall, this work provides a capacity optimization roadmap for the utility-scale WPEB system, which plays a significant role in renewable electricity-based hydrogen production and curtailment reduction.

**Keywords:** Wind-photovoltaic-electrolysis-battery (WPEB) system; Power-to-Hydrogen; Capacity optimization; Minimize levelized cost of electricity (LCOE); 8760-h production simulation

---

## Nomenclature

### Symbols

| Symbol | Description |
|---|---|
| Capacity_wind | The capacity of the wind farm |
| Capacity_PV | The capacities of the PV farm |
| Capacity_electrolyzer | The capacities of the electrolysis station |
| C_annualized,nominal | The annualized nominal cost of the system |
| E_grid,sales | Electricity sales to the grid |
| E_PV | Electrical production from PV farm |
| E_served | The total electric load served |
| E_served,ACprim | AC primary load served |
| E_served,DCprim | DC primary load served |
| E_served,def | Deferrable load served |
| E_wind | Electrical production from wind farm |
| f_global,n | The occurrence probability of the nth category Pearson coefficient during 2001–2021 |
| f_t,1, f_t,2, f_t,n | The occurrence probability of the nine Pearson coefficient classes |
| n | The nth category Pearson coefficient class |
| P_battery loss | Inverter loss |
| P_electrolyzer load | Electrolyzer load |
| P_grid purchase | Power purchased from grid |
| P_grid sales | Power sales to grid |
| P_plant load | Battery power loss |
| P_PV | PV power output |
| P_rectifier loss | Rectifier loss |
| P_wind | Wind power output |
| r_xy | Pearson coefficient |
| x | The wind speed at a specific height above the ground |
| x̄ | The mean value of wind speed at a specific height above the ground |
| y | Global horizon irradiance |
| ȳ | The mean value of global horizon irradiance |

### Abbreviations

| Abbreviation | Meaning |
|---|---|
| AC | Alternating Current |
| AGSR | Annual Grid Sales Rate |
| BA | Bat Algorithm |
| CAPEX | Capital Expenditure |
| CF | Capacity Factor |
| COV | Coefficient Of Variation |
| CRF | Capital Recovery Factor |
| CRITIC | Criteria Importance Though Intercrieria Correlation |
| CS | Chaotic Search |
| DC | Direct Current |
| DE | Differential Evolution |
| ED | Euclidean distance |
| GA | Genetic Algorithm |
| GHI | Global Horizontal Irradiance |
| GHR | Global Horizontal Radiation |
| HCHSA | Hybrid chaotic search/harmony search/simulated annealing |
| HVAC | High Voltage Alternating Current |
| HVDC | High Voltage Direct Current |
| IGDT | Information Gap Decision Theory |
| IRENA | International Renewable Energy Agency |
| BEIS | Department for Business, Energy and Industrial Strategy |
| IRR | Internal Rate of Return |
| LCOE | Levelized Cost of Electricity |
| LCOH | Levelized Cost of Hydrogen |
| LOLP | Loss Of Load Possibility |
| MOEA/D | Multi-Objective Evolutionary Algorithm based on Decomposition |
| MOPSO | Multi-Objective Particle Swarm Optimization |
| NASA | National Aeronautics and Space Administration |
| NPC | Net Present Cost |
| NPV | Net Present Value |
| NSGA-II | Non-Dominated Sorting Genetic Algorithm II |
| O&M | Operating & Maintenance |
| POWER | Prediction of Worldwide Energy Resources |
| PSO | Particle Swarm Optimization |
| PV | Photovoltaics |
| ROI | Return On Investment |
| SOC | State Of Charge |
| SPEA-II | Strength Pareto evolutionary algorithm |
| STDEV | Standard Deviation |
| TOPSIS | Technique for Order Preference by Similarity to Ideal Solution |
| WPEB | Wind-Photovoltaic-Electrolysis-Battery |

---

## 1. Introduction

China is striving to reach peak CO₂ emission before 2030 and achieve carbon neutrality before 2060 [1–3]. China has accelerated the energy scheme transition from a fossil fuel-dominant system to a renewable energy-dominant system since 2020. The wind & solar energy installed capacity in 2020 (planning base year) is 535.21 GW, and the planned installed capacity in 2030 is 1200 GW [4]. The average annual increment is 66.479 GW (Fig. 1).

China has planned two batches of large-scale wind & PV power bases in 2021 and 2022, each with a total installed capacity of 97.05 GW and 165 GW [5,6]. These large-scale wind & PV power bases concentrate on desertified land, Gobi desert, desertification land, and coal mining subsidence areas. The desertified land and desertification land cover 1.69 million km² and 2.57 million km², accounting for 17.58% and 26.81% of China's land area [7].

The vast majority of the large-scale wind & PV power bases are located in the northern provinces (e.g., Inner Mongolia Autonomous Region, Shanxi) and northwestern provinces (e.g., Ningxia, Shaanxi, Gansu, Qinghai, Xinjiang) [8,9] due to premium wind and solar energy resources (Fig. 2a). However, the bases are far from the load centers in the Southeast coastal areas (Fig. 2b) [10,11]. This inverse distribution between wind-solar resource endowment and electric load centers leads to huge wind and solar energy curtailment in resource-rich areas.

Inner Mongolia Autonomous Region (including two balancing regions, Mengxi and Mengdong) has the maximum wind energy curtailment of 6.09 TWh (5.06 TWh for Mengxi, 1.03 TWh for Mengdong) in 2021, with curtailment rates of 8.9% and 2.4% respectively (Fig. 3a). Solar energy curtailment is 0.59 TWh (0.56 TWh for Mengxi, 0.03 TWh for Mengdong), with curtailment rates of 3.5% and 0.6% respectively.

### WPEB Deployment in Inner Mongolia

So far, Inner Mongolia has launched 5 batches (1st batch issued in 2021; 2nd, 3rd batches issued in 2022; 4th, 5th batches issued in 2023) of wind-photovoltaic-electrolysis-battery (WPEB) systems to reduce curtailment rates [12–15]. The WPEB system utilizes wind & solar power to split water into hydrogen and oxygen.

The total installed capacity of the 5 batches of WPEB projects is 13.6525 GW (wind: 8.9835 GW, PV: 4.669 GW), with electrolyzer capacity accounting for 43.55% (5.945 GW) of the wind-PV capacity (Table 1).

**Characteristics of the 5 batches of WPEB systems:**

1. **Wind power dominance:** Wind power makes up the majority of total installed capacity due to higher capacity factor. Wind power proportion is 16.65% in the 1st batch but grows massively to 74.03%, 72.71%, 74.97%, and 100% in the 2nd–5th batches.

2. **Battery bank scaling:** The power and energy capacity of the battery bank increases with higher renewable energy penetration. The power capacity and discharge duration of the battery bank is 10% of the wind-PV installed capacity and 1 h in the 1st batch, increasing to 15% and 4 h in later batches.

3. **Grid sales constraints:** The electrolysis station operates only on renewable electricity, with excess electricity sold to the grid. The 1st batch requires the annual grid sales rate (AGSR) to be less than 20% of generated electricity with no purchase constraint. The 2nd–5th batches further demand that the WPEB system cannot purchase electricity from the grid except in emergency conditions.

4. **Primary mission:** The utility-scale WPEB system mainly applies to hydrogen generation and curtailment reduction rather than power supply to the grid. It is preferred that the system can operate in off-grid mode and reduce peak regulation pressure on the bulk electric grid.

### System Size Classification

The WPEB system can be divided into 3 categories by size:
- **Residential-scale:** 0–0.1 MW
- **Commercial/facility-scale:** 0.1–10 MW
- **Utility-scale:** >10 MW [16]

Capacity configuration plays a key role in WPEB system planning & design, influencing project economic viability. Published articles on WPEB capacity optimization are summarized in Table 2.

### Key Differences from Published Literature

1. **Application scenario:** Most WPEB systems (cases 1–20 in Table 2) are designed for reliable power supply to islands, buildings, communities, business parks, industrial parks, etc., similar to traditional hybrid energy systems for power generation only [17,18]. Hydrogen acts only as a long-term energy carrier, converted back to electricity. This work uses the WPEB system to produce large-scale hydrogen for curtailment reduction, with hydrogen used as fuel and chemical feedstock in transportation and industrial sectors.

2. **System architecture:** Power-supply WPEB systems typically adopt a close loop "Power-to-H₂-Power" structure using fuel cells or hydrogen engines to convert hydrogen back to power. This work applies an open loop "Power-to-H₂" structure where electrolyzers operate on renewable electricity for hydrogen production, mainly for chemical feedstock (methanol, ammonia, other electro-fuels) [19–21], with surplus electricity sold to the grid.

3. **System size:** Most capacity optimization research focuses on residential-scale and commercial/facility-scale WPEB systems (cases 1–19, 22–23 in Table 2). This work applies to utility-scale (>10 MW) systems.

4. **Optimization method:** Previous work uses techno-economic analysis, genetic algorithms, evolutionary algorithms (NSGA-II, Differential Evolution, SPEA-II), heuristic algorithms (PSO, Bat Algorithm), simulated annealing, weighting methods (CRITIC), decision analysis methods (TOPSIS, IGDT), and hybrid optimization methods (HCHSA). This work combines 8760-h production simulation and the gradient descent method, based on long-term (8760 h) wind & solar resource data rather than typical day/week data, accounting for volatility, intermittency, and randomness of renewable energy [22].

### Knowledge Gaps Addressed

1. Capacity configuration and source-grid interaction mechanisms of utility-scale grid-tied WPEB systems for large-scale hydrogen production [23].
2. The combination and roles between short-term energy storage (battery) and long-term energy storage (hydrogen) [24].
3. Cost reduction pathways needed to make WPEB systems economically competitive [25].

### Research Framework

This work proposes a multi-constrained single-objective capacity optimization method for a utility-scale grid-tied WPEB system. The objective is to minimize LCOE, with decision variables being the wind-to-generation capacity ratio and electrolyzer-to-generation capacity ratio. Constraints include power balance, annual grid sales rate, total installed capacity, and electrolysis capacity ratio.

- **Section 1:** Introduces renewable energy curtailment in northern/northwestern China and reviews WPEB deployment and optimization methods.
- **Section 2:** Presents system configuration and specifications (2.1), wind/solar resource analysis (2.2), and the optimization method (2.3).
- **Section 3:** Elaborates on the capacity optimization process (3.1), grid integration impact and cost reduction pathway (3.2), and study limitations/prospects (3.3).
- **Section 4:** Concluding remarks.

---

## 2. Methodology

### 2.1 Study Area, System Configuration and Specification

The studied site is located in Damao Banner, Baotou City, Inner Mongolia Autonomous Region, China. The practical WPEB system has 120 MW wind farm, 80 MW PV farm, 60 MW electrolysis station, and 20 MW battery bank; its techno-economic performance has been explored in ref. [59]. This work maintains a constant total wind-PV installed capacity of 200 MW, resizing wind power, PV, electrolysis station, and battery bank capacities to minimize LCOE.

The wind farm and PV farm connect to the AC bus through an AC/AC converter and DC/AC inverter respectively. The electrolysis station comprises a series of 5 MW electrolyzers (number depends on total capacity) that split water into hydrogen and oxygen. Hydrogen flow rate is controlled by a buffer tank, and hydrogen is transported to the storage system, compressed to 80 MPa, and stored in a high-pressure tank.

Two major hydrogen loads exist: the hydrogen fueling station (transportation sector) and ammonia synthesis (industry sector). Component specifications are in Table 3; costs by component type are in Table 4.

- Electrolyzer efficiency (ratio of higher heating value of produced hydrogen to consumed electricity): **75%**
- Nominal discount rate: **8%**
- Expected inflation rate: **3.5%**
- Real discount rate: **4.35%**
- Project lifetime: **25 years**

### 2.2 Resource Assessment

The summer of Damao Banner ranges from June to August, winter from December to February [60]. Wind speed at 50 m and global horizontal irradiance (GHI) are used as representative metrics of wind and solar resources. Hourly data from 2001 to 2021 was collected from NASA's Prediction of Worldwide Energy Resources (POWER) Project [61].

#### 2.2.1 Wind Energy Resource

- Mean wind speed: **7.56 m/s**
- Mean wind power density: **353.81 W/m²**
- Classification: **"Class 4 Good"** per national standard GB/T 18710–2002 [62], suitable for the vast majority of wind turbine applications.

Wind speed and power density show significant seasonal variation:
- **Windiest season (March–May):** average wind speeds of 8.22, 8.47, 8.46 m/s
- **Windier months (Nov–Dec):** 8.23 m/s, 8.19 m/s — all five months show a bulge from 9:00–16:00 (Fig. 5b), coinciding with solar PV output
- **Calmest months (July–August):** 6.05, 6.08 m/s
- **Calmer seasons (June, September):** 6.83, 6.85 m/s — these four months show troughs at 7:00–17:00, complementary to solar radiation occurrence

#### 2.2.2 Solar Energy Resource

- Annual global horizontal radiation (GHR): **1731 Wh/m²** (2001–2021 average)
- Classification: **"Class B Very abundant"** per national standard QX/T 89–2018 [63]

- **Brighter period (April–July):** daily radiation of 6.1638, 6.9201, 6.6419, 6.3643 kWh/m²/day
- **Darker seasons (Nov, Dec, Jan):** 2.7453, 2.1634, 2.4513 kWh/m²/day
- **Most atmospheric clearness:** October (clearness index 0.684)
- **Worst clearness:** July (clearness index 0.565)
- **Longest PV output period (May, June, July):** 4:00–18:00
- **Shortest PV output period:** December (6:00–15:00), February (7:00–16:00)

#### 2.2.3 Typical Meteorological Year Based on Wind-Solar Complementarity

The Pearson coefficient r_xy is used as the wind-solar complementarity metric:

```
r_xy = Σ(x - x̄)(y - ȳ) / √[Σ(x - x̄)² · Σ(y - ȳ)²]      (Eq. 1)
```

where x, y are wind speed and GHI, and x̄, ȳ are their means. r_xy ranges from -1 to 1, divided into 9 classes (Table 5), where -1 indicates perfect negative correlation (ideal complementarity) and +1 indicates perfect positive correlation.

The studied site shows an overall negative correlation between wind speed and GHI. Occurrence probability of -1.0 ≤ r_xy < -0.2 is **41.47%**; of 0.2 ≤ r_xy < 1.0 is **36.73%**.

Euclidean distance (ED) measures wind-solar complementarity between a particular year and the long-term average:

```
ED = √[Σₙ₌₁⁹ (f_t,n - f_global,n)²]      (Eq. 2)
```

**2008** was selected as the typical meteorological year, as its wind-solar complementarity is closest to the long-term (2001–2021) trend.

### 2.3 Multi-Constraints Single Objective Optimization

The objective function minimizes the levelized cost of electricity (LCOE):

```
min LCOE = C_annualized,nominal / E_served      (Eq. 3)
```

```
C_annualized,nominal = CRF(i, N) × NPC      (Eq. 4)
```

```
E_served = E_served,ACprim + E_served,DCprim + E_served,def + E_grid,sales      (Eq. 5)
```

**Decision variables:** wind-to-generation capacity ratio and electrolyzer-to-generation capacity ratio, determined by wind farm capacity and electrolysis station capacity.

- Initial search step for wind farm: 20 MW (5 MW turbine × 4), dynamically adjusted to 10 MW to accelerate solving
- Initial search step for electrolysis station: 5 MW (stack sizes up to 5 MW)
- Battery bank power capacity: set as 30% of electrolysis station capacity (matching minimum load ratio)
- Dispatch strategy: load following, with surplus electricity sold to the grid

**Constraints (Eq. 6):**

```
Power balance:
P_wind + P_PV + P_grid,purchase = P_electrolyzer,load + P_plant,load + P_battery,loss + P_inverter,loss + P_rectifier,loss + P_grid,sales

Annual Grid Sales Rate (AGSR):
AGSR = E_grid,sales / (E_wind + E_PV) × 100% ≤ 20%

Total installed capacity:
Capacity_wind + Capacity_PV = 200 MW

Electrolyzer capacity limit:
0 < Capacity_electrolyzer ≤ 0.5 × (Capacity_wind + Capacity_PV)

Capacity_wind ≥ 0
Capacity_PV ≥ 0
```

**Optimization process (Fig. 9):** Input electric/hydrogen load, system architecture, natural resource, and economic parameters. Decision variables (wind-to-generation ratio, electrolyzer-to-generation ratio) are varied independently to form the solution set. 8760-h production simulation identifies feasible solutions meeting all constraints. The minimum LCOE occurs at the point with the lowest gradient or at boundary points; the global optimum is found via gradient descent, compared against boundary point LCOEs.

---

## 3. Results and Discussion

### 3.1 Main Findings

**Feasibility domain (Fig. 10):**
- Electrolyzer-to-generation capacity ratio must be ≥ 0.375 (75 MW electrolysis station) to satisfy AGSR ≤ 20%, corresponding to a wind-to-generation ratio of 0.5 (100 MW wind, 100 MW PV).
- The feasible wind-to-generation ratio widens as the electrolyzer-to-generation ratio increases — e.g., at electrolyzer ratio 0.425 (85 MW), wind ratio feasible range is 0.3–0.7 (60–140 MW wind farm).
- This expansion occurs only on the wind farm side, not the PV side, since PV output (daytime only) requires greater electrolyzer capacity to reduce grid sales compared to wind power.

**LCOE minimization (Fig. 11):**
- Increased wind-to-generation ratio and reduced electrolysis station capacity decrease LCOE within the feasible domain.
- Capacity factor (CF): wind farm **40%**, PV farm **19%** (vs. China 2021 averages of 36.1% [65] and 16.02% [26])
- **Optimal case: W190-PV10-E95-B30** (190 MW wind, 10 MW PV, 95 MW electrolysis, 30 MW/30 MWh battery)
  - Minimum LCOE: **0.2692 ¥/kWh**
  - Electrolysis station capacity factor: **62.40%**

**Electrolysis station operation scenarios:**
1. **Grid-connected:** stable power operation, CF > 90%, or off-peak only operation, CF ~40–70% [66,67]
2. **Renewable energy connected:** CF depends on renewable generation, ranges 51–62% in this scenario [67]
3. **Curtailed electricity:** operates only on curtailed electricity, low CF ~8–35% [66,67]

**Comparison of cases** (W190-PV10-E95-B30, W120-PV80-E80-B25, W60-PV140-E85-B27.5):

| Metric | W190-PV10-E95-B30 | W120-PV80-E80-B25 | W60-PV140-E85-B27.5 |
|---|---|---|---|
| Total electricity generation (21 yrs) | 13.72 TWh | 11.32 TWh | 9.26 TWh |
| Equivalent capacity factor | 37.29% | 30.77% | 25.18% |
| Dominant wind-PV power range | 120–130 MW | 70–80 MW | 30–40 MW |
| Max electric power generation | 154.07 MW | 176.93 MW | 199.91 MW |
| Required grid sales capacity | 59.07 MW | 96.93 MW | 144.91 MW |
| Max net power output (typical day) | 58.71 MW | 96.57 MW | 114.26 MW |
| Average net generation | 15.37 MW | 9.29 MW | 10.64 MW |
| STDEV | 14.60 MW | 16.77 MW | 18.65 MW |
| COV | 0.9498 | 1.8061 | 1.7524 |

Increasing the wind-to-generation capacity ratio attenuates net generation curve fluctuation and reduces battery bank discharge duration requirements for downtime reduction.

### General Directions for WPEB Capacity Configuration

1. At the design/planning stage, minimizing LCOE is the objective, primarily determined by local wind/solar resources, performance parameters, technology-specific costs (CAPEX, O&M, replacement, salvage), and AGSR. At the operation stage, the objective shifts to maximizing NPV, determined by hydrogen selling price, feed-in tariff, tradable green certificate price, and dispatching priority.
2. Increasing wind-to-generation ratio exploits the higher capacity factor of wind vs. PV.
3. Increasing wind-to-generation ratio lowers required grid sales capacity and grid extension costs.
4. Increasing wind-to-generation ratio reduces net generation curve fluctuation, benefiting large-scale renewable integration.
5. Electrolysis station capacity is co-determined by the AGSR constraint and the LCOE objective.
6. Battery bank power capacity should be no less than the electrolysis station's minimum load ratio, with discharge duration no less than 1 h to reduce electrolyzer downtime.

### 3.2 Comparison with Other Studies

**Economic performance ranking:** W190-P10-E95-B30 > W120-P80-E80-B25 > W60-P140-E85-B27.5

| Metric | W190-P10-E95-B30 | W120-P80-E80-B25 | W60-P140-E85-B27.5 |
|---|---|---|---|
| NPV | 2113.08 million¥ | 1605.46 million¥ | 750.91 million¥ |
| Payback period | 9 years | 10 years | 13 years |
| IRR | 12.60% | 12.17% | 8.32% |
| ROI | 78.97% | 68.31% | 32.45% |
| LCOE | 0.2692 ¥/kWh | 0.2886 ¥/kWh | 0.3617 ¥/kWh |
| NPC | 2675.73 million¥ | 2350.38 million¥ | 2114.53 million¥ |

- Global weighted average LCOE (2021): onshore wind **0.2283 ¥/kWh**, solar PV **0.3332 ¥/kWh** [65]
- LCOE order (low to high): global onshore wind < W190-PV10-E95-B30 < W120-P80-E80-B25 < global solar PV < W60-P140-E85-B27.5
- LCOH (alkaline electrolyzers, 2022): W190-P10-E95-B30 **10.6248 ¥/kg**, W120-P80-E80-B25 **32.8202 ¥/kg**, W60-P140-E85-B27.5 **43.4450 ¥/kg**
- BEIS estimated LCOH for alkaline electrolyzers using curtailed electricity: 15.5051 (2020), 13.9451 (2025), 12.6460 (2035), 12.3646 (2040), 12.2164 (2045), 12.0690 ¥/kg (2050)
- **W190-P10-E95-B30 is cost-competitive in both LCOE and LCOH.**

**Cost breakdown (W190-P10-E95-B30):**
- CAPEX accounts for **73.30%** of total NPC (fixed cost dominates)
- Electrolysis station is the most expensive component: **44.65%** of NPC for W190-P10-E95-B30, **42.81%** for W120-P80-E80-B25, **50.56%** for W60-P140-E85-B27.5
- Unit CAPEX: wind farm (5917 ¥/kW) > PV farm (4633 ¥/kW), but wind's higher capacity factor results in lower overall LCOE

### 3.3 Research Implication, Limitation and Prospect

During planning & design of renewable-based hydrogen production, capacity configuration of the renewable power source and electrolysis station is the top priority, influencing component capacity factor, excess electricity rate (especially off-grid), CAPEX, and ultimately NPC/IRR. This work's optimization method (8760-h simulation + gradient descent) improves real-world economic performance by using a typical meteorological year.

**Limitations and future research directions:**
1. This work uses LCOE as a single objective function; future work should consider multi-objective optimization (NPC, LCOE, LCOH, excess electricity rate, etc.).
2. Total wind-PV installed capacity is held constant at 200 MW; future work could vary this and study impact on excess electricity rate.
3. This work assumes hydrogen load is high enough to consume all produced hydrogen; future work should study impact of hydrogen/electric load profile volatility and uncertainty on capacity configuration.
4. Battery bank power capacity and discharge duration (1 h in this work) could become decision variables in future studies, examining effects on system cost, excess electricity rate, and electrolysis capacity factor.

---

## 4. Conclusions

Utility-scale WPEB systems are an emerging technology for large-scale hydrogen production and curtailment reduction in resource-abundant areas. Capacity configuration is the first step of power system planning & design, determining life-cycle techno-economic performance. This work proposes a WPEB capacity optimization method for open-loop "Power-to-H₂" systems, using Damao Banner (200 MW total wind-PV capacity) as a case study.

**Key results:**

1. LCOE is used as the objective function, with wind-to-generation and electrolyzer-to-generation capacity ratios as decision variables. Constraints include power balance, annual grid sales rate, and wind/PV/electrolysis station capacity. Combining 8760-h production simulation and gradient descent identifies the feasible domain and optimal solution: **190 MW wind farm - 10 MW PV farm - 95 MW electrolysis station - 30 MW/30 MWh battery bank (W190-P10-E95-B30)** at the studied site.

2. Increasing the wind-to-generation capacity ratio reduces LCOE, required grid sales capacity, and net generation curve fluctuation. Electrolysis station capacity is co-determined by AGSR constraint and LCOE objective. The battery bank's power capacity should equal or slightly exceed the electrolysis station's minimum load ratio, with discharge duration no less than 1 h.

3. For W190-P10-E95-B30: NPV = **2113.08 million¥**, NPC = **2675.73 million¥**, IRR = **12.60%**, ROI = **78.97%**, LCOE = **0.2692 ¥/kWh**, LCOH = **14.1574 ¥/kg**, payback period = **9 years**.

---

## Declaration of Competing Interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Acknowledgement

This work is supported by the Postdoctoral International Exchange Program Introduction Project (No. YJ20220212), Postdoctoral Research Project Start-Up Fund of Huangpu District, Guangzhou (No. RS20220714-008), Postdoctoral Research Project of Guangdong Electric Power Design Institute (No. EV10401W), and R&D Program of Guangdong Electric Power Design Institute (No. EV11041W). The authors thank Shenzhen Gas Corporation Ltd. for technical guidance and the reviewers for their invaluable suggestions.

---

## References

*(Full reference list available in the original publication; key sources include national policy documents from China's NDRC/NEA, NASA POWER project data, national standards GB/T 18710-2002 and QX/T 89-2018, and IRENA/BEIS techno-economic reports.)*

See original PDF for the complete numbered reference list [1]–[68].

---

*Note: This Markdown document is a structured conversion of the original PDF. Tables 1–5 and Figures 1–16 referenced in the text (photos, charts, and detailed project tables) are described in-text but are best viewed in the original PDF for full visual/graphical content.*
