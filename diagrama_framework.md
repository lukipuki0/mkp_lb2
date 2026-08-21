# Diagrama de Arquitectura y Flujo del Framework (incluyendo HRES2-H2)

Este documento contiene los diagramas del framework dibujados en cajas y líneas de texto (ASCII / Unicode Art) para su rápida lectura, inclusión en documentación Markdown o referencia visual.

---

## 1. Diagrama de Arquitectura del Framework (Vista de Capas)

```text
┌───────────────────────────────────────────────────────────────────────────────────┐
│                            1. DOMINIOS DE PROBLEMA                                │
├──────────────────────────┬──────────────────────────┬─────────────────────────────┤
│   MKP DISCRETO           │   CEC2022 CONTINUO       │   HRES2-H2 MIXTO (WPEB)     │
│ - OR-Library Instances   │ - Funciones F1 a F12     │ - Sistema Eólico-PV-Bat-El  │
│ - Problem Structure      │ - Unimodales,            │ - Minimización de LCOE/LCOH │
│ - Greedy Repair Engine   │   Multimodales, etc.     │ - Variables Continuas/Enteras│
└────────────┬─────────────┴────────────┬─────────────┴──────────────┬──────────────┘
             │                          │                            │
             ▼                          │                            ▼
┌──────────────────────────┐            │             ┌─────────────────────────────┐
│  2. BINARIZACIÓN (LB2)   │            │             │ 2. DECODIFICADOR WPEB       │
│ - Transfer (V1-V4, S1-S4)│            │             │ - Mapeo 4D Continuo a       │
│ - Reglas Mapeo L1 / L2   │            │             │   Variables Discretas/Reales│
└────────────┬─────────────┘            │             └──────────────┬──────────────┘
             │                          │                            │
             └──────────────────────────┼────────────────────────────┘
                                        │
                                        ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│               3. SUITE DE METAHEURÍSTICAS Y ROTACIÓN EN PIPELINE                  │
├──────────────────────────────────────────┬────────────────────────────────────────┤
│       METAHEURÍSTICAS POBLACIONALES      │       METAHEURÍSTICAS DE TRAYECTORIA   │
│  - PSO, GWO, WOA, ABC, EHO, ACO, GA, DE  │  - SA, TS, ILS, VNS                    │
└──────────────────────────────────────────┴───────────────────┬────────────────────┘
                                                               |
                                                               v
┌───────────────────────────────────────────────────────────────────────────────────┐
│                  4. MONITOR DINÁMICO Y CONTROL DE ESTANCAMIENTO                   │
├───────────────────────────────────────────────────────────────────────────────────┤
│  - DTW / DDTW Stagnation Monitor (Comparación D1 vs Rampa y D2 vs Constante)      │
│  - Orquestador Híbrido: Control de Racha de Paciencia y Rotación Secuencial       │
└──────────────────────────────────────────┬────────────────────────────────────────┘
                                           │
                                           v
┌───────────────────────────────────────────────────────────────────────────────────┐
│                      5. EVALUACIÓN, VISUALIZACIÓN Y ANALÍTICA                     │
├──────────────────────────────────────────┬────────────────────────────────────────┤
│         MÓDULO DE VISUALIZACIÓN          │          REPORTES & ANALÍTICA          │
│  - Curvas de Convergencia Global         │  - Generación de CSVs de Desempeño    │
│  - Fitness Instantáneo por Iteración     │  - Informes Markdown / TXT y GAP %    │
│  - Historial de Deltas DTW               │  - Pruebas Estadísticas HRES2         │
└──────────────────────────────────────────┴────────────────────────────────────────┘
```

---

## 2. Diagrama de Flujo Operativo y Ciclo de Ejecución (Pipeline DTW)

```text
                     ┌───────────────────────────┐
                     │     INICIO PIPELINE       │
                     └─────────────┬─────────────┘
                                   │
                                   ▼
                     ┌───────────────────────────┐
                     │ Inicializar Población /   │
                     │ Solución Inicial          │
                     └─────────────┬─────────────┘
                                   │
                                   ▼
                     ┌───────────────────────────┐
                     │  Seleccionar Algoritmo    │
                     │ (Poblacional/Trayectoria) │
                     └─────────────┬─────────────┘
                                   │
        ┌──────────────────────────┴──────────────────────────┐
        │  CICLO DE ITERACIÓN DE LA METAHEURÍSTICA (EPOCH)    │
        │                                                     │
        │  ┌───────────────────────────────────────────────┐  │
        │  │ 1. Generar nuevas soluciones                  │  │
        │  └───────────────────────┬───────────────────────┘  │
        │                          │                          │
        │                          ▼                          │
        │  ┌───────────────────────────────────────────────┐  │
        │  │ 2. Aplicar Binarización (LB2) / Decoder (WPEB)│  │
        │  └───────────────────────┬───────────────────────┘  │
        │                          │                          │
        │                          ▼                          │
        │  ┌───────────────────────────────────────────────┐  │
        │  │ 3. Evaluar Fitness & Actualizar Mejor Global  │  │
        │  └───────────────────────┬───────────────────────┘  │
        │                          │                          │
        │                          ▼                          │
        │  ┌───────────────────────────────────────────────┐  │
        │  │ 4. Actualizar Monitor DTW / DDTW              │  │
        │  └───────────────────────┬───────────────────────┘  │
        │                          │                          │
        └──────────────────────────┬──────────────────────────┘
                                   │
                                   ▼
                      /─────────────────────────\
                     < ¿Estancamiento Detectado? >
                      \  (D2 <= θc y D1 >= θr)  /
                       \───────────────────────/
                         │                   │
                     SÍ  │                   │  NO
                         ▼                   ▼
     ┌─────────────────────────┐       /─────────────────────────\
     │ 1. Registrar Switch Log │      < ¿Fin del Epoch Actual?  >
     │ 2. Abortar Epoch        │       \            ?            /
     └───────────┬─────────────┘        \───────────────────────/
                 │                                │
                 │                            SÍ  │  NO (Continuar Epoch)
                 │                                ▼  │
                 │                     ┌─────────────┴───────────┐
                 │                     │ Rotar a Siguiente MH    │
                 │                     │ Inyectar Mejor Solución │
                 │                     └─────────────┬───────────┘
                 │                                   │
                 └─────────────────┬─────────────────┘
                                   │
                                   ▼
                      /─────────────────────────\
                     < ¿Criterio de Parada?     >
                      \  (Max Iters / Tiempo)   /
                       \───────────────────────/
                         │                   │
                     NO  │                   │  SÍ
                         │                   ▼
                         │     ┌───────────────────────────┐
                         │     │ Generar Gráficos, CSVs    │
                         │     │ e Informes de Resultados  │
                         │     └─────────────┬─────────────┘
                         │                   │
                         │                   ▼
                         │     ┌───────────────────────────┐
                         └────>│           FIN             │
                               └───────────────────────────┘
```

---

## 3. Diagrama del Sistema HRES2-H2 (Modelo WPEB Extendido)

```text
┌───────────────────────────────────────────────────────────────────────────────────┐
│                    SISTEMA HRES2-H2: MODELO WPEB EXTENDIDO                        │
├───────────────────────────────────────────────────────────────────────────────────┤
│ ENTTRADA METEOROLÓGICA & PARÁMETROS OPERACIONALES                                 │
│ - NASA POWER API: Irradiación Solar G(t), Velocidad del Viento v(t), Temp T(t)    │
│ - Simulación de Demanda Eléctrica / Producción de Hidrógeno (8760 horas al año)   │
└──────────────────────────────────────┬────────────────────────────────────────────┘
                                       │
                                       ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│               ESPACIO DE BÚSQUEDA 4D & DECODIFICADOR MIXTO                        │
├──────────────────────────────┬──────────────────────────────┬─────────────────────┤
│  Potencia Eólica (wind_mw)   │  Electrolizador (n_el_cont)  │ Batería (mw & dur)  │
│  [0.0, 200.0] MW (Continua)  │  {10, 11, ..., 20} unidades  │ {0..50}MW, {1,2,4}h │
└──────────────────────────────┴──────────────┬───────────────┴─────────────────────┘
                                              │
                                              ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│                    SIMULACIÓN FÍSICA Y OPERACIONAL (8760 HRS)                     │
├───────────────────────────────────────────────────────────────────────────────────┤
│ 1. Cálculo de Generación PV (eficiencia térmica) y Eólica (curva cúbica)          │
│ 2. Despacho Prioritario: Generación -> Electrolizador -> Batería -> Red           │
│ 3. Verificación de Restricciones (AGSR <= 20%, Carga Mínima Electrolizador >= 30%)│
└──────────────────────────────────────┬────────────────────────────────────────────┘
                                       │
                                       ▼
┌───────────────────────────────────────────────────────────────────────────────────┐
│                      EVALUACIÓN DE OBJETIVOS ECONÓMICOS                           │
├──────────────────────────────────────────────┬────────────────────────────────────┤
│  LCOE (Levelized Cost of Electricity)        │  LCOH (Levelized Cost of Hydrogen) │
│  LCOE = NPC_total * CRF / E_anual            │  LCOH = NPC_total * CRF / m_H2     │
└──────────────────────────────────────────────┴────────────────────────────────────┘
```
