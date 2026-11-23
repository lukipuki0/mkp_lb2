# 07 · Híbridos GWO-DE

Seis artículos que combinan GWO y DE, leídos en texto completo para ubicar el GWO-DE actual: una población compartida, un candidato GWO y uno DE por individuo que compiten con aceptación greedy, y perfiles fijos de `a`, `F` y `CR` que se eligen leyendo la trayectoria del mejor fitness. Las páginas citadas son páginas físicas del PDF.

## Fichas

| Clave | Ficha | Problema | Tipo |
| --- | --- | --- | --- |
| `Zheng_2023` | [H-GWO, actuador hidráulico](./2023-hybrid-gwo-hydraulic-actuator.md) | Identificación de parámetros, continuo | Método y caso |
| `Ghasemi_2023` | [HDE](./peerj-cs-1420.md) | CEC-2014 y CEC-2019, continuo | Método |
| `Sifat_2024` | [IGWO-DE, SHE en PUC-5](./Engineering%20Reports%20-%202024%20-%20Sifat%20-%20Selective%20harmonic%20elimination%20in%20PUC%E2%80%905%20multilevel%20inverter%20using%20hybrid%20IGWO%E2%80%90DE.md) | Ángulos de conmutación, dos variables | Caso |
| `Wang_2025` | [de-AGWO, D{0-1}KP](./s00521-023-09075-x.md) | Mochila 0-1 con descuento, binario | Método y caso |
| `Yin_2026` | [AMSGWO-RL-MO, riego](./2026-adaptive-gwo-rl-irrigation.md) | Calendario de riego, tres objetivos | Método y caso |
| `Jadhav_2026` | [GWO-DE, capacitor de SEIG](./s41598-026-62604-1.md) | Capacitancia, una variable | Caso |

## Comparación de arquitecturas

La Tabla 1 resume cómo cada fuente resuelve las decisiones que separan un GWO-DE de otro. Cada fila se apoya en la ficha correspondiente; "no informa" significa que el texto completo no lo dice, no que falte en el código de los autores.

**Tabla 1.** Acoplamiento, control y evidencia en las seis fuentes.

| Clave | Acoplamiento | Candidatos por individuo | Aceptación | `a`, `F`, `CR` | Señal de la búsqueda | Élite y reinicio | Evidencia comparativa |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `Zheng_2023` | GWO y luego DE en serie (pp. 6-8) | 1 | Greedy (ec. 23) | Lineal, fijo, aleatorio | Ninguna | Sin archivo ni reinicio | 30 corridas en banco, sin pruebas; identificación sin réplicas declaradas |
| `Ghasemi_2023` | Moneda `Hm` elige mutación GWO o DE (p. 11) | 1 | Greedy (ec. 11) | `Hm`, `CR` fijos; `F` aleatorio; jHDE hereda jDE | Ninguna en HDE | Sin archivo ni reinicio | 30 corridas, 300 000 evaluaciones; Friedman y Wilcoxon entre funciones |
| `Sifat_2024` | Cruce binomial entre candidato GWO y donante DE (ec. 33) | 2 mezclados, no evaluados por separado | Reemplazo directo | Sigmoides en `t` | Ninguna | Sin archivo ni reinicio | Un resultado por método, sin réplicas |
| `Wang_2025` | `g` elige GWO o DE para toda la población (Algorithm 2) | 1 | Reemplazo directo | Horario con ruido | Ninguna | Líderes históricos; sin reinicio | 30 corridas, sin pruebas; presupuestos desiguales con resultados citados |
| `Yin_2026` | Q-learning elige 1 de 4 estrategias por generación (pp. 10-11) | 1 (granularidad no informada) | No informa | Estrategia DE con `F` aleatorio y `CR` fijo | ΔHV, diversidad, progreso; recompensa ΔHV | Archivo de Pareto; sin reinicio | 30 corridas, Wilcoxon; ablación que mezcla dos cambios |
| `Jadhav_2026` | `r < ρ(t)` elige GWO o DE por lobo (Algorithm 1) | 1 | GWO directo; DE greedy | `ρ(t)` exponencial, `F` lineal, `CR` fijo | Estancamiento del mejor (ec. 82) | Archivo de 3; reinicia el 20 % peor y conserva archivo y líderes | 30 corridas, Wilcoxon por pares; `z` inconsistentes en Tabla 16 |

## Síntesis

Ninguna de las seis fuentes genera dos candidatos completos por individuo y los hace competir por fitness. Cuatro eligen un único candidato, por moneda fija (`Ghasemi_2023`), por horario (`Jadhav_2026`, `Wang_2025`) o por política aprendida (`Yin_2026`); `Zheng_2023` los encadena y `Sifat_2024` los mezcla por dimensión. Esto delimita el lugar del modo competitivo del GWO-DE actual dentro de este grupo, no dentro de toda la literatura.

El control es casi siempre de lazo abierto. `a`, `F`, `CR` y la probabilidad de cada operador dependen de `t` o de ruido en cinco fuentes. Solo dos diseños propios leen la búsqueda: `Yin_2026` elige operador con un estado que incluye mejora del hipervolumen y diversidad, y `Jadhav_2026` reinicia parte de la población cuando el mejor fitness se estanca. La variante jHDE de `Ghasemi_2023` hereda la autoadaptación de jDE, en la que los `F` y `CR` asociados a ensayos exitosos pasan a la generación siguiente (p. 2). Ninguna usa la trayectoria del mejor fitness para cambiar parámetros, que es lo que hace el GWO-DE actual. `Ghasemi_2023` nombra la adaptación de `Hm` y `CR` como su carencia principal (p. 31).

El reinicio de `Jadhav_2026` conserva el archivo élite y a alfa, beta y delta (p. 20). Cualquier comparación con el GWO-DE actual debe partir de ese hecho, no de la lectura previa que suponía lo contrario.

`Wang_2025` es el único caso binario con reparación, pero resuelve la mochila 0-1 con descuento, con una restricción de capacidad y restricciones de grupo. No es el MKP y sus resultados no se trasladan sin una prueba propia.

La evidencia estadística es desigual. `Sifat_2024` no tiene réplicas; `Zheng_2023` y `Wang_2025` no tienen pruebas; `Yin_2026` y `Jadhav_2026` tienen pruebas cuyo reporte presenta inconsistencias señaladas en sus fichas. Ninguna fuente incluye una ablación que aísle el componente de control que propone. Estos seis artículos sirven para describir diseños, no para afirmar que un acoplamiento supera a otro.

## Extracción

- Herramienta: `pdftotext` (Xpdf 4.00) con `-layout`, página por página; para `Wang_2025` sin `-layout` y por página con `-f/-l`. Los fragmentos con pseudocódigo, ecuaciones o tablas dudosas se verificaron renderizando la página con `pdftoppm`.
- Páginas extraídas: `Zheng_2023` 19 de 19; `Yin_2026` 24 de 24; `Sifat_2024` 22 de 22; `Ghasemi_2023` 36 de 36; `Wang_2025` 17 de 17; `Jadhav_2026` 38 de 38.
- Problemas: las Tablas 2 de `Zheng_2023` y 4 de `Ghasemi_2023` salen desalineadas y no se transcriben; Algorithm 1 y 2 de `Wang_2025` y Algorithm 1 de `Jadhav_2026` no tienen capa de texto y se leyeron sobre la imagen renderizada.
- El texto extraído quedó fuera del repositorio; no hay `.txt` en esta carpeta, igual que en las secciones 01-06.

## Integración pendiente

`catalogo.json`, `../README.md`, `../ESTADO_DEL_ARTE.md` y `../papers_que_sirven.md` no incluyen todavía esta sección. `downloads.json` registra como no disponibles cuatro PDF que ahora están en la carpeta.
