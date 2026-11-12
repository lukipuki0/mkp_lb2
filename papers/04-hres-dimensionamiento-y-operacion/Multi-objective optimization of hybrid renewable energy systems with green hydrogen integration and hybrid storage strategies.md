# P036 — NSGA-II integrado para un HRES con hidrógeno verde

**Título:** Multi-objective optimization of hybrid renewable energy systems with green hydrogen integration and hybrid storage strategies

**Cita:** `Tezer_2025` — Tuba Tezer (2025). *International Journal of Hydrogen Energy*, 142, 1249–1271. DOI: 10.1016/j.ijhydene.2025.03.006.

**PDF:** [PDF](./Multi-objective%20optimization%20of%20hybrid%20renewable%20energy%20systems%20with%20green%20hydrogen%20integration%20and%20hybrid%20storage%20strategies.pdf) · **Bibliografía:** [referencias.bib](../referencias.bib)

## Qué aporta

Dimensiona y opera un HRES con FV, eólica, reformador de hidrógeno verde continuo, electrolizador, tanque, batería y pila. El solver es i-NSGA-II, un NSGA-II acoplado a una estrategia de gestión de potencia, comparado con MOPSO. No es una red que optimiza pesos: es un algoritmo evolutivo multiobjetivo.

## Problema y método

Los objetivos son costo anualizado (ACS), LPSP y energía desperdiciada (PEW), con almacenamiento solo de hidrógeno o híbrido tanque–batería. Tamaño y gestión se optimizan juntos. Hay tres climas; el detalle tabulado es Çanakkale.

## Hallazgos del texto

En almacenamiento de hidrógeno, el resumen dice que las soluciones de mínimo ACS de i-NSGA-II bajan el ACS un 10 % respecto de MOPSO; las de mínimo LPSP lo reducen un 78 %; las de mínimo PEW lo reducen un 83 %. En almacenamiento híbrido, el ACS baja un 9 % y el LPSP un 85 % en las soluciones del objetivo correspondiente, y el PEW mínimo llega a 0 %. Quitar la batería sube el LPSP de 1,33 % a 1,85 % y el PEW de 1,02 % a 6,15 %. Sin tanque, el hidrógeno del reformador que excede la demanda no se guarda y el LPSP empeora.

## Límites

| Origen | Límite |
| --- | --- |
| Autores | La comparación es contra MOPSO, no contra un solver exacto. Los tres sitios muestran que priorizar un objetivo desplaza a los otros en el frente. |
| Lectura crítica | En la página 20 el relato del mínimo ACS habla de un LPSP 19 % menor, mientras las últimas columnas legibles de la Tabla 16 no cuadran con esa frase; no cito celdas de componentes porque el texto plano desalinea la tabla. La ganancia es del par algoritmo–gestión, no solo del NSGA-II. No hay control adaptativo en línea ni DTW, ni validación de campo. |

## Uso en el estado del arte

Caso multiobjetivo reciente de hidrógeno verde más almacenamiento híbrido, para contrastar con el WOA monoobjetivo de Sun y con el MILP sin hidrógeno de Malheiro.

Etiquetas secundarias: hidrógeno verde, reformador, MOPSO, Çanakkale, PEW.

## Evidencia

- PDF p. 1: i-NSGA-II, PMS y porcentajes frente a MOPSO.
- PDF p. 20: efecto de quitar tanque o batería (1,33 % → 1,85 % y 1,02 % → 6,15 %).
- PDF p. 21: Tabla 16, escenario 1 en Çanakkale, y lectura del frente.
