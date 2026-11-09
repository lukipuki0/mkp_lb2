# P027 — Catálogo de metaheurísticas híbridas multiobjetivo

**Título:** Hybrid Metaheuristic Frameworks for Multi-Objective Engineering Optimization Problems

**Cita:** `Makhmudova_2024` — Dilfuza M. Makhmudova y Saman M. Almufti (2024). *Qubahan Techno Journal*, 3(1). DOI de portada: 10.48161/qtj.v3n1a23.

**PDF:** [PDF](./Hybrid%20Metaheuristic%20Frameworks%20for%20Multi-Objective%20Engineering%20Optimization%20Problems.pdf) · **Bibliografía:** [referencias.bib](../../referencias.bib)

## Qué aporta

Revisión y catálogo de marcos híbridos para optimización multiobjetivo de ingeniería, con el ejemplo MOGSABAT (búsqueda gravitacional más murciélago). No es un banco de pruebas nuevo ni un controlador neuronal.

## Problema y método

El texto plantea objetivos en conflicto —costo, desempeño, confiabilidad— bajo incertidumbre y no linealidad. Clasifica hibridaciones paralelas, secuenciales y asistidas por aprendizaje, y discute formulaciones robustas con pruebas estadísticas. La Tabla 1 resume tipos de integración; el cierre remite a un catálogo parcial de algoritmos (Tabla 2) del que podrían construirse híbridos.

## Hallazgos del texto

La conclusión, en tabla, afirma que unir búsqueda global (GSA) y local (BAT) equilibra el desempeño, que formulaciones robustas y pruebas estadísticas aumentan la confiabilidad, y que el esquema se paraleliza. También dice que el ajuste de parámetros sigue siendo sensible y que faltan bancos comparables. La superioridad “en velocidad de convergencia y diversidad” se enuncia en el resumen y se apoya en resultados ya publicados (el texto remite a la Tabla 1 “based on results reported in” otras fuentes), no en un experimento cuyas medias aparezcan en la conclusión.

## Límites

| Origen | Límite |
| --- | --- |
| Autores | El catálogo no es exhaustivo. Señalan complejidad de implementación, costo computacional y falta de benchmarks estándar. El ajuste adaptativo queda como trabajo futuro. |
| Lectura crítica | Crossref lista autores (Zayyanu Yunusa, Rizwan Hameed) que no figuran en la portada; la ficha sigue a la portada. No hay tamaños de efecto reproducibles en las páginas de cierre. No trata HRES, hidrógeno ni DTW, y no abre una laguna de un proyecto solo porque no lo menciona. |

## Uso en el estado del arte

Mapa cualitativo de hibridación multiobjetivo (global/local, paralelo, con aprendizaje). Debe citarse como revisión, no como evidencia empírica primaria de MOGSABAT.

Etiquetas secundarias: multiobjetivo, GSA, BAT, incertidumbre.

## Evidencia

- PDF p. 1: autores de portada, DOI y promesa de Wilcoxon y MOGSABAT.
- PDF p. 11: sensibilidad paramétrica y falta de bancos comunes.
- PDF p. 12: Tabla 3 de conclusiones y aviso de que el catálogo es parcial.
