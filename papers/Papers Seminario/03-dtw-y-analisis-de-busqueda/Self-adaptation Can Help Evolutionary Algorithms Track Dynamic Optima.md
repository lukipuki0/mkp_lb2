# Self-adaptation Can Help Evolutionary Algorithms Track Dynamic Optima

- PDF: [PDF](./Self-adaptation%20Can%20Help%20Evolutionary%20Algorithms%20Track%20Dynamic%20Optima.pdf)
- Bibliografía: [referencias.bib](../../referencias.bib)
- Clave de cita: `Lehre_2023`

## Lectura útil

Resultado teórico, no empírico: en Dynamic Substring Matching, una tasa de mutación fija pierde el óptimo móvil con alta probabilidad; la autoadaptación lo sigue.

## Ficha

| Campo | Dato verificado |
| --- | --- |
| Autores | Per Kristian Lehre y Xiaoyu Qin (orden alfabético) |
| Soporte | GECCO ’23, 15–19 julio 2023, Lisboa; pp. 1619–1627; ACM |
| DOI | 10.1145/3583131.3590494 |
| Tipo | Análisis de tiempo de ejecución; sin banco numérico |

El PDF incluye una portada de repositorio de la Universidad de Birmingham (p. 1 del archivo) y el artículo ACM desde la página siguiente. Las páginas citadas abajo son las del PDF completo, no las del artículo impreso 1619–1627.

## Qué demuestra

DSM pide seguir una secuencia de óptimos cuya estructura cambia. Mutación estática —Algoritmos 3 y 4 del apéndice— tiene probabilidad a lo sumo constante de alcanzar el óptimo vigente dentro del presupuesto, si la tasa es demasiado alta o baja (Lema 5.1, p. 9). El EA autoadaptativo, con la tasa codificada en el genotipo, sigue DSM con probabilidad 1 − e^{−Ω(n)} bajo las condiciones del Teorema indicado en la p. 9: μ polinomial, χ/μ = 0,4, A > 1 y cotas sobre b y p_inc. Aportan además un teorema por niveles con colas para acotar la probabilidad de hallar el óptimo actual dentro del presupuesto. No hay curva de convergencia ni comparación con PPO, Q-learning o DE.

## Límites

| Origen | Límite |
| --- | --- |
| Autores | El trabajo futuro es relajar DSM, otros problemas dinámicos y otros controles de parámetros, e identificar límites de la autoadaptación (p. 9). |
| Lectura crítica | Es un paisaje binario artificial con presupuestos de evaluación definidos para la prueba. No dice nada sobre tasas continuas, poblaciones de enjambre, DTW ni sistemas renovables. Una política neuronal no hereda esta garantía. La portada de Birmingham no cambia autores, año ni DOI. |

## Para el estado del arte

Ancla teórica: autoadaptar la mutación puede ser cualitativamente distinto de fijarla cuando el óptimo se mueve. Colocarlo junto a control adaptativo, no como evidencia experimental de un controlador aprendido.

Etiquetas secundarias: teoría de runtime; optimización dinámica; mutación; GECCO.

## Evidencia

- p. 2 del PDF: resumen, DSM y enunciado de probabilidad abrumadora.
- p. 9: parámetros del teorema, Lema 5.1 y conclusión.
