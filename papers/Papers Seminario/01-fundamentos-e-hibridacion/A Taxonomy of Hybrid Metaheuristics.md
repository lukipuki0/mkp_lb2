# A Taxonomy of Hybrid Metaheuristics

**Clave de cita:** `Talbi_2002`  
**PDF:** [PDF](./A%20Taxonomy%20of%20Hybrid%20Metaheuristics.pdf) · **Bibliografía:** [referencias.bib](../../referencias.bib)

| Campo | Dato |
| --- | --- |
| Autores | E.-G. Talbi |
| Año | 2002 |
| Venue | Journal of Heuristics, 8(5), 541-564 |
| DOI | 10.1023/A:1016540724870 |
| Sección primaria | `01-fundamentos-e-hibridacion` |
| Etiquetas | taxonomía; alto y bajo nivel; relay y teamwork; gramática; combinatoria |

## Aporte para el estado del arte

Es la referencia de vocabulario para decir qué clase de híbrido se está proponiendo. Sin esta taxonomía, llamar híbrido a LB2, a un memético o a un WOA de sizing mezcla diseños que no se comparan.

## Problema y método

Talbi no introduce un solver. Propone una clasificación común para la variedad de híbridos ya publicados en optimización combinatoria, extensible a heurísticas específicas y a métodos exactos. Separa diseño e implementación. En diseño, el primer corte es bajo nivel frente a alto nivel: en el bajo nivel una función interna de un método se sustituye por otro método; en el alto nivel los métodos siguen autocontenidos. El segundo corte es relay, en tubería, frente a teamwork, agentes cooperando en paralelo. Esas dicotomías generan cuatro clases. La implementación se trata como eje ortogonal, dependiente de plataforma y modelo de programación. La sección 4 generaliza los esquemas con una gramática. Cierra con una bibliografía anotada de 124 referencias clasificadas. El grano es alto: no abre el interior de lo que en 2002 ya se llamaba GA.

## Qué sostiene el texto

Sostiene que los mejores resultados conocidos de muchos problemas prácticos o académicos salen de híbridos, como combinación de búsqueda local, recocido, tabú y evolutivos. En la conclusión observa que gran parte de la literatura hibrida poblacionales con búsqueda local, porque los poblacionales puros exploran mal el ajuste fino en espacios muy combinatorios. También nota que la mayoría de los híbridos reseñados son secuenciales y que sus autores dejan la paralelización como trabajo futuro.

## Límites

**Hallazgos de los autores.** La taxonomía compara algoritmos de forma cualitativa. No hay métrica de calidad ni protocolo experimental.

**Lectura crítica.** El DOI no está impreso en el texto extraído; se acepta porque la consulta Crossref por título devolvió este artículo de Talbi, 2002, y la transformación BibTeX coincidió en título, autor y revista. El artículo es anterior al aprendizaje automático como componente de hibridación; esa extensión está en el survey de 2013 del mismo autor, no aquí. No menciona renovables, hidrógeno ni DTW, y esa ausencia no es un resultado.

## Relación con hibridación

Permite clasificar un balanceador por regresión como híbrido de alto nivel, relay o teamwork según si el modelo estadístico está dentro del operador o coopere por fuera. No dice cuál de esas clases es mejor.

## Evidencia

- pp. 1-2: objetivo, diseño frente a implementación y bibliografía de 124 referencias.
- p. 3: bajo nivel, alto nivel, relay y teamwork.
- p. 15: conclusión sobre poblacionales con búsqueda local y predominio secuencial.
