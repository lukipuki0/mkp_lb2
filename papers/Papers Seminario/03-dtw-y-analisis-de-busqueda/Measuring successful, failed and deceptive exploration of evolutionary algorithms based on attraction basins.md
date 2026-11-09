# Measuring successful, failed and deceptive exploration of evolutionary algorithms based on attraction basins

- PDF: [PDF](./Measuring%20successful%2C%20failed%20and%20deceptive%20exploration%20of%20evolutionary%20algorithms%20based%20on%20attraction%20basins.pdf)
- Bibliografía: [referencias.bib](../../referencias.bib)
- Clave de cita: `Ravber_2026`

## Lectura útil

Propone medir exploración y explotación por pertenencia a cuencas de atracción, sin precalcularlas: mismo cuenca que el padre es explotación; otra cuenca es exploración. No es un algoritmo optimizador nuevo.

## Ficha

| Campo | Dato verificado |
| --- | --- |
| Autores | Miha Ravber, Jernej Jerebic, Matej Črepinšek, Luka Mernik, Stephen Chen, Marjan Mernik |
| Soporte | *Applied Soft Computing* 203 (2026) 115973 |
| DOI | 10.1016/j.asoc.2026.115973 |
| Tipo | Medida y estudio empírico de comportamiento de búsqueda |

## Método

Definen cuatro tipos: exploración exitosa, fallida, engañosa y rechazo exitoso, medidos por individuo. Si el hijo cae en la cuenca del padre, cuentan explotación; si no, exploración (p. 1). Lo aplican a seis problemas continuos de un objetivo y al problema real de Lennard-Jones, con DE/rand/1/bin, JADE, LSHADE, ABC y TLBO. Comparan la medida con indicadores indirectos de diversidad, que consideran imprecisos (p. 39). El código está en https://github.com/UM-LPM/exploration-and-exploitation.

## Qué sostiene el texto

Los EA se distinguen por cómo reparten esos tipos de exploración. ABC y TLBO se separan de las variantes DE (p. 39). La medida directa es más eficiente que medidas previas de cuencas que exigían precomputarlas. La selección elitista por fitness concentra la exploración fallida más problemática, atribuida a la selección voraz. No afirman que un algoritmo sea universalmente mejor.

## Límites

| Origen | Límite |
| --- | --- |
| Autores | Identificar la cuenca sin precomputarla sigue dependiendo de su procedimiento; el trabajo futuro incluye otras formas de selección (p. 39). |
| Lectura crítica | Seis funciones y Lennard-Jones no cubren combinatoria, multiobjetivo, DTW ni energía. La medida diagnostica una corrida; no controla parámetros por sí sola. No debe citarse como validación de un balanceador concreto del proyecto. |

## Para el estado del arte

Herramienta para discutir exploración sin equirarla a «hay mutación» o a «baja diversidad». Encaja en análisis de búsqueda, junto a Malan, y como criterio posible —no usado aún— para evaluar controladores adaptativos del lote.

Etiquetas secundarias: DE; JADE; LSHADE; ABC; TLBO; diversidad como proxy insuficiente.

## Evidencia

- p. 1: definición por cuencas y cuatro tipos.
- p. 39: algoritmos, Lennard-Jones, límite de la selección elitista y código.
