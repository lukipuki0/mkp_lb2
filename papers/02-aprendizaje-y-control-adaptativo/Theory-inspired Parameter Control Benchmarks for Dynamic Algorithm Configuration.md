# Theory-inspired Parameter Control Benchmarks for Dynamic Algorithm Configuration

- PDF: [PDF](./Theory-inspired%20Parameter%20Control%20Benchmarks%20for%20Dynamic%20Algorithm%20Configuration.pdf)
- Bibliografía: [referencias.bib](../referencias.bib)
- Clave de cita: `Biedenkapp_2022`

## Lectura útil

Construye benchmarks de parameter control con política óptima conocida, para no evaluar un configurador dinámico solo contra heurísticas opacas.

## Ficha

| Campo | Dato verificado |
| --- | --- |
| Autores | André Biedenkapp; Nguyen Dang; Martin S. Krejca; Frank Hutter; Carola Doerr |
| Soporte | GECCO 2022, pp. 766-775 |
| DOI | 10.1145/3512290.3528846 |
| Tipo | Benchmark de control de parámetros |
| Páginas del PDF | 10 |

## Método

El resumen (p. 1) extiende LeadingOnes a políticas que eligen la tasa solo desde un portafolio finito, y calcula portafolios óptimos de un tamaño dado. Prueba el benchmark con DDQN.

## Qué sostiene el texto

La p. 1 sostiene que casi no hay casos con política óptima conocida y que LeadingOnes es una excepción. No se copia el resultado numérico de DDQN.

## Límites

| Origen | Límite |
| --- | --- |
| Autores | El caso analizado en la apertura es LeadingOnes, no un problema de ingeniería. |
| Lectura crítica | Es un banco para DAC, no un solver. No sustituye a Adriaensen_2022, que formaliza el problema. No es DTW. |

## Para el estado del arte

Permite decir si un controlador dinámico se acerca a una política conocida. Un umbral de estancamiento no tiene esa verdad de terreno.

Etiquetas secundarias: DAC; LeadingOnes; portafolio de parámetros; DDQN.

## Evidencia

- p. 1: DOI, LeadingOnes, portafolio finito de parámetros y el uso de DDQN.

## Procedencia del archivo

El PDF llegó suelto en `papers/3512290.3528846.pdf` y se movió a esta sección. La clave ya existía en `referencias.bib`. La ficha usa la apertura del PDF, no una relectura de las tablas.
