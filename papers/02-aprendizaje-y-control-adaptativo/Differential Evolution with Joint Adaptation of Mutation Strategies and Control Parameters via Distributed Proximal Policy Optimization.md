# Differential Evolution with Joint Adaptation of Mutation Strategies and Control Parameters via Distributed Proximal Policy Optimization

- PDF: [PDF](./Differential%20Evolution%20with%20Joint%20Adaptation%20of%20Mutation%20Strategies%20and%20Control%20Parameters%20via%20Distributed%20Proximal%20Policy%20Optimization.pdf)
- Bibliografía: [referencias.bib](../referencias.bib)
- Clave de cita: `Ding_2025`

## Lectura útil

DRLDE no fusiona DE con otro metaheurístico: una política de PPO distribuido elige la pareja discreta de mutación y F/CR.

## Ficha

| Campo | Dato verificado |
| --- | --- |
| Autores | Wenjie Ding, Mengtao Qian, Chao Lu, Jin Yi, Huayan Pu, Jun Luo |
| Soporte | *Tsinghua Science and Technology*, vol. 31, n.º 1, pp. 101–124; cabecera impresa: febrero de 2026 |
| DOI | 10.26599/TST.2024.9010185 |
| Tipo | Algoritmo de control adaptativo con experimento numérico, no revisión |

Se conserva la clave `Ding_2025` y el DOI 10.26599/TST.2024.9010185. El PDF verifica el número oficial: Volume 31, Number 1, February 2026, pp. 101–124, y «© The author(s) 2026» (p. 1). Crossref registra issued y published-print en agosto de 2025 y no trae published-online. La entrada bibliográfica usa el año y el mes del número impreso (2026, febrero) y deja las dos fechas en esta nota; no se creó otra clave.

## Qué resuelve y cómo

La mayoría de los controles por aprendizaje en DE adaptan la mutación o bien F/CR, no ambos (pp. 1 y 4). DRLDE elige en un espacio de 12 acciones (Tabla 1, p. 8): `DE/rand/1`, `DE/current-to-pbest/1`, `DE/current-to-rand/1` y `DE/rand-to-best/2`, cada una con tres pares (F, CR). El estado junta paisaje de fitness, estadísticas del proceso y diversidad. Entrenan offline con 750 instancias BBOB y aplican el modelo en línea. La red no sustituye al evaluador: solo elige la configuración; DE sigue mutando, cruzando y seleccionando.

## Qué sostiene el texto

En CEC2013 y CEC2017 no vistos, y en un rover de 60 dimensiones, los autores describen un desempeño competitivo (pp. 1, 4 y 20). F15 y F23, de paisaje parecido, reciben selecciones parecidas (p. 16). En el rover, DRLDE queda primero entre los comparados en 10 repeticiones (p. 20). No se copian errores medios que el pasaje no fija.

## Límites

| Origen | Límite |
| --- | --- |
| Autores | El entrenamiento sigue siendo costoso aun con 50 agentes; proponen RL más eficiente en muestras y estudiar la política aprendida (p. 20). |
| Lectura crítica | Doce acciones discretas no cubren F y CR continuos. Entrenar en BBOB y probar en CEC no demuestra transferencia a control, DTW, hidrógeno o dimensionamiento de HRES: el PDF no trata esos dominios. «Universal» (p. 20) es una proyección, no un resultado. |

## Para el estado del arte

Precedente de control conjunto estrategia–parámetro con política neuronal y paisaje en el estado. No confundirlo con híbridos DE+PSO/ABC, que el artículo trata como otra familia (p. 2).

Etiquetas secundarias: paisaje de fitness; caja negra continua; trayectoria, no energía.

## Evidencia

- p. 1: título, autores, DOI, resumen y hueco de adaptación conjunta.
- p. 4: 750 instancias BBOB, CEC2013/CEC2017 y rover 60-D; contribuciones.
- p. 8: Tabla 1, 12 acciones.
- pp. 16 y 20: adaptación por paisaje, ranking del rover y límite de cómputo.

## Procedencia del archivo

- La clave propuesta `La entrada reutilizada `Ding_2025` fija 2025 y agosto. El PDF dice febrero de 2026 y «© The author(s) 2026» (p. 1). Se conserva la clave y el DOI; el desfase queda advertido.` ya identificaba otro estudio. Esta ficha y su entrada BibTeX usan `Ding_2025`.
