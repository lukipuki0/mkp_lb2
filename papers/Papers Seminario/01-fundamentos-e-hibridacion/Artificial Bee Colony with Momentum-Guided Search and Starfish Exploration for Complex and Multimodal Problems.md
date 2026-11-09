# Artificial Bee Colony with Momentum-Guided Search and Starfish Exploration for Complex and Multimodal Problems

**Clave de cita:** `Akbulut_2026`  
**PDF:** [PDF](./Artificial%20Bee%20Colony%20with%20Momentum-Guided%20Search%20and%20Starfish%20Exploration%20for%20Complex%20and%20Multimodal%20Problems.pdf) · **Bibliografía:** [referencias.bib](../../referencias.bib)

| Campo | Dato |
| --- | --- |
| Autores | Harun Akbulut |
| Año | 2026 |
| Venue | Applied Soft Computing, 201, 115699; versión previa a la Version of Record |
| DOI | 10.1016/j.asoc.2026.115699 |
| Sección primaria | `01-fundamentos-e-hibridacion` |
| Etiquetas | MSE-ABC; híbrido de operadores; CEC 2022; HPO de CNN; pre-proof |

## Aporte para el estado del arte

Ejemplifica un híbrido de bajo nivel entre dos bioinspirados: se conservan fases de ABC y se injertan operadores de exploración del starfish optimisation algorithm, más momento y paso exponencial. No es un controlador aprendido sobre trazas ni un modelo de planta renovable.

## Problema y método

El diagnóstico del ABC clásico es exploración limitada y convergencia temprana en problemas multimodales. MSE-ABC mantiene observadoras y exploradoras, añade operadores trigonométricos inspirados en SFOA, un mecanismo de momento que acumula historia de búsqueda y un control dinámico del paso. El autor lo distingue de un intercambio mecánico de operadores o de una ejecución secuencial: el momento debe sesgar la búsqueda desde exploración amplia hacia la explotación de ABC. La evaluación tiene tres bloques: CEC 2022, cinco problemas de diseño en ingeniería con restricciones y optimización de hiperparámetros de aprendizaje profundo en CIFAR-10. La comparación es contra ocho metaheurísticas, con Wilcoxon y Friedman. El bloque de CIFAR usa el algoritmo para ajustar una red; eso no convierte a MSE-ABC en un modelo neuronal.

## Qué sostiene el texto

El resumen y la conclusión afirman rendimiento superior o competitivo y significación no paramétrica. No transcribo medias de CEC porque el extracto revisado no fija una tabla única de errores. En CIFAR, el autor interpreta que el HPO regulariza la distribución de errores y que los fallos se concentran en clases parecidas, como gato y perro. La conclusión describe una transición dinámica del balance, no una regla fija por iteración.

## Límites

**Hallazgos de los autores.** El PDF es un pre-proof: no es la Version of Record y puede cambiar. El propio cierre dice que no se evaluó estabilidad de entrenamiento, energía ni hardware en el experimento de deep learning, y que falta análisis teórico de convergencia. Tampoco cubre entornos dinámicos o con restricciones variables en el tiempo.

**Lectura crítica.** Un buen CEC 2022 no transfiere a sizing de hidrógeno ni a alineación de series con DTW. El HPO de CNN es una aplicación del optimizador, no una hibridación de arquitectura de red con la metaheurística. No hay evidencia de control en línea por regresión.

## Relación con hibridación

Es hibridación de operadores y de memoria de momento dentro de un solo lazo de búsqueda. El experimento de CIFAR debe citarse aparte, como uso del mismo solver para hiperparámetros, no como prueba de pronóstico energético.

## Evidencia

- pp. 1-2: motivación, componentes y tres bancos de prueba.
- pp. 31-32: conclusión, límites declarados y carácter de pre-proof en la portada.
