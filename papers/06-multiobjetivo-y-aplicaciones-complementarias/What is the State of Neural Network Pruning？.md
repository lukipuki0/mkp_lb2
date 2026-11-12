# What is the State of Neural Network Pruning?

Ficha de lectura para el estado del arte. PDF: [PDF](./What%20is%20the%20State%20of%20Neural%20Network%20Pruning%EF%BC%9F.pdf). Entrada bibliográfica: [referencias.bib](../referencias.bib). Clave BibTeX: `Blalock_2020`.

Etiquetas secundarias: meta-análisis, compresión de redes, ShrinkBench, evaluación estandarizada. Sección primaria: multiobjetivo y aplicaciones complementarias.

## Qué aporta

Blalock, Gonzalez Ortiz, Frankle y Guttag (MLSys 2020) no proponen un optimizador ni un sistema energético. Hacen un meta-análisis de la poda de redes neuronales y concluyen que el campo no puede compararse a sí mismo por falta de bancos de prueba y métricas comunes. Introducen ShrinkBench para evaluar métodos de poda bajo el mismo protocolo.

## Alcance, método y hallazgos

La poda anula parámetros con una máscara y suele reentrenar. El algoritmo tipo (p. 2) inicializa, entrena, puntúa, poda y ajusta, en uno o varios pasos. La poda no estructurada reduce parámetros sin garantizar velocidad en hardware denso; la estructurada quita neuronas, filtros o canales. La puntuación puede ser magnitud, importancia o contribución a activaciones y gradientes, local o global.

Revisan 81 artículos y podan cientos de modelos en condiciones controladas. El hallazgo central (p. 1) es que no se puede decir qué método gana ni cuánto avanzó el campo en tres décadas. Un cuarto de los papers no se compara con otro método; la mitad, como máximo con uno. Ningún par conjunto/red aparece en un tercio de los trabajos. Sí extraen que podar por magnitud comprime sin una caída sustancial de exactitud en los casos revisados, y que muchos métodos superan a la poda aleatoria. ShrinkBench unifica primitivas, modelos y entrenamiento. El apéndice (p. 18) mezcla publicados y preprints y convierte métricas de tamaño, no de velocidad, a partir del número de parámetros.

## Límites

No es una revisión de metaheurísticas ni de hidrógeno. La poda usa puntuaciones y, a veces, búsqueda sobre subredes, pero eso no la convierte en un algoritmo híbrido de optimización global.

Lectura crítica: el paralelo útil es metodológico. Igual que aquí no se puede coronar un método de poda sin protocolo común, no se debe coronar una metaheurística híbrida, un controlador o un DTW porque un resumen energético no los mencione. No aporta evidencia sobre HRES. El PDF no imprime DOI; arXiv:2003.03033 coincide en título, autores y resumen, y declara la publicación en MLSys 2020.

## Evidencia

Título, autores, MLSys 2020 y hallazgo de los 81 papers en la p. 1. Algoritmo de poda en la p. 2. Limpieza del corpus en la p. 18.

## Procedencia del archivo

- El nombre de archivo sustituye solo caracteres prohibidos en Windows (?) por equivalentes de ancho completo. El título de la ficha permanece completo.
- DataCite verifica el DOI de preprint 10.48550/arXiv.2003.03033 (título y autores coinciden; tipo Preprint). No reemplaza la cita de proceedings MLSys 2020. El campo DOI del estudio queda nulo.
