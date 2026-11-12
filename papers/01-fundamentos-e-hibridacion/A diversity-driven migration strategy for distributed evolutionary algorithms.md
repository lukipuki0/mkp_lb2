# A diversity-driven migration strategy for distributed evolutionary algorithms

Ficha de lectura para el estado del arte. PDF: [PDF](./A%20diversity-driven%20migration%20strategy%20for%20distributed%20evolutionary%20algorithms.pdf). Entrada bibliográfica: [referencias.bib](../referencias.bib). Clave BibTeX: `Araujo_2023`.

Etiquetas secundarias: islas, clustering en línea, exploración y explotación, CEC 2014. Sección primaria: fundamentos e hibridación.

## Qué aporta

Araujo y Batista (2023) cambian la política migratoria de un evolutivo distribuido. DDMS decide cuándo migrar según la pérdida de diversidad y elige un migrante de diversidad sana, no solo el mejor individuo. El clustering en línea sirve a esa política; no optimiza el problema.

## Problema, método y contribución

Las islas colaboran con migraciones para retrasar la convergencia prematura. Migrar a intervalos fijos o probabilísticos acelera la convergencia si la frecuencia es alta, con riesgo de mínimos, o explora más si es baja, con convergencia lenta. Enviar siempre al mejor individuo homogeneiza las islas. DDMS estima el momento de migrar por la pérdida de diversidad. Un clustering en línea forma nubes de migrantes de buen fitness; el elegido sale de una nube cuya distribución difiere de la subpoblación de destino. Los autores llaman diversidad sana a esa variedad entre individuos de alto fitness. No es una hiperheurística ni un híbrido GA-PSO: es una política de migración dentro de evolutivos distribuidos.

## Resultados que el texto sostiene

La comparación es contra migraciones fija y probabilística en CEC 2014 (p. 1). En error medio de la función objetivo, DDMS es especialmente mejor en instancias de 50 y 100 dimensiones. En diversidad, es mejor en el 100 % de las funciones a 50D y en cerca del 96 % a 100D. El beneficio se concentra, según los autores, en funciones híbridas y de composición. La conclusión (p. 14) insiste en que la mayoría de los estudios sigue migrando a intervalos fijos o con probabilidad fija, y que el momento y la identidad del migrante importan. Las tablas de error, por ejemplo la tabla 12 en 10D (p. 14), muestran empates en varias funciones unimodales sencillas, coherente con la cautela de no vender DDMS como ganador universal en baja dimensión.

## Límites

CEC 2014 no es una microrred. No hay transferencia demostrada a problemas binarios de cobertura o a despacho energético.

Lectura crítica: el clustering no convierte el método en «red neuronal más optimizador». Es un control de la migración basado en distribución empírica. Útil para el estado del arte de exploración y explotación en modelos distribuidos. No llena ni abre, por omisión, un hueco de hidrógeno o de DTW.

## Evidencia

Título, autores, *Swarm and Evolutionary Computation* 82 (2023) 101361 y DOI en la p. 1. Cifras de diversidad en el resumen, p. 1. Conclusiones y tabla 12 en la p. 14. Entrada reutilizada de `papers/referencias.bib`.
