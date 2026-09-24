
**Comentarios de los Revisores (Traducción al Español)

# Revisor 1

1. Falta de evidencia estadística para la afirmación principal. El resultado principal del artículo —que el método Binary-Complex es superior— se basa en una prueba de Friedman que no es estadísticamente significativa a nivel global (χ² = 4.13, p = 0.2474). Aunque la diferencia entre los rangos medios (2.22 frente a 2.83 para el peor método base) es real, resulta bastante reducida, y los propios autores admiten que esta clasificación debe interpretarse como una «tendencia favorable» y no como una prueba de superioridad. Esta es la mayor debilidad del manuscrito, ya que contradice directamente el argumento central de la investigación.
2. Sin comparación con sistemas de control adaptativo existentes.
3. Conjunto reducido de experimentos. Toda la experimentación se limita exclusivamente al problema de la mochila multidimensional (MKP); además, solo se selecciona el primer problema de cada uno de los nueve grupos de Chu–Beasley.
4. Ausencia de un estudio de sensibilidad de hiperparámetros. Decisiones de diseño clave —como la longitud de la ventana (W=200), el ancho de la banda de Sakoe-Chiba, los umbrales de percentiles (40/60), el factor de persistencia (τ_pat=3) y el umbral de mejora (π_max=5)— se fijaron de antemano sin examinar cómo influye su variación.
5. Inconsistencia de los efectos entre los distintos algoritmos.

# Revisor 2

El manuscrito propone un marco de control de configuración adaptativo guiado por trayectorias, en el que se utiliza el alineamiento temporal dinámico derivativo (Derivative Dynamic Time Warping, DDTW) para analizar las trayectorias recientes del mejor valor de aptitud (fitness) histórico y guiar la alternancia entre configuraciones predefinidas orientadas a la exploración y a la explotación. El marco se evalúa utilizando BPSO, GA, BGWO y BDE en nueve instancias del problema de la mochila multidimensional (MKP). Los resultados experimentales indican que la estrategia propuesta (Binary-Complex) puede aportar mejoras en diversas condiciones experimentales. El marco propuesto es potencialmente interesante, pero es necesario abordar varias cuestiones. A continuación, presento mis comentarios:

* Es necesario detallar la estructura del artículo al final de la Sección 1, describiendo brevemente el contenido de cada sección.
* El componente metodológico central del trabajo es el DDTW. Sin embargo, no se incluye un estudio de ablación lo suficientemente convincente que demuestre que el DDTW es el causante real de las mejoras observadas.
* El método propuesto utiliza una trayectoria de referencia lineal para el progreso sostenido y una trayectoria constante para el estancamiento, con una pendiente de referencia fija. No obstante, el manuscrito no justifica adecuadamente por qué estos dos únicos patrones son suficientes para caracterizar las dinámicas de búsqueda, altamente diversas, de BPSO, GA, BGWO y BDE.
* La comparación experimental con métodos representativos del estado del arte es insuficiente. Los autores deberían incorporar métodos adaptativos punteros adicionales como líneas base. De lo contrario, resulta difícil discernir si el controlador basado en DDTW aporta una mejora significativa frente a las técnicas existentes de control adaptativo de parámetros, o si simplemente supera a dos configuraciones fijas seleccionadas manualmente.
* El manuscrito describe el procedimiento algorítmico principal y la configuración experimental, lo cual es positivo. Sin embargo, en un método adaptativo basado en trayectorias, la reproducibilidad exige detalles de implementación más exhaustivos. De ser posible, convendría que los autores proporcionen un repositorio de código público para facilitar la reproducibilidad y verificación de los resultados.
* El resumen (abstract) debe corregirse desde la primera frase.

# Revisor 3

El manuscrito propone un mecanismo adaptativo basado en trayectorias que utiliza DDTW para identificar patrones de progreso y estancamiento, alternando entre configuraciones de parámetros orientadas a la exploración y a la explotación. El tema es relevante y el marco propuesto resulta potencialmente útil. No obstante, el manuscrito en su estado actual requiere mayores aclaraciones metodológicas y una validación experimental más sólida.

Mis comentarios:

* Definir claramente la brecha de investigación (research gap) y explicar en qué se diferencia el método propuesto respecto a los enfoques actuales de control adaptativo de parámetros e hiperheurísticas.
* Fortalecer la revisión de la literatura, en particular en lo relativo a control adaptativo de parámetros, aprendizaje por refuerzo, control difuso y análisis de búsqueda basado en trayectorias.
* Proporcionar una formulación matemática completa de DDTW, que incluya el cálculo de derivadas, la normalización de distancias y el tratamiento de condiciones de frontera.
* Incluir gráficas representativas que ilustren las trayectorias de fitness, las distancias DDTW, los umbrales y los estados del controlador.
* Aclarar el criterio de selección de los valores asignados a los parámetros de exploración y explotación, precisando si fueron ajustados de forma específica para las instancias evaluadas.
* Incluir el pseudocódigo completo del algoritmo.
* Los resultados muestran que el controlador adaptativo resulta más beneficioso para ciertos algoritmos, en especial BDE y GA, mientras que las mejoras en BPSO y BGWO son menos consistentes. Los autores deberían analizar el porqué de estas diferencias de interacción según el algoritmo. Diferencias en la dinámica poblacional, la sensibilidad a los parámetros, los mecanismos de reparación y las estrategias de exploración podrían explicar este comportamiento.
* El manuscrito ganaría notable claridad si incluyera figuras que muestren:
  * La trayectoria del mejor fitness histórico (best-so-far).
  * Las trayectorias de referencia de progreso y de estancamiento (meseta).
  * DR(t), DC(t) y Delta ti.
  * La evolución temporal del estado del controlador.
  * Las transiciones o cambios de parámetros correspondientes.

**
