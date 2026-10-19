# Análisis de comentarios de revisores y acciones recomendadas

Este documento compara **cada comentario** de [`revisores.md`](revisores.md) con el manuscrito [`paper_final.tex`](paper_final.tex). El archivo de revisiones parece referirse a otro estudio en sus detalles (Binary-Complex y BPSO/BGWO/BDE), así que separo lo que no corresponde literalmente de las observaciones metodológicas que sí sirven para este trabajo de **DTW/DDTW, MKP, CEC2022 continuo y HRES2-H2**.

## Resumen ejecutivo

- **No copies** los algoritmos, cifras ni valores de parámetros de la revisión del compañero: no son los de este manuscrito.
- **Sí toma en serio** las críticas sobre controles adaptativos comparables, ablación, sensibilidad de parámetros, motivación metodológica y análisis por algoritmo.
- El paper ya contiene **pseudocódigo** y varias figuras de convergencia/monitor; algunas peticiones están atendidas parcialmente, pero conviene verificar si muestran todos los estados y señales que el revisor pide.
- Antes del envío, además, reemplaza el texto de plantilla y corrige la discrepancia de “nueve” frente a “diez” algoritmos base en HRES.

### Estados

- **Aplica**: la debilidad también está presente o no está resuelta en el paper actual.
- **Parcial**: el paper ya cubre una parte; falta justificar, completar o verificar.
- **No aplica literalmente**: el comentario critica otro método o resultados que no aparecen aquí; conserva solo la lección general indicada.

---

## Revisor 1

### 1. [x] Revisado — La afirmación de superioridad no estaría respaldada por Friedman

**Estado: Revisión completada; la objeción estadística del compañero no aplica literalmente, pero encontré una afirmación contradictoria que sí debes corregir.** Los valores χ²=4.13 y p=0.2474 son del estudio Binary-Complex, no de este paper. En el manuscrito actual, el Friedman de las cuatro configuraciones MKP es significativo (χ²=24.0667, p=2.419×10⁻⁵); en CEC2022 los Friedman por función también son significativos. Esos Friedman omnibus muestran diferencias entre métodos/configuraciones, pero por sí solos no demuestran que el híbrido sea superior.

**Qué muestran los contrastes reportados:** en MKP, los contrastes por instancia agregados para DTW--3K dan 41 victorias, 15 empates y 7 derrotas frente a siete baselines; en CEC2022, DTW da 38/16/6 y DDTW 37/19/4 frente a cinco baselines; en HRES2-H2, Mann--Whitney con Holm da 9/0/0 para cada variante frente a nueve baselines. Esto respalda mejoras frente a varios métodos evaluados, no superioridad universal. HRES usa Mann--Whitney porque los datos baseline se tratan como independientes; no es evidencia de Friedman.

**Cambio necesario en `paper_final.tex`:** la conclusión dice que no se detectó diferencia entre DTW y DDTW en ningún dominio (aprox. línea 1806), pero el análisis MKP afirma que los seis contrastes entre configuraciones son significativos tras Holm; incluye DTW--1K vs DDTW--1K y DTW--3K vs DDTW--3K. Corrige esa frase para decir que la diferencia depende del dominio: los contrastes MKP sí detectan diferencias entre variantes/configuraciones, mientras que CEC2022 y HRES2-H2 no detectan diferencia entre DTW y DDTW tras el ajuste correspondiente. No llames equivalencia al resultado no significativo.

**También conviene ajustar el abstract:** cambia “superior cross-domain performance” por una formulación acotada, por ejemplo: *“competitive performance across the three evaluated domains, with statistically significant improvements over several tested standalone baselines”*. La frase actual sobre mejoras significativas “over several standalone baselines” sí está respaldada por los contrastes reportados, siempre que no se interprete como superioridad general.

**Qué hacer:** conserva la afirmación únicamente con su alcance correcto; el Friedman no es el contraste directo de superioridad. Usa las pruebas pareadas por problema para MKP/CEC2022 y Mann--Whitney para las comparaciones independientes de HRES2-H2. No afirmes dominancia universal ni equivalencia a partir de un resultado no significativo.

**Prioridad: Alta.** El comentario ya fue revisado y marcado como tal; queda pendiente corregir el manuscrito (la conclusión y el alcance del abstract).

### 2. Falta comparar con sistemas de control adaptativo existentes

**Estado: Aplica.** El manuscrito compara contra metaheurísticas individuales, pero no presenta una comparación controlada con otros controladores adaptativos de parámetros/selección.

**Qué hacer:** incorpora una o más líneas base adaptativas representativas de la literatura, elegidas por cercanía al problema de coordinación (por ejemplo, un control por contador de estancamiento o rotación fija y un método adaptativo publicado). Usa presupuestos de evaluaciones comparables, mismos problemas y semillas cuando sea metodológicamente válido; describe sus parámetros y evita presentar la comparación como justa si sus presupuestos difieren. Si no puedes ejecutar esa comparación, delimita explícitamente la contribución como comparación contra baselines standalone y déjalo como limitación.

**Prioridad: Alta.**

### 3. Cobertura experimental limitada y selección de instancias MKP

**Estado: Aplica en alcance, no en el número literal de dominios.** A diferencia del paper del compañero, el tuyo evalúa tres dominios; sin embargo, selecciona nueve instancias, una por familia Chu–Beasley (ver configuración MKP y análisis estadístico en `paper_final.tex`, aprox. líneas 716–733 y 1364–1365).

**Qué hacer:** si los recursos lo permiten, amplía la evaluación MKP con más instancias por familia/tamaño o con otra colección reconocida. Si no, justifica por qué esas nueve representan los rangos de tamaño y restricciones que quieres estudiar, reporta esa selección como limitación y acota la generalización a esas instancias. No describas nueve instancias como cobertura exhaustiva del MKP.

**Prioridad: Alta.**

### 4. Falta de análisis de sensibilidad de hiperparámetros

**Estado: Aplica.** El manuscrito reconoce que ventana, percentiles y paciencia se fijaron sin un análisis sistemático (conclusiones, aprox. línea 1810). Los valores del comentario del compañero no coinciden con este paper: aquí se informan ventanas W=40/75, banda w=4/7, percentiles 30/70, tolerancia Kmax=15 y paciencia P=8/25 (aprox. líneas 738–755).

**Qué hacer:** no adoptes automáticamente los valores W=200, 40/60, etc. del comentario. Ejecuta un análisis de sensibilidad propio variando al menos W, paciencia, percentiles y tolerancia; conserva el mismo presupuesto y reporta el efecto en calidad, costo y número de cambios de solver. Aclara qué valores se fijaron antes de ver resultados y si los parámetros dependen del horizonte de 1K/3K.

**Prioridad: Alta.**

### 5. Los efectos cambian entre algoritmos

**Estado: Aplica parcialmente.** El paper informa que GA/SA pueden ser competitivos en MKP y que los resultados varían por dominio, pero la explicación de por qué ocurre necesita quedar vinculada a evidencia y no solo a hipótesis.

**Qué hacer:** organiza resultados por algoritmo base y dominio; analiza interacciones con tamaño de población, búsqueda local, reparación y mecanismo de transferencia de élite solo cuando la implementación o los datos lo respalden. Reporta pérdidas y empates, no solo victorias. Si no puedes demostrar causalidad, presenta las explicaciones como hipótesis y plantea pruebas futuras.

**Prioridad: Media-alta.**

---

## Revisor 2

### 1. Añadir la estructura del artículo al final de la introducción

**Estado: Aplica.** No encontré un párrafo de organización del manuscrito al final de la Sección 1.

**Qué hacer:** agrega un párrafo breve que indique el contenido de las secciones siguientes, usando su numeración y títulos definitivos. Actualízalo si reorganizas el paper.

**Prioridad: Baja; corrección directa.**

### 2. Falta una ablación que aísle el aporte de DDTW

**Estado: Aplica.** El propio manuscrito reconoce que no incluye ablación sin monitor ni control de rotación fija (aprox. línea 1810); la discusión también señala que no separa monitor, rotación y memoria élite (aprox. líneas 1795–1798).

**Qué hacer:** diseña controles con el mismo portfolio y presupuesto: sin monitor/sin cambios, rotación fija, monitor DTW y monitor DDTW. Si quieres atribuir el efecto de la memoria élite, agrega una condición sin transferencia o un diseño factorial acotado. Compara por instancia y con pruebas/corrección coherentes. No atribuyas las mejoras específicamente a DDTW si solo evalúas el sistema completo.

**Prioridad: Muy alta.**

### 3. No se justifica que una rampa y una meseta basten como referencias

**Estado: Aplica.** El método define una referencia lineal de progreso y otra constante para estancamiento, pero falta explicar por qué esas dos formas son suficientes frente a trayectorias con progreso escalonado, ruido o retrocesos.

**Qué hacer:** explica el propósito operacional de cada referencia y sus supuestos; justifica cómo se calcula la pendiente y cómo se orienta la señal para maximización/minimización. Discute casos que no representan. Idealmente compara contra una o más referencias/controladores alternativos en una ablación; sin esa evidencia, limita la afirmación a “detector de estos dos patrones”.

**Prioridad: Alta.**

### 4. Faltan métodos adaptativos de referencia

**Estado: Aplica.** Corresponde a la misma brecha del Revisor 1, comentario 2.

**Qué hacer:** selecciona baselines adaptativos comparables y evalúalos con protocolo, presupuesto y datos reproducibles. Si no los agregas, deja claro que el paper prueba competitividad frente a algoritmos standalone, no superioridad frente al estado del arte de control adaptativo.

**Prioridad: Alta.**

### 5. Faltan detalles de reproducibilidad y repositorio

**Estado: Parcial.** El paper documenta entorno, varios parámetros y pseudocódigo, pero la disponibilidad de datos/código aún tiene texto de plantilla y faltan detalles verificables del experimento.

**Qué hacer:** publica un repositorio si tienes autorización y puedes hacerlo; incluye instrucciones de ejecución, dependencias/versiones, semillas, configuración completa por dominio y algoritmo, criterio de presupuesto, archivos de resultados por corrida y scripts de análisis estadístico/figuras. Completa la sección de disponibilidad con enlaces reales. Si no puedes compartir código o datos, dilo expresamente y explica qué material sí estará disponible.

**Prioridad: Alta.**

### 6. “El abstract debe corregirse desde la primera frase”

**Estado: Parcial / comentario poco específico.** La revisión no explica qué está mal. El resumen actual empieza con contexto general y después presenta método y resultados, pero no puedo inferir la objeción exacta del revisor.

**Qué hacer:** revisa que las primeras frases nombren el problema concreto y la brecha, y que el resumen identifique claramente el aporte como un controlador de cambio de solver basado en DTW/DDTW —no como ajuste interno de parámetros si eso no es lo que hace—. Después verifica coherencia de cifras y afirmaciones estadísticas. Si es posible, pide el comentario original en inglés para entender qué significa “desde la primera frase”.

**Prioridad: Media.**

---

## Revisor 3

### 1. Definir la brecha y diferenciar el aporte de control adaptativo e hiperheurísticas

**Estado: Parcial.** La introducción ya afirma que el análisis explícito de la forma de la trayectoria está poco explorado, pero conviene delimitar mejor la novedad frente a control por parámetros, selección de algoritmo e hiperheurísticas.

**Qué hacer:** termina la revisión relacionada con una brecha en 2–3 puntos concretos y enlaza cada punto con una contribución comprobable. Explica que el monitor decide cuándo rotar entre solvers; no afirmes que adapta parámetros internos si solo los cambios de solver están controlados.

**Prioridad: Alta.**

### 2. Fortalecer la literatura sobre control adaptativo, RL, control difuso y trayectorias

**Estado: Parcial.** Hay discusión de aprendizaje por refuerzo y búsqueda guiada por trayectorias, pero no encontré tratamiento explícito del control difuso.

**Qué hacer:** añade referencias primarias recientes y síntesis comparativa de: (a) control adaptativo de parámetros, (b) RL/control difuso aplicados a búsqueda y (c) información de trayectorias. Para cada familia, di qué señal usa, qué decisión controla y cómo difiere del método propuesto. Evita agregar citas sin explicar su relación con la brecha.

**Prioridad: Media-alta.**

### 3. Completar la formulación matemática de DDTW

**Estado: Parcial.** Hay una recurrencia DTW con condiciones de borde y una primera diferencia finita; no se define normalización de la distancia y la condición de borde derivativa es simplificada (aprox. líneas 592–619).

**Qué hacer:** define exactamente el operador derivativo utilizado, índices y fronteras; aclara si el costo final se normaliza por longitud de camino o se deja acumulado. Asegura que la ecuación coincida con el código y que la decisión de no normalizar —si esa es la implementación— esté justificada. Explica la orientación de fitness para maximización/minimización y cómo se construye la referencia derivada.

**Prioridad: Alta.**

### 4. Mostrar trayectorias, distancias, umbrales y estado del controlador

**Estado: Parcial.** Ya hay figuras de convergencia y del indicador Δ para MKP, CEC2022 y HRES2, además de diagnósticos sincronizados de HRES2. No queda claro que se muestren umbrales, distancias por separado, estado y transiciones del controlador en una misma ejecución.

**Qué hacer:** comprueba las figuras existentes y completa al menos una figura diagnóstica legible por modalidad o dominio que sincronice señales y marque los eventos de cambio. No dupliques todas las figuras si no aportan información nueva; puedes usar un caso representativo y explicar que es ilustrativo, no evidencia estadística de las 31 corridas.

**Elementos que pide el revisor:**

- best-so-far en el tiempo;
- referencia de progreso y referencia de meseta;
- distancias a cada referencia y Δ(t) (verifica la notación de `revisores.md`, que escribe DR/DC/Delta ti);
- umbrales y estado del controlador;
- momentos de transición y solver antes/después.

**Prioridad: Media-alta.**

### 5. Justificar los parámetros de exploración/explotación

**Estado: Parcial.** Se describen pools de exploración y explotación y parámetros del monitor, pero falta aclarar de forma sistemática cómo se eligieron todas las configuraciones internas de los solvers y si hubo ajuste por instancia.

**Qué hacer:** crea una tabla por dominio con configuración de cada algoritmo, rango permitido, valor elegido y fuente del valor (literatura, ajuste previo o experimento piloto). Declara cualquier tuning por instancia y evita usar resultados de prueba para presentar el ajuste como general.

**Prioridad: Alta.**

### 6. Incluir pseudocódigo completo

**Estado: Parcial; el manuscrito ya incluye un algoritmo.** El Algoritmo 1 describe el flujo del orquestador, pero hay que verificar que cubra completamente la implementación real y no deje términos sin definir.

**Qué hacer:** valida cada paso contra el código: selección/alternancia de pool y solver, límite de época, actualización del incumbente, reinicio de buffers, arranque de umbrales, presupuesto global y condiciones de parada. Añade la política de inicialización/inyección para solvers de población y trayectoria o referencia a una tabla. No agregues un segundo pseudocódigo si el Algoritmo 1 puede completarse.

**Prioridad: Media-alta.**

### 7. Analizar por qué el efecto varía según algoritmo

**Estado: Aplica parcialmente.** La observación original menciona BDE y BGWO, que no son los algoritmos de este manuscrito; la heterogeneidad entre algoritmos sí aparece en los resultados actuales.

**Qué hacer:** repite el análisis con los algoritmos reales evaluados aquí. Separa resultados por dominio y algoritmo, describe cuándo el híbrido pierde o empata y relaciona las diferencias con mecanismos concretos solo cuando haya evidencia. Si la explicación sobre diversidad, reparación o intensificación es especulativa, etiquétala como hipótesis y propone una medición futura.

**Prioridad: Media-alta.**

### 8. Añadir figuras aclaratorias de la dinámica del controlador

**Estado: Parcial; solapa el comentario 4.** Las curvas de convergencia y Δ ya ayudan, pero revisa si faltan trayectorias de referencia, umbrales, estado y cambios de solver.

**Qué hacer:** usa la lista del comentario 4 como checklist de las figuras. Integra señales sincronizadas, leyenda y anotaciones que permitan al lector comprobar cuándo y por qué se activa la rotación. Indica el número de corrida y el criterio para seleccionar la traza; acompáñala con resultados agregados para no presentar una corrida como representativa de todas.

**Prioridad: Media.**

---

## Acciones adicionales antes del envío

Estos puntos no son comentarios específicos de los revisores, pero aparecen en `paper_final.tex` y pueden perjudicar la presentación:

- Reemplazar el título que todavía dice “Expert Systems with Applications LaTeX template” y eliminar los autores/afiliaciones/contactos de ejemplo que duplican la lista de autores real.
- Reemplazar el texto instructivo de “Acknowledgements” y “Data availability” con declaraciones reales.
- Corregir la síntesis de HRES: las tablas enumeran nueve algoritmos base, mientras que las líneas 1574 y 1612 dicen “all ten”.
- Alinear la descripción estadística general de comparaciones pareadas con la sección HRES, donde las muestras base se tratan como independientes; reporta el diseño por dominio.
- Verificar que el abstract y conclusión usen la misma población de baselines, presupuesto, significancia y métricas que las tablas.

## Orden recomendado de trabajo

1. **Resolver exactitud y consistencia:** título/plantilla, conteo de baselines, emparejamiento estadístico y afirmaciones del abstract.
2. **Decidir si ampliar experimentos:** ablación, sensibilidad y baselines adaptativos son las brechas más importantes; si no se pueden ejecutar, reducir las afirmaciones y explicitar limitaciones.
3. **Cerrar método y reproducibilidad:** parámetros completos, formulación DTW/DDTW acorde al código, pseudocódigo comprobado, semillas/datos/código.
4. **Mejorar presentación:** brecha y literatura, párrafo de estructura, análisis por algoritmo y figuras diagnósticas.

**Nota:** este análisis compara el texto de los comentarios y del manuscrito; no recalcula estadísticas, no valida los datos fuente y no compila el documento.
