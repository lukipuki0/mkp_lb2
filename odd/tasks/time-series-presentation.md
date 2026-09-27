# Presentación: series temporales en la optimización

## Objetivo y problema

Reorientar la presentación para que la contribución central sea usar la trayectoria temporal del mejor valor objetivo para coordinar metaheurísticas. La explicación actual dedica demasiado espacio principal al cálculo interno de DTW y no muestra con claridad la ventana de comparación sobre una curva de convergencia.

## Alcance y restricciones

- Editar `Presentación_luka/main.tex` y compilar `Presentación_luka/main.pdf`.
- Conservar el estilo Beamer de PUCV y las derivaciones académicas útiles en el apéndice.
- La gráfica nueva es esquemática, editable en PGFPlots y sin cifras experimentales inventadas.
- Dos límites verticales identifican exactamente la ventana reciente.
- No realizar operaciones remotas ni crear un commit desde esta tarea delegada.

## Ruta y verificación

Ruta: trabajo directo delegado. El cambio requiere reorganizar varias diapositivas, redactar el argumento y compilar/inspeccionar el PDF. TDD: no aplica a una presentación; verificación mediante compilación LaTeX, extracción de texto e inspección visual de las diapositivas modificadas.

## Tareas

- [x] TSP-1 Reforzar problema, objetivo y contribución alrededor de las series temporales de convergencia.
- [x] TSP-2 Añadir curva esquemática del mejor LCOE con rampa ideal, meseta y dos límites de ventana; simplificar DTW en el cuerpo principal y trasladar el detalle técnico al apéndice.
- [x] TSP-3 Compilar y verificar legibilidad, orden y exactitud de la ventana en el PDF.

## Criterios de aceptación

La presentación deja explícito qué serie se observa y para qué decisión se usa. La diapositiva de convergencia distingue observación, rampa y meseta sin presentar datos ficticios como resultados. DTW se entiende antes de ver su recurrencia. El PDF compila sin errores y las diapositivas afectadas no contienen cortes ni superposiciones.

## Progreso y evidencia

- TSP-1: portada, brecha, objetivo, posicionamiento y cierre reorientados a la serie temporal. Ruta delegada por análisis, edición y verificación coordinados de varias diapositivas.
- TSP-2: diapositiva 11 con curva PGFPlots esquemática, rampa y meseta, delimitada por exactamente dos líneas verticales; diapositiva 12 con explicación de DTW en lenguaje simple. Recurrencia, ejemplo numérico y discriminante conservados en diapositivas 30–32 del apéndice. La diapositiva 11 aclara que el detector recibe `-LCOE` aunque la gráfica de LCOE desciende.
- TSP-3: compilación con `pdflatex -jobname=main_rebuild` (tres pasadas iniciales y dos finales) y `biber main_rebuild`: salida final de 36 páginas, sin errores fatales. Verificación de texto mediante PyMuPDF y visual de las páginas 1, 7, 9, 11, 12, 27 y 29–32. La compilación directa con `main.aux` preexistente falló; el nombre de trabajo limpio evitó el auxiliar desactualizado. Persisten avisos de sobrecaja anteriores en otras diapositivas (páginas 6 y 18), fuera del alcance de este cambio.

Espejo Engram: sincronizado. Commit: pendiente, lo realizará el agente principal.

## Siguiente paso

Revisión final del agente principal y commit convencional de la unidad de trabajo.
