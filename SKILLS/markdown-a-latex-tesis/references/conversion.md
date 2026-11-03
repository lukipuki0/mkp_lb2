# Conversión fiel de Markdown a LaTeX

## Revisión previa y selección

Registra fuente actual, fragmento, etapa, raíz LaTeX, destino y decisiones de formato. Lee fuentes extensas por partes para evitar truncarlas. Si una ruta dejó de existir, localiza la fuente actual y confirma cualquier selección ambigua; no reconstruyas el borrador antiguo.

Separa cuerpo académico, rúbrica, discusión y apuntes. Respeta regiones `#ignorar` y registra exclusiones fuera del cuerpo. Las marcas de avance, como ✔️ en un título, se excluyen solo si su función editorial es inequívoca. Conserva símbolos que formen parte del contenido. No elimines comentarios humanos, definiciones o notas por parecer incompatibles con la plantilla.

Antes de escribir, prepara estas correspondencias:

| Elemento | Comprobación |
| --- | --- |
| Encabezados | Nivel del origen, comando LaTeX y tratamiento de niveles profundos. |
| Figuras | Referencia original, archivo resuelto, versión, copia de salida, caption y etiqueta. |
| Tablas | Filas, columnas, notas, archivo, entorno y etiqueta. |
| Bibliografía | Marcador local, fuente identificada y clave global verificada. |
| Referencias | Mención original, destino confirmado y etiqueta única. |

No hace falta crear un archivo de inventario si el reporte de conversión conserva esta información.

## Jerarquía y encabezados

La rúbrica determina las secciones macro y su orden; el autor determina su desglose interno. Conserva títulos, secuencia y relaciones entre bloques. Mapea por posición semántica, no solo por cantidad de `#`. Para una sección seleccionada como capítulo, aplica:

| Markdown | LaTeX |
| --- | --- |
| Título principal | `\chapter` |
| Nivel interno 1 | `\section` |
| Nivel interno 2 | `\subsection` |
| Nivel interno 3 | Rótulo interno en negrita, dentro de la subsección, sin numeración ni entrada propia de índice. |
| Nivel interno 4 | Subrótulo interno en negrita y cursiva, dentro del bloque anterior, sin numeración ni entrada propia de índice. |

La configuración PUCV limita la subdivisión formal a tres niveles: capítulo, sección y subsección. Los niveles adicionales del Markdown se conservan como rótulos de contenido con sus bloques y orden originales, no como nuevas secciones LaTeX. Usa párrafos con `\textbf{Título}` y, para el subrótulo, `\textbf{\emph{Título}}`, sin crear contadores ni alterar `secnumdepth`/`tocdepth`. Mantén diferenciadas las relaciones padre-hijo; no elimines títulos ni fusiones sus contenidos. Registra esta correspondencia en la entrega, sin volver a pedir una decisión de jerarquía cuando el origen es inequívoco. Si aparecen niveles adicionales no contemplados o una contradicción real con la rúbrica, consulta el caso concreto.

No emitas `\ref` a un rótulo como si tuviera un número propio: conserva la referencia por nombre y por su sección contenedora. La presentación final de los rótulos también se revisa en el PDF.

Retira prefijos manuales como "8.1" únicamente si son numeración estructural confirmada. No retires números que formen parte del título, preguntas Q ni identificadores HE, HC o NO. Estos últimos siguen siendo contenido; la negrita no los convierte en encabezados.

El número de punto de rúbrica, el nombre de carpeta y el contador de capítulo son independientes. No fijes contadores para hacerlos coincidir.

## Texto, listas y contenido especial

- Conserva párrafos, énfasis, negritas, listas y anidación con comandos y entornos nativos. No resumas elementos.
- Escapa texto ordinario: `&` → `\&`, `%` → `\%`, `$` → `\$`, `#` → `\#`, `_` → `\_`, `{` → `\{` y `}` → `\}`. Para literales, usa `\textbackslash{}`, `\textasciitilde{}` y `\textasciicircum{}` cuando corresponda. Evita el doble escape.
- Distingue texto de ecuaciones, código y LaTeX existente antes de escapar. Conserva significado y delimitadores matemáticos.
- Usa `\verb` solo donde sea válido y `verbatim` cuando baste. No supongas paquetes de resaltado. En argumentos, captions o celdas, elige una representación compatible sin perder contenido.
- Conserva enlaces y URL. Comprueba que `\url` o `\href` estén disponibles antes de usarlos; no añadas dependencias por suposición.

## Tablas y archivos independientes

Cada tabla va en `avance/seccion-N/tablas/tabla-identificador.tex` y contiene un único entorno completo, caption y etiqueta. El capítulo usa `\input{avance/seccion-N/tablas/tabla-identificador}`, sin envolver el archivo en un segundo entorno de tabla.

Reconstruye encabezados, filas, columnas, alineación, notas y cada celda. Un `<br>` conserva una separación dentro de la celda; una lista conserva todos sus elementos. Usa columnas de párrafo y contenedores compatibles; no introduzcas saltos de fila dentro de una celda sin protección.

### Composición según el contenido

No traslades todas las tablas como una cuadrícula uniforme. Determina qué función cumple cada fila y columna antes de componerla:

- **Registros homogéneos:** conserva las columnas de datos; no inventes agrupaciones a partir de celdas vacías.
- **Guía o cuestionario organizado por temas:** separa el título de cada grupo de las columnas que describen sus preguntas. El título ocupa el ancho completo.
- **Matriz de comparación o respuestas:** conserva las dimensiones de comparación y sus encabezados. No la conviertas en una guía de preguntas ni le impongas columnas ID/Tema.

La disposición visual puede cambiar para expresar esos roles; el contenido y el orden de los registros no. Registra la correspondencia entre las columnas originales y las de salida cuando las reordenes. Si la pertenencia a un grupo no se puede deducir del origen, consulta el caso en lugar de inventarla.

#### Guías con un tema general y preguntas asociadas

La tabla A.2 del Markdown de la sección 8 contiene tres columnas de origen: Tema, ID y Pregunta o instrucción. Las filas con textos como "B. Aclaraciones del contexto durante la entrevista" o "C.1. Concepción de la IR en la organización", sin ID ni pregunta, actúan como encabezados de grupo. No deben ocupar una celda angosta de Tema con las otras dos vacías.

Para este patrón:

1. Conserva íntegro el título del grupo, incluidos sus prefijos B., C.1. o C.2., y hazlo abarcar todas las columnas mediante `\multicolumn`.
2. Debajo del título, presenta **ID | Tema | Pregunta o instrucción**. En este caso, reordena las columnas del origen como ID, Tema y Pregunta, manteniendo juntos los tres campos de cada registro.
3. Incluye después todas las preguntas del grupo en su orden original, hasta el siguiente encabezado temático. No añadas grupos, resumas preguntas ni repitas el título como si fuera una pregunta.
4. Mantén el ancho de la tabla: para títulos largos, usa una celda de párrafo que abarque el ancho calculado de las columnas y sus separaciones. El formato `l` del ejemplo sirve únicamente si el título cabe; no permitas que desborde.
5. Evita que el título de grupo o el encabezado de columnas queden solos al pie de página. En `longtable`, `\\*` evita el salto después de esas filas; comprueba que también quepa la primera pregunta.
6. En tablas multipágina, repite los encabezados de columnas y la indicación de continuación. Comprueba que siga siendo inequívoco a qué grupo pertenece cada pregunta. Repetir un rótulo como continuación no crea un grupo nuevo ni duplica registros.
7. Conserva una sola identidad, caption y etiqueta para la tabla completa, no una tabla numerada distinta por tema.

Fragmento ilustrativo del primer grupo de A.2, dentro de un entorno ya definido con tres columnas de párrafo. No es una tabla autónoma ni sustituye al instrumento completo:

```latex
\multicolumn{3}{l}{\textbf{B. Aclaraciones del contexto durante la entrevista}} \\*
\textbf{ID} & \textbf{Tema} & \textbf{Pregunta o instrucción} \\*
\midrule
Q0a & Madurez del proceso & Si la empresa tiene alguna certificación, ¿cuál es el nivel de madurez de su proceso de desarrollo? Seleccione uno: muy bajo, bajo, medio, alto o muy alto. \\
\midrule
Q0b & Proyecto seleccionado & Aclaraciones sobre el proyecto seleccionado: su dominio, ciclo de vida y otros aspectos. \\
```

La comprobación no se limita a contar celdas: coteja título de grupo, ID, tema y pregunta asociados. Para la A.2 actual deben conservarse los tres grupos y las doce preguntas, sin convertir las tres filas temáticas en preguntas vacías. Vuelve a comprobar esos recuentos contra la fuente seleccionada si cambia el Markdown.

### Separadores según la densidad del cuerpo y los encabezados

Decide los separadores horizontales y verticales por separado. Evalúa las líneas que ocupa el texto **al componer la tabla con sus anchos y tipografía finales**, no la cantidad de líneas del archivo Markdown.

| Cuerpo de la tabla | Encabezado de columnas | Separadores horizontales entre registros | Separadores verticales |
| --- | --- | --- | --- |
| Datos breves, normalmente una línea por fila | Una línea | No; conservar el diseño compacto | No |
| Preguntas o registros textuales de dos líneas o más | Una línea | Sí | No |
| Datos breves, normalmente una línea por fila | Algún nombre ocupa dos líneas o más | No; un encabezado alto no vuelve extenso al cuerpo | Solo en la fila de encabezados |
| Preguntas o registros textuales de dos líneas o más | Algún nombre ocupa dos líneas o más | Sí | Solo en la fila de encabezados |

#### Separación horizontal de preguntas o registros

- En tablas con contenido desarrollado por casilla, separa las preguntas o registros con reglas horizontales. A.1, A.2, A.3 y A.4 son los ejemplos de referencia; el criterio también se aplica a otras tablas textuales semejantes.
- Sigue la tabla de RQ de `avance/problema.tex`: una pregunta por registro y `\midrule` entre registros, conservando el contenido íntegro. No pongas una regla entre las líneas de una misma pregunta o lista de opciones.
- En tablas de datos breves como 3, 4 y 5, conserva el cuerpo sin una regla por fila. No lo conviertas en cuadrícula solo porque un encabezado o un valor aislado se ajuste en dos líneas.
- Mantén las reglas de apertura, separación del encabezado y cierre. No dupliques reglas ya presentes en cambios de grupo o continuaciones multipágina.

#### Separación vertical limitada al encabezado

- Si los nombres de columnas ocupan dos líneas o más (por ejemplo, en 3, A.3 y A.4), delimita las celdas del encabezado con separadores verticales entre columnas. El resto del cuerpo conserva su composición sin líneas verticales.
- No agregues barras `|` a la especificación global de `tabular` o `longtable`, porque se extenderían a todos los registros. Aplica el formato local a las celdas de encabezado mediante `\multicolumn{1}{...}{...}`, manteniendo su ancho de párrafo e incorporando la barra solo en el límite correspondiente.
- Conserva el ancho total al calcular esas celdas: incluye el grosor de las reglas verticales y las separaciones, sin ensanchar el encabezado respecto del cuerpo. No uses columnas `l` sin ajuste de línea para resolver nombres largos.
- Aplica el mismo tratamiento al encabezado repetido de `longtable`. No lo extiendas al título temático que abarca todas las columnas ni a las preguntas.
- Revisa que las reglas y el texto no se superpongan y que no queden líneas verticales prolongadas dentro del cuerpo. La tabla 3 con encabezados altos puede tener estas divisiones verticales sin recibir separadores horizontales entre sus registros breves.

### Política de composición

1. **Tabla breve:** entorno `table` con colocación de referencia `[hbtp]`, caption arriba, label después y `tabular` con `booktabs` si siguen disponibles en el proyecto.
2. **Texto extenso:** columnas `p{...}` con ajuste de línea. Asigna ancho mínimo legible a identificadores y tipos, y distribuye el espacio restante según el contenido de las otras columnas. Incluye separaciones en el cálculo: sin eliminar espacios exteriores ni usar reglas verticales, la suma de anchos más `2n\tabcolsep` debe caber en `\linewidth` para n columnas. Si el encabezado tiene reglas verticales locales, contempla además su grosor sin cambiar el ancho del cuerpo. Ajusta el cálculo si cambias el formato de columnas.
3. **Tabla que excede una página:** utiliza `longtable`; comprueba su disponibilidad e incorpora la dependencia necesaria. Va fuera de `table`; define encabezado inicial y repetido, continuación y cierre. La caption principal y su etiqueta se declaran una sola vez. Una fila individual también debe caber en una página: compruébalo.
4. **Dependencia ausente:** incorpora al preámbulo el paquete mínimo necesario, comprobando que no esté cargado y que sea compatible con el estilo. Obtén la dependencia si falta en el entorno permitido, según las reglas de instalación de esta guía. No sustituyas una tabla multipágina por un flotante gigante. Dividirla en tablas diferentes o cambiar orientación requiere justificación y no debe alterar el contenido ni su orden.
5. **Legibilidad:** no uses `\resizebox` para encoger toda una tabla ni tipografía microscópica para hacerla caber. Revisa anchos, desbordamientos, filas cortadas y encabezados repetidos en el PDF.

No asumas `longtable`, `tabularx`, `makecell` o `pdflscape` cargados. Usa la fuente, separación, estilo de caption y márgenes del proyecto, no los de otra plantilla. Compara todas las celdas con el Markdown después de resolver escapes y saltos.

## Imágenes Markdown, Obsidian y rutas

### Interpretación y resolución

Reconoce imágenes Markdown, incluidas rutas entre ángulos con espacios, y las inserciones Obsidian `![[ruta/figura.svg]]`, `![[figura.svg]]` o con alias/ancho. Separa el nombre del archivo de un alias, tamaño o fragmento; no los interpretes como parte de la ruta ni como un caption.

1. Determina la raíz del repositorio y, cuando corresponda, la del vault. No asumas que coinciden sin comprobarlo.
2. Clasifica la referencia: ruta relativa explícita al Markdown, ruta desde la raíz o nombre sin directorio. Resuelve rutas explícitas sin buscar sustitutos silenciosamente.
3. Para nombres abreviados que no se resuelvan literalmente, indexa los recursos de la sección y del proyecto. Conserva las rutas candidatas completas y distingue recursos finales de pruebas, copias y carpetas temporales. Un resultado en `scratch/` no acredita que sea la figura final.
4. Si varias interpretaciones o candidatos válidos identifican archivos distintos, detente con la lista y pide selección. No elijas el primer resultado ni elimines candidatos para ocultar una ambigüedad. Si no hay coincidencias, reporta el recurso ausente.
5. Normaliza separadores y codificación de espacios, pero verifica el archivo real. No salgas del proyecto ni sigas recursos externos sin autorización.

### Formato y empaquetado

- Prefiere un PDF vectorial existente si corresponde a la misma figura y versión. Un nombre o fecha coincidente no demuestra equivalencia: contrasta contenido, datos, rótulos y procedencia de generación; usa comparación visual cuando esté disponible.
- `\includegraphics` no incorpora SVG directamente con la configuración verificada. Resuelve primero los formatos compatibles ya existentes, sin regenerarlos por rutina. Si solo existe el SVG, obtén una exportación fiel con una herramienta disponible u obtenida dentro del entorno permitido; conserva el original y registra la operación. No habilites shell-escape ni añadas `svg` por comodidad. Una alternativa PNG también necesita equivalencia y legibilidad comprobadas.
- Resuelve e incorpora todas las figuras utilizadas por el fragmento, no solo las primeras o las mencionadas con sintaxis Markdown estándar. Copia únicamente los recursos elegidos a `avance/seccion-N/figuras/`, sin pedir permiso por cada copia. Evita colisiones de nombre; si necesitas un nombre distinto en la copia, registra la correspondencia y conserva el original. Comprueba que cada inserción del Markdown tenga su correspondencia en el `.tex`.
- Emite rutas desde la raíz LaTeX: `\includegraphics[width=\linewidth]{avance/seccion-N/figuras/figura.pdf}`. Ajusta el ancho conservando proporción, sin recortar evidencia. El ancho de Obsidian no se interpreta como centímetros de LaTeX.
- Sigue el patrón observado: entorno `figure[hbtp]`, imagen, caption debajo y label después. Conserva la fuente externa cuando exista. No fuerces `[H]` ni añadas paquetes para fijar posiciones sin una necesidad justificada.
- Título, leyenda y fuente se mantienen como texto externo cuando el origen los proporciona. No inventes un pie desde el nombre del archivo ni modifiques una imagen para extraerle títulos como efecto lateral. Si falta caption y el contexto no permite identificarlo inequívocamente, reporta la decisión pendiente.

## Citas, etiquetas y numeración

Una cita local `[1]` no identifica una entrada global. Coteja la bibliografía del Markdown con `references.bib` por fuente, título y DOI u otro identificador; construye una correspondencia explícita antes de emitir `\cite{clave-verificada}`. No reutilices una entrada solo por compartir apellido.

Acopla todas las fuentes citadas o enumeradas en la bibliografía local del fragmento a la bibliografía compartida durante la conversión, sin una autorización adicional por cada entrada:

1. Busca coincidencias por DOI normalizado u otro identificador estable; contrasta también título y autoría. Si no hay identificador, verifica los demás metadatos y la URL de la fuente. No confundas la misma obra con otra edición o versión.
2. Reutiliza la clave existente cuando corresponda a la misma fuente. Conserva las entradas y claves ajenas, y no reescribas ni ordenes todo el `.bib` como efecto lateral.
3. Si la fuente no existe, verifica sus metadatos en el material disponible o en una fuente original y añade una entrada BibLaTeX válida con clave nueva y única. No inventes fechas, DOI, autores ni campos para completar una ficha.
4. Sustituye los marcadores locales por las claves verificadas y comprueba citas múltiples, tanto en texto como en captions y celdas. Biber y el estilo IEEE determinan la numeración global; no conserves artificialmente los números locales.
5. Integra la información de la lista bibliográfica local en el `.bib`, no como una bibliografía manual paralela. Registra las entradas reutilizadas, añadidas y las referencias de la lista que no se citen en el fragmento; no fuerces `\nocite{*}`.

Si no se puede verificar una fuente o hay entradas incompatibles para la misma obra, informa la evidencia faltante o el conflicto. Solo en un fragmento declarado incompleto se conserva el marcador como texto: ese resultado no está listo para entrega.

Para figuras, tablas y secciones, usa prefijos estables y específicos de sección, como `fig:captura-desafios-elicitacion`, `tab:captura-hallazgos` o `sec:captura-entrevistas`. Verifica destinos y unicidad en el proyecto completo.

Los números originales "6a" o "9s" se conservan en la correspondencia de conversión, no como contadores forzados. Si son figuras individuales, dales etiquetas individuales; no supongas subfiguras ni añadas `subcaption`. Sustituye las menciones del cuerpo y de las celdas por `\ref` solo con destino confirmado. El caption no repite manualmente "Figura 6a" o "Tabla 9". Informa la numeración automática resultante y las referencias sin destino.

## Integración y comprobación

Mantén clase, estilo, tipografía, márgenes y configuración. No añadas `babel`, macros ni numeración preliminar por suposición. Biber es el backend bibliográfico, no determina el motor.

La incorporación sugerida del capítulo se expresa desde la raíz, por ejemplo `\include{avance/seccion-8/seccion-8}`. Si el proyecto usa otro mecanismo, conserva su patrón. Activa esa inclusión solo si se pidió integrar el capítulo. Si la conversión requiere ajustes mínimos de dependencias en el preámbulo, aplícalos y repórtalos por separado; no dupliques inclusiones ni alteres otros capítulos.

### Dependencias necesarias

Comprobar e incorporar bibliografía y paquetes necesarios forma parte de convertir la sección. Para dependencias, primero revisa el preámbulo, los archivos cargados y las herramientas instaladas. No descargues otra copia si la existente sirve.

Añade únicamente lo necesario para el contenido real (por ejemplo, `longtable` para tablas multipágina), sin cambiar clase, tipografía ni estilo PUCV. Cuando falte un paquete o herramienta, utiliza fuentes oficiales como CTAN o la distribución LaTeX, con el gestor o un destino local permitido. Comprueba procedencia y disponibilidad después de obtenerlo, y registra el paquete, motivo y cambio de configuración.

La necesidad de una dependencia no autoriza instalaciones globales, privilegios, cambios fuera del proyecto ni habilitar ejecución insegura. Si la operación exige un destino externo o un permiso no concedido, informa la ruta y solicita ese permiso específico. No presentes una descarga fallida como dependencia resuelta ni omitas contenido para evitarla.

Comprueba antes de entregar:

- Conservación de párrafos, listas, ecuaciones, código, preguntas, definiciones, identificadores y todas las celdas.
- Correspondencia de todos los encabezados y registro de exclusiones, sin eliminar contenido académico.
- Recursos resueltos, versiones cotejadas, copias presentes y rutas comprobadas desde la raíz.
- Etiquetas únicas, inclusiones únicas, claves bibliográficas reales y ninguna referencia aparentemente resuelta con un destino inventado.
- Compilación con el motor confirmado y ciclo bibliográfico completo, si está autorizado. Revisa errores, citas/referencias indefinidas y avisos de cajas desbordadas; un código de salida cero no basta.
- Inspección visual de páginas, títulos, tablas multipágina, captions, proporción y legibilidad de figuras. En tablas, comprueba también separadores horizontales en registros textuales, cuerpo compacto en datos breves y líneas verticales restringidas al encabezado cuando corresponda.

Registra comandos, resultados y pendientes fuera del cuerpo académico. Sin compilación, declara "compilación no comprobada"; sin inspección visual, no afirmes maquetación correcta. La revisión de estas instrucciones no sustituye una conversión real y su validación.
