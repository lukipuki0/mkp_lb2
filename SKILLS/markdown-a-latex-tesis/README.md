# Markdown a LaTeX para la tesis MASRE

Guía para convertir contenido académico seleccionado al formato PUCV, conservando el texto y su desglose. Organiza la salida por sección, con tablas y figuras propias, sin duplicar el proyecto LaTeX. La conversión es asistida; no incluye un conversor automático.

## Uso

Ejemplo de pedido:

> Convierte el Markdown de la sección 8 para Seminario final. Entrega el paquete en avance/seccion-8/, con tablas y figuras internas. Conserva el contenido y el formato PUCV. No integres todavía en main.tex.

1. Cargar [SKILL.md](SKILL.md).
2. Consultar [proyecto y rúbrica](references/proyecto-y-rubrica.md) y [reglas de conversión](references/conversion.md).
3. Localizar la fuente actual y verificar jerarquía, recursos, citas y dependencias antes de escribir.
4. Resolver la bibliografía, incorporar figuras y dependencias necesarias, y preparar el paquete. Consultar solo ambigüedades reales o permisos específicos pendientes.
5. Comparar contenido, comprobar incorporación y, cuando corresponda, compilar e inspeccionar el PDF.

El paquete contiene un archivo principal `avance/seccion-N/seccion-N.tex`, tablas independientes en `tablas/` y figuras utilizadas en `figuras/`. Se crean solo carpetas necesarias. Las rutas se expresan desde la raíz de `main.tex`; bibliografía, estilo y configuración siguen compartidos.

## Qué debe resolver antes del traspaso

- Orden macro según la rúbrica y desglose interno del autor: tres niveles formales; títulos más profundos como rótulos internos diferenciados, sin perder contenido ni orden.
- Recursos Markdown y Obsidian con rutas relativas, desde raíz o nombres abreviados, sin elegir entre duplicados por su orden de aparición.
- Figuras en un formato compatible y de la versión correcta; un PDF homónimo no acredita equivalencia con un SVG.
- Tablas compuestas según su contexto, no como cuadrículas uniformes: distinguir grupos temáticos, encabezados y registros; además, resolver anchos y continuidad multipágina.
- Correspondencia de citas con claves reales y de referencias con etiquetas únicas, sin fijar contadores.

Para la sección 8, el diagnóstico registrado en [proyecto y rúbrica](references/proyecto-y-rubrica.md) identifica los elementos que deben comprobarse al convertir. Se resuelven durante el trabajo, no solicitando aprobación por cada recurso, entrada bibliográfica o paquete.

## Comportamiento durante la conversión

- Reutiliza las claves de fuentes ya presentes en el `.bib` y añade las fuentes verificadas que falten, sin duplicar la bibliografía ni inventar datos.
- Sigue el orden de la rúbrica y conserva títulos y bloques internos del autor. Los niveles profundos no crean una numeración adicional incompatible con la plantilla.
- En guías agrupadas como la tabla A.2, el tema general abarca todo el ancho; debajo se presentan ID, Tema y Pregunta o instrucción, con todas las preguntas asociadas. El título de grupo no se trata como una fila de datos incompleta. Otras clases de tablas conservan la estructura que requieren sus contenidos.
- Separa horizontalmente las preguntas o registros con texto de varias líneas (A.1–A.4, como la tabla de RQ). Conserva el cuerpo sin reglas por fila en tablas compactas como 3–5. Si los nombres de columnas ocupan varias líneas, añade separadores verticales únicamente en el encabezado (por ejemplo, 3, A.3 y A.4), sin extenderlos a los datos.
- Resuelve y copia todas las figuras del fragmento a su paquete, manteniendo captions y referencias, sin regenerar recursos existentes por rutina.
- Añade o descarga las dependencias necesarias desde fuentes oficiales dentro del entorno permitido; registra los cambios. Las instalaciones externas o privilegiadas siguen requiriendo permiso específico.

La inclusión del capítulo en `main.tex` es independiente de esos preparativos: si no se pidió integración, entrega la inclusión sugerida sin activarla.

## Entrega y límites

Entrega el paquete, la inclusión realizada o sugerida, la correspondencia con el origen y el reporte de exclusiones, dependencias y verificaciones. No sobrescribe el Markdown ni modifica otras partes del informe como efecto lateral. No publica en Overleaf.

Una pasada de conversión requiere que los insumos y las decisiones previas estén resueltos. Distingue contenido cotejado, compilación ejecutada y maquetación inspeccionada. No promete un PDF perfecto sin esas comprobaciones ni cumplimiento académico por la mera presencia de secciones.

## Procedencia

La guía adaptó conceptualmente prácticas de Informe 3: separar fuente y salida, verificar fidelidad y documentar pendientes. Los archivos de aquella skill no están presentes en este proyecto y no son dependencias. No hereda macros OneByte, numeración manual, formularios, APA ni configuración de su conversor.

La estructura se fundamenta en la copia local PUCV y la rúbrica enlazadas. No acredita sincronización con Overleaf. El motor remoto sigue por confirmar. La gestión de fuentes bibliográficas y dependencias necesarias forma parte de la conversión; la integración del capítulo, publicación e instalaciones fuera del entorno permitido no se presumen.
