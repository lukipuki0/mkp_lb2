---
name: markdown-a-latex-tesis
description: "Trigger: pasar Markdown a LaTeX, convertir MD a TEX, preparar capítulos para Overleaf o formatear el informe de tesis MASRE. Conserva contenido y jerarquía, organiza cada sección en un paquete con tablas y figuras, y verifica compatibilidad con la plantilla PUCV existente."
---

## Contrato de activación

Convierte contenido académico seleccionado de Markdown a LaTeX para MASRE. Trabaja sobre formato y estructura documental, no sobre redacción, metodología ni resultados. Funciona sin depender de otras skills.

Lee [proyecto y rúbrica](references/proyecto-y-rubrica.md), [conversión](references/conversion.md) y el [ejemplo de uso](README.md). La copia local de referencia está en `informes/avance_seminario/archivos_latex/Informe-avance-tesis/`. Verifica las rutas actuales: una copia archivada no acredita sincronización con Overleaf.

## Límites

- Conserva ideas, cifras, citas, orden, jerarquía, preguntas, identificadores y todas las celdas del origen. No resumas ni conviertas propuestas en hechos. Reporta problemas de contenido fuera del cuerpo académico.
- Compón cada tabla según su estructura semántica: distingue encabezados de grupo, encabezados de columnas y registros. Una fila temática no es un registro incompleto. Puedes adaptar la disposición de columnas para expresar sus roles, sin alterar el orden de las preguntas ni su pertenencia a cada grupo. Aplica el caso de guía agrupada de las reglas de conversión; no impongas el mismo diseño a todas las tablas.
- En tablas textuales cuyas preguntas o registros ocupan dos líneas o más, añade separadores horizontales entre registros, como en la tabla de RQ y las tablas A.1–A.4. Conserva el cuerpo compacto, sin reglas entre cada fila, en tablas de datos breves como 3–5. Si los nombres de columnas ocupan dos líneas o más, añade separadores verticales únicamente en la fila de encabezados, también al repetirla en otra página. Son decisiones independientes; no dibujes una cuadrícula en todo el cuerpo.
- La selección del usuario manda, incluso si corresponde a un borrador de `informes/final_seminario/`. No sustituyas fuentes movidas por copias antiguas ni recrees rutas eliminadas. `docs/informe_avance/` sigue siendo la autoridad académica general.
- Respeta clase `report`, estilo PUCV, bibliografía IEEE/Biber y configuración existente. No importes convenciones OneByte, APA, macros ajenas ni un conversor de otra tesis.
- No modifiques Markdown, `default/`, portadas, estilo PUCV ni glosario como efecto lateral. Durante una conversión, acopla las referencias al `.bib` compartido (reutiliza las existentes y añade las verificadas que falten) e incorpora las dependencias mínimas necesarias. Son parte del trabajo, no consultas rutinarias. No alteres entradas bibliográficas ajenas ni añadas paquetes por conveniencia.
- Obtén o descarga dependencias necesarias desde fuentes oficiales dentro del entorno permitido; registra qué se añadió y por qué. No cambies de motor, habilites shell-escape ni instales globalmente como solución automática. Si una instalación requiere salir del proyecto, privilegios o permisos no concedidos, indica destino y alcance y solicita el permiso específico.
- No publiques ni sincronices con Overleaf, ni hagas staging o commits sin solicitud expresa. No sobrescribas destinos ajenos; inspecciónalos antes de escribir.
- No conviertas rúbricas, comentarios operativos o regiones `#ignorar` en contenido académico. Registra exclusiones inequívocas; consulta si sus límites son ambiguos.
- Una conversión asistida en una pasada necesita insumos y decisiones resueltos antes de escribir. No es un conversor automático ni una garantía de maquetación perfecta sin compilar e inspeccionar el PDF.

## Organización de salida

Cada sección seleccionada se entrega como un paquete dentro del proyecto LaTeX:

```text
avance/
└── seccion-8/
    ├── seccion-8.tex
    ├── tablas/
    │   ├── tabla-09.tex
    │   └── tabla-10.tex
    └── figuras/
        └── desafios-elicitacion.pdf
```

El ejemplo muestra solo parte de los recursos; no limita cuántas tablas o figuras contiene la sección. Usa identificadores de origen estables, sin imponer su número como contador LaTeX. Si el destino ya existe, conserva su organización salvo cambio autorizado. Crea únicamente carpetas con contenido real.

- El archivo principal (por ejemplo, `avance/seccion-8/seccion-8.tex`) contiene el capítulo, texto y referencias a sus recursos.
- Cada tabla se entrega en un archivo propio con su entorno completo, caption y etiqueta; se incorpora una sola vez con `\input`.
- Copia a `figuras/` solo recursos utilizados y verificados. Conserva los originales y registra cualquier cambio de nombre en la copia.
- Bibliografía, estilo, portadas y configuración siguen compartidos en la raíz. No crees una bibliografía ni un proyecto autónomo por sección.
- Todas las rutas de `\input` y `\includegraphics` se escriben desde la raíz que contiene `main.tex`, no desde el archivo incluido. Ejemplos: `\input{avance/seccion-8/tablas/tabla-09}` y `\includegraphics[width=\linewidth]{avance/seccion-8/figuras/desafios-elicitacion.pdf}`.
- Si no se pidió integración, entrega la inclusión sugerida del capítulo sin activarla en `main.tex`. La bibliografía compartida y los ajustes mínimos de dependencias se gestionan durante la conversión; distingue esos cambios de la inclusión del capítulo.

## Procedimiento

1. Localiza la fuente actual, el fragmento, la etapa y la raíz LaTeX. Inspecciona destino, `main.tex`, estilo, bibliografía y ejemplos reales. No asumas que la ruta de una conversación anterior sigue vigente.
2. Haz una revisión previa: inventario de encabezados, párrafos, listas, tablas, figuras, citas y referencias. Resuelve las imágenes Markdown y Obsidian con las reglas de conversión.
3. Aplica la rúbrica al orden macro y conserva el desglose interno del autor con la correspondencia de encabezados descrita en conversión. Resuelve las claves bibliográficas y la composición de tablas: identifica sus roles y agrupaciones antes de elegir columnas, fusiones y saltos; incorpora lo necesario. Consulta únicamente selecciones ambiguas, datos no verificables o permisos específicos que falten, no operaciones normales de la conversión.
4. Contrasta las secciones macro con la rúbrica de la etapa sin rellenar capítulos ni alterar el desglose del autor.
5. Convierte al paquete de sección con las reglas de conversión. Mantén una correspondencia entre identificadores de origen, archivos, etiquetas y numeración automática.
6. Integra solo lo solicitado: acopla las fuentes al `.bib` compartido y añade las dependencias necesarias sin modificar otros capítulos. Si no se pidió incluir el capítulo en `main.tex`, entrega su inclusión sugerida; los ajustes mínimos del preámbulo se identifican por separado.
7. Compara contenido y recursos con el origen, incluidas las preguntas y su asociación con los temas generales de cada tabla; verifica rutas desde la raíz, etiquetas y claves en el proyecto completo. Un capítulo sin errores locales puede introducir etiquetas duplicadas o referencias incorrectas en el informe.
8. Compila si está dentro del alcance y el motor está confirmado, preferentemente en una copia aislada dentro del proyecto. Verifica el ciclo bibliográfico y las referencias; después inspecciona visualmente tablas y figuras.
9. Entrega archivos, instrucciones de incorporación, decisiones aplicadas, exclusiones y resultados observados. Separa fidelidad, compilación, maquetación y cobertura de rúbrica.

## Criterio de entrega

- **Conversión preparada:** contenido y recursos cotejados; se declaran dependencias, citas o decisiones que impiden integrar.
- **Compilación comprobada:** se ejecutó el motor indicado y el ciclo bibliográfico, sin errores ni citas o referencias indefinidas.
- **Maquetación comprobada:** además se inspeccionó el PDF y se revisaron encabezados, tablas, figuras, pies y saltos.

No intercambies estos estados. Si falta una cita o recurso, el resultado no está listo para entrega aunque compile. Informa comandos, resultado y limitaciones; no atribuyas pruebas de conversión a una simple revisión de esta skill.

## Referencias

- [Proyecto PUCV y rúbrica por etapa](references/proyecto-y-rubrica.md).
- [Conversión fiel y casos especiales](references/conversion.md).
- [Uso y límites](README.md).
