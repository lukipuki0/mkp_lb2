# Figuras académicas para MASRE

[SKILL.md](SKILL.md) define el contrato de creación y revisión de figuras de apoyo. El paquete es independiente: no requiere otras skills. No sustituye la redacción académica ni decide alcance, metodología o arquitectura.

## Uso

Indica al agente el informe destino, el fragmento fuente y qué debe comprender el lector. Puede crear procesos/metodología, arquitectura conceptual o lógica, flujos, secuencias y comparaciones respaldadas. No necesita una figura real para probar el generador.

El contenido vigente del proyecto está en `docs/informe_avance/`; `docs/informe_avance/Escrito.md` enlaza el informe formal en Overleaf. No se presupone un informe LaTeX local. Elige la ubicación de figuras junto a los artefactos del informe solicitado, fuera de esta skill. Si no está resuelto el destino, acláralo antes de escribir entregables.

## Qué gráfico conviene

La [guía de selección de gráficos y diagramas](references/seleccion-de-graficos.md) relaciona preguntas, estructuras de datos y representaciones: comparación, evolución, distribución, relación, composición, incertidumbre, evidencia cualitativa y procesos. Incluye cuándo evitar cada opción, ejemplos y fuentes de consulta. La skill debe consultarla antes de generar o sugerir una figura y puede recomendar una tabla si resulta más útil.

### Infografías locales como referencia

El [catálogo operativo](references/catalogo-infografias.md) convierte las ocho páginas del [PDF aportado](references/Infografias_graficos_estadisticos.pdf) y la [imagen complementaria](references/graficos-visualizacion-datos_1.jpg) en criterios textuales: variantes, cuándo usarlas y errores que no deben copiarse. Ambos originales se conservan. La skill debe leer este catálogo y la guía general en cada generación, adaptación o recomendación; abrir los originales solo cuando haga falta revisar detalles visuales. No es una plantilla estética obligatoria ni evidencia de la tesis.

## Herramienta según la figura

- **Diagramas conceptuales, arquitectura y flujos:** generador propio `scripts/dgen.py` (API `D`), Python estándar y formato editable draw.io. No usa Matplotlib ni Plotly.
- **Gráficos estáticos de datos para el informe:** Matplotlib, si está disponible; conserva script y referencia a los datos para reproducir la figura.
- **Gráficos interactivos:** Plotly cuando la interactividad sea necesaria y esté disponible; comprueba aparte cualquier exportación estática para el informe.

Consulta la [selección de herramienta](references/flujo.md#elección-de-herramienta). Busca claridad y legibilidad, no decoración: no inventes datos ni distorsiones escalas para mejorar el aspecto. No se incluyen generadores específicos de Matplotlib o Plotly ni se instalan dependencias automáticamente. La prueba y el render descritos a continuación son del generador draw.io.

## Prueba local sin dependencias externas

Desde la raíz del repositorio, con Python 3 disponible:

```bash
PYTHONDONTWRITEBYTECODE=1 python .agents/skills/figuras-academicas/scripts/ejemplo.py RUTA_TEMPORAL_AISLADA
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s .agents/skills/figuras-academicas/tests -v
```

Sustituye el marcador por una carpeta temporal de sesión, nunca por el directorio de la skill. Conserva cualquier caché preexistente. La muestra ficticia genera exactamente:

- `Muestra-Diagrama-v3.drawio`: tres páginas (completa, p1, p2).
- `page-full.drawio`, `page-p1.drawio`, `page-p2.drawio`: páginas individuales.
- `meta.json`: `name`, `title`, `parts`, `extent`, `windows`.

Los nombres y el título «Muestra técnica · versión 3» son compatibilidad de la demostración, no una convención para nuevas figuras. No genera PNG ni PDF. Los tests comprueban contratos XML/metadatos, páginas, atenuación e iconos; no legibilidad visual.

## Render opcional

Consulta el [flujo local](references/flujo.md). El renderizador requiere Pillow, Playwright, Chromium compatible y `viewer-static.min.js` local revisado; no se instalan ni descargan automáticamente. Si faltan, entrega XML como borrador con revisión visual pendiente. No ejecutes código de red para suplirlos.

`build.py` ejecuta el Python local de la figura: úsalo solo con fuentes locales confiables y revisadas, nunca como sandbox. Escribe bajo `out/<nombre-del-script>/` del directorio de trabajo. Trabaja en scratch para pruebas y copia únicamente entregables revisados al destino elegido. Produce PNG, no PDF; PDF requiere exportación separada disponible/autorizada.

## Diseño y entrega

Usa formas neutras antes que iconos, fuentes disponibles y contraste suficiente. La paleta, tamaños y tipografía del código importado son valores técnicos heredados, no identidad obligatoria de MASRE. El XML conserva metadatos históricos `agent="OneByte"` y versión del formato por compatibilidad; no expresan autoría de la figura ni reglas de diseño. Revisa cada exportación al tamaño de inserción real.

Entrega solo el gráfico o diagrama, con título opcional, etiquetas, ejes y leyendas necesarios; sin subtítulos, citas, fuentes, notas explicativas ni acotaciones de alcance incrustadas por defecto. Solo añade anotaciones internas por petición explícita o semántica indispensable. Deja pie sugerido, atribución verificable, muestra, supuestos, limitaciones y consejos en un README externo; el autor los incorporará al escrito, sin inserción automática. Incluye `.drawio` para diagramas o script y referencia a los datos para gráficos de datos, junto con las exportaciones efectivamente generadas. Declara pendientes. No publiques ni instales nada como efecto lateral.

## Procedencia y mantenimiento

Adaptación del paquete `diagramas-drawio-onebyte` aportado por el usuario, conservado en este directorio renombrado. El material aportado no especifica licencia ni autoría verificable; no se atribuye una licencia nueva. La muestra completa y ficticia de la integración sustituía un ejemplo original incompleto; no representa MASRE.

Se mantienen la API y los contratos existentes del generador y renderizador. Los recursos de proveedores conservados son opcionales, no decisiones tecnológicas. La actualización del índice de skills del runtime queda a cargo del orquestador; este cambio no modifica registros externos al paquete.
