# Flujo de generación y render

## Ubicación y preparación

Selecciona informe, fuentes y espacio final antes de generar figuras reales. Guarda los entregables junto a los artefactos/figuras de ese informe, nunca en este paquete. No presupongas un árbol LaTeX local ni crees carpetas vacías. Para la muestra y verificaciones, usa scratch temporal de sesión y `PYTHONDONTWRITEBYTECODE=1`.

La generación XML requiere solo Python estándar. El render es opcional y requiere herramientas ya disponibles y autorizadas. No instales, descargues, ejecutes código de red ni publiques automáticamente. Si falta alguna herramienta, informa cuál y conserva el editable como borrador.

Antes de generar, separa el cuerpo visual (gráfico/diagrama y título opcional) del pie externo. Conserva ejes, unidades y leyendas necesarios; deja subtítulos, fuentes/citas, muestra, alcance, supuestos y límites en el README. No insertes el pie automáticamente en el escrito. Las anotaciones internas adicionales requieren petición explícita o semántica indispensable. Al retirar notas, reajusta la geometría para no dejar bandas vacías y comprueba recortes.

## Elección de herramienta

Primero elige el tipo de figura con la [guía de selección](seleccion-de-graficos.md); la pregunta y los datos determinan la representación, y después se selecciona la librería.

El generador local no usa Matplotlib ni Plotly: genera XML draw.io con Python estándar. No lo fuerces para representar series numéricas.

| Necesidad | Herramienta adecuada | Entrega |
| --- | --- | --- |
| Arquitectura, procesos, flujos, secuencias o relaciones conceptuales | API local `D` / draw.io | `.drawio` editable y exportaciones disponibles |
| Barras, líneas, dispersión, histogramas u otras distribuciones de datos para el informe | Matplotlib, si está disponible | Script, referencia a datos de entrada y SVG/PDF o PNG efectivamente generado |
| Exploración con zoom, filtros o información al pasar el cursor | Plotly, si está disponible y se necesita interactividad | Script y HTML; para el informe, exportación estática comprobada por separado |

Matplotlib es la opción por defecto para gráficos estáticos destinados al informe o a Overleaf; Plotly no mejora por sí solo la estética ni garantiza exportación estática. Comprueba las dependencias locales de la ruta elegida; no instales ni descargues nada automáticamente. Si falta una dependencia, informa el bloqueo y ofrece una alternativa disponible sin cambiar la semántica del gráfico.

Para gráficos de datos, conserva el origen y las transformaciones en el script o README externo; no inventes valores, agregaciones, incertidumbre ni resultados. Elige la representación por la pregunta: comparación, evolución, relación o distribución. Prioriza tipografía legible al tamaño final, alineación, espacio útil y paleta accesible; evita 3D decorativo, efectos y ejes o escalas que distorsionen la comparación. Mantén ejes, unidades y leyendas necesarios, y verifica que la exportación conserve etiquetas, símbolos y proporciones.

Las instrucciones de API y render siguientes corresponden únicamente a draw.io; no son requisitos para Matplotlib o Plotly.

## API conservada

[scripts/dgen.py](../scripts/dgen.py) ofrece `D(name, title)`, `P`, `vertex`, `img`, `iw`, `icons_row`, `box`, `container`, `badge`, `num`, `text`, `edge`, `extent`, `bbox` y `save`. Usa la API existente sin crear otro generador.

- `D('slug-semantico', 'Título')` define nombre y título; no exige una versión en el nombre.
- `hdr` es la parte inicial para el encabezado; `P('base')` marca elementos siempre visibles. `P('p1', 'Etiqueta')` registra una parte, sin duplicar claves.
- Cada celda pertenece a la parte activa. `save(OUT, windows={'p1': 'with-base'})` genera multipágina, páginas individuales y metadatos. Una parte produce solo página completa; dos o más producen páginas parciales con las otras partes atenuadas al 16 %. El encabezado `hdr` aparece solo en la completa; `base` permanece visible.
- `windows` acepta rectángulos `(x0, y0, x1, y1)` o `with-base`; por defecto usa la caja envolvente de la parte. Incluye al menos un vértice por parte. El generador no valida entradas vacías ni está diseñado para entradas no confiables.
- Ajusta geometría y texto al tamaño final. Los tamaños, colores, tipografía y dimensiones de página predefinidos son valores heredados, no requisitos académicos. El estilo `FONT` heredado declara una fuente web: no presupongas que está instalada ni autorices su acceso externo; usa una fuente local disponible y estilo sin recurso remoto para trabajo sin red.
- Usa `ex/ey/nx/ny` y `pts` para rutas. Si las etiquetas de secuencia colisionan, deja `label` vacío y coloca `text` junto a la línea sin ocultar flechas.

[scripts/ejemplo.py](../scripts/ejemplo.py) es una muestra ficticia ejecutable; consulta sus salidas exactas en el [README](../README.md). Para una figura real, importa `D` desde `dgen`, define un `generate(out)` o usa `OUT` inyectado por el renderizador y llama `save`. El directorio de scripts debe estar en la ruta de importación, por ejemplo mediante `PYTHONPATH` apuntando al `scripts` de este paquete.

## Iconos opcionales

[scripts/icons.py](../scripts/icons.py) conserva alias de productos del paquete importado. `style(alias)` devuelve un estilo nativo y falla con alias desconocido. No uses el catálogo como propuesta de stack. Prefiere cajas neutras; usa un icono solo si aporta significado respaldado.

`IC` apunta a `ic/` del directorio de trabajo; `img` incrusta un SVG local o acepta una etiqueta `tag:Nombre:#color`. No necesita descargas. Para material externo ya autorizado, comprueba procedencia, licencia y ausencia de scripts/recursos externos en SVG; registra esos datos junto al recurso y revisa su aspecto. Ante un símbolo desconocido, basta una etiqueta textual, no un logo improvisado.

## Render opcional y límites

[scripts/build.py](../scripts/build.py) ejecuta el script Python local indicado con `runpy`; úsalo solo con fuentes locales confiables y revisadas. No es un sandbox. Desde el directorio de trabajo aislado, con rutas resueltas, la invocación es `python RUTA_AL_PAQUETE/scripts/build.py RUTA_A_LA_FIGURA.py`.

Requiere Pillow, Playwright, Chromium compatible y `viewer-static.min.js` local revisado en el directorio de trabajo. No incluye esos recursos. Verifica su disponibilidad antes de invocarlo y evita accesos externos no autorizados del visor, fuentes o SVG. La presencia de los módulos por sí sola no acredita disponibilidad del navegador ni del visor.

Escribe en `out/<nombre-del-script>/`, espera el SVG del visor y captura a escala 2: `r-full.png`, y `r-<k>.png` si hay varias partes. Pillow genera `full.png`, `part-<k>.png` (margen de 22 px) y `map.png`; conserva también `render.html` como intermediario. No produce PDF. El mapa se genera incluso sin partes parciales; su existencia no obliga a entregarlo.

Los recortes presuponen escala 2 y anclas de origen; no garantizan equivalencia entre versiones del visor. Ajusta ventanas si cortan elementos. Para partes no rectangulares, una eventual adaptación manual del mapa debe conservar correspondencia con los recortes; no es una función automatizada. Sigue la [revisión visual](revision-y-diapositivas.md) antes de considerar final una exportación.
