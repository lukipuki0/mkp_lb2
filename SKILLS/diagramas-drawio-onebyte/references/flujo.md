# Flujo de generación y render

## Preparación e iconos

Usa el entorno aislado y autorizado del [README](../README.md). Ejecuta los scripts desde el workspace: `ic/`, `node_modules/`, `out/` y el visor se resuelven allí, no dentro del paquete compartido. Invoca los scripts por su ruta relativa desde ese workspace.

[scripts/icons.py](../scripts/icons.py) mapea alias a estilos nativos de draw.io (`mxgraph.azure2` con el nombre oficial del producto, `fortinet`, `rack`, `cisco`); úsalos con `d.vertex(x, y, w, h, nombre, style('alias'))`. Usa primero el ícono nativo. Si no existe, el equipo autoriza imágenes de internet (logos de empresa, banderas, íconos de producto, también desde Iconify) con estas condiciones:

- Fuente oficial del fabricante o colección con licencia que permita su uso (p. ej. MIT, CC0, CC BY o kit de prensa); nada de imágenes de buscador sin licencia.
- Descárgala una vez, preferentemente en SVG, a `ic/<alias>.svg` y úsala con `d.img(x, y, w, h, 'alias')`, que la incrusta en el `.drawio`. Nunca enlaces una URL en caliente.
- Registra en `ic/FUENTES.md` alias, URL de origen, licencia y fecha de descarga.
- Revisa el contenido del SVG (sin scripts ni recursos externos) y su aspecto en el primer PNG.

Si no hay ícono nativo ni imagen con fuente verificable: etiqueta `tag:Nombre:#color`.

El render requiere `viewer-static.min.js` de JGraph/drawio: referencia original `https://raw.githubusercontent.com/jgraph/drawio/dev/src/main/webapp/js/viewer-static.min.js`. Esa rama es mutable: selecciona una revisión concreta, verifica procedencia/integridad y revisa el archivo antes de colocarlo en el workspace. No descargues ni ejecutes código remoto automáticamente. El navegador ejecuta ese visor; sin revisión/autorización, detén el render. Las fuentes web y SVG pueden requerir recursos externos: revisa también esos accesos y no uses contenido confidencial.

## API conservada

[scripts/dgen.py](../scripts/dgen.py) ofrece `D(name, title)`, `P`, `vertex`, `img`, `iw`, `icons_row`, `box`, `container`, `badge`, `num`, `text`, `edge`, `extent`, `bbox` y `save`. Conserva sus parámetros y constantes OneByte. `IC` se resuelve en `ic/` del workspace.

- Crea el título en `hdr` (parte inicial). `P('base')` marca elementos siempre visibles; `P('p1', 'Parte 1 · Idea')` registra cada parte, sin duplicar claves.
- Cada celda pertenece a la parte activa al crearla. `save(OUT, windows={'p1': 'with-base'})` genera el multipágina, páginas individuales y metadatos. Una sola parte no crea páginas parciales; dos o más crean una por parte, con las otras al 16 %. El título `hdr` solo aparece en la completa; `base` siempre aparece.
- `windows` acepta rectángulos `(x0, y0, x1, y1)` o `with-base`; por defecto usa la caja envolvente de la parte. Incluye al menos un vértice por parte; el generador no valida entradas vacías ni es un servicio para entradas no confiables.
- Usa lienzo de aproximadamente 1700 px, texto mínimo 20 px, títulos 22–24 px, grilla de 20 px y filas alineadas con separación fija.
- Usa `ex/ey/nx/ny` y `pts` para rutas. En secuencias, deja `label` vacío y coloca etiquetas cortas con `text` encima de las líneas.

Toma [scripts/ejemplo.py](../scripts/ejemplo.py) como muestra runnable; al escribir una figura real importa `D` y constantes desde `dgen`, usa `OUT` inyectado por el renderizador y llama `save`. No copies el `fig43.py` incompleto del original.

## Render y recortes

Con entorno, visor y contenido revisados, ejecuta desde el workspace `.venv/bin/python RUTA_RELATIVA_AL_PAQUETE/scripts/build.py RUTA_A_LA_FIGURA.py`. `build.py` ejecuta la figura como Python: úsalo solo con fuentes locales revisadas. El directorio del script de figura debe poder importar `dgen` (por ejemplo, configura `PYTHONPATH` a la carpeta `scripts` compartida).

El [renderizador](../scripts/build.py) escribe en `out/<nombre-del-script>/`, abre el visor local con Playwright, espera SVG y captura a escala 2. Pillow crea `full.png`, recortes `part-<k>.png` con margen 22 px y `map.png` con bordes discontinuos y números. Revisa y ajusta ventanas: la captura presupone escala 2 y anclas de origen; no promete equivalencia entre versiones del visor.

Si una parte no es rectangular, dibuja dos recuadros en el mapa con el mismo número (solo en el primero). El script genera un rectángulo por parte: esta adaptación es manual, no está automatizada. Conserva la correspondencia con los recortes y no entregues el mapa sin verificarlo. Sigue la [revisión y montaje](revision-y-diapositivas.md) antes de entregar.
