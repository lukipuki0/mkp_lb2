# Revisión y diapositivas

## Revisión obligatoria (mirar los PNG antes de entregar)

- Ninguna flecha atraviesa cajas ni texto; salidas y entradas alineadas (`ex/ey/nx/ny`). Reencaminar con `pts` si hace falta.
- **Etiquetas de flechas cortas** (diagramas de secuencia): no usar `label` del edge, poner un `d.text` encima de la línea.
- Encabezados de carril o de columna no se superponen con el primer paso; si se tocan, bajar los pasos 30 px.
- Íconos al mismo tamaño dentro de una fila (36–40 px); nativos de draw.io o imágenes de internet con origen y licencia registrados en `ic/FUENTES.md`, verificados en el PNG.
- Líneas gruesas, espaciado generoso, ninguna flecha sobre texto, nada suelto.
- Cuerpo vertical legible a 7–8 pt efectivos; horizontal solo en anexos.
- Una secuencia por imagen; gobierno arriba y Zero Trust/seguridad abajo en vistas lógicas.
- Gabinetes: listado U por U y U libres revisadas.
- Cada recorte de parte contiene cajas completas (no cortadas). Si el recorte queda muy alto o muy ancho, ajustar `windows` o dividir de nuevo.
- Sin agrupaciones: transiciones repetidas se dibujan una por una; nunca «×3», «N nodos» ni contenedores que las agrupen.
- Sin texto explicativo ni cajas `NOTE`; sin datos de más.

## Montaje en diapositivas (Slides de Claude, 1920×1080)

- **Diapositiva mapa:** imagen `map.png` a la izquierda (≈1140×944) y a la derecha eyebrow, título «<Tema> en N partes», una frase de cómo leerlo y la lista numerada de partes con una línea cada una. Para diagramas muy anchos (relación > 1,8), imagen arriba a todo el ancho y lista en columnas abajo.
- **Diapositiva por parte:** eyebrow «0X · Figura X.Y · Parte N de M», título que afirma la idea (no «Parte 1»), la imagen de la parte en un panel blanco y tres tarjetas: Qué mostramos (Search), Cómo funciona (Settings), Por qué importa (Trust, borde superior cian). Layout ancho (imagen arriba 1696×528, tarjetas abajo) o lateral (imagen 1108×848 a la izquierda, tarjetas apiladas a la derecha) según cuál deje la imagen más grande.
- No agregar miniaturas «Parte 1 de 2» dentro de cada diapositiva: el mapa previo ya orienta.
- Notas del orador con el detalle y la sección del documento fuente.
- Límite del editor: máximo 24 hijos posicionados por `div`; en líneas de tiempo, envolver cada fila en su propio `div` relativo.

## Entrega

- Enviar todos los `.drawio` (multipágina) junto con PNG/PDF al usuario y, si hay presentación, publicar las imágenes como assets y referenciarlas por su `/_blob/<id>`.
- Nombrar `PCVTE-Figura-X-Y-v3.drawio` (o `<Proyecto>-<Diagrama>-v3.drawio`). Diagramas nuevos que no existen en el informe se indican como «diagrama nuevo a partir de <sección>».
