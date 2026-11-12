# Marca y contenido OneByte

## Resultado esperado por diagrama

1. `PCVTE-Figura-X-Y-v3.drawio` multipágina: página «completa» + una página por parte (la parte activa nítida, el resto atenuado al 16 %).
2. PNG renderizados: `full.png`, `part-<k>.png` (recorte de la parte) y `map-<fig>.png` (diagrama completo con un borde punteado y un número por parte).
3. En la presentación: **una diapositiva mapa** (diagrama completo + lista numerada de partes) y luego **una diapositiva por parte** con tres tarjetas: «Qué mostramos», «Cómo funciona», «Por qué importa». Diagramas que caben legibles van completos en una sola diapositiva.

## Reglas de contenido

- No inventar componentes, conexiones ni nombres: cada flecha y componente sale explícitamente del texto fuente. Las brechas se consultan como observación, no se adivinan. Si se mejora el diagrama, se conserva el significado y se indica «versión 3» en el título.
- Sin texto explicativo ni cajas `NOTE`: solo cajas con nombre corto, conexiones, etiquetas cortas sobre las líneas y, si hace falta, un glosario.
- Sin información de más (p. ej. listas de ciudades): solo lo que la figura necesita.
- No agrupar nunca: cada nodo con su símbolo; nunca «×3», «N nodos» ni contenedores que agrupen transiciones.
- Todo lo que el texto nombra lleva su ícono nativo de draw.io: `mxgraph.azure2` con el nombre oficial del producto Azure, `fortinet`, `rack`, `cisco`. Si no existe ícono nativo, se usa una imagen de internet de fuente oficial o con licencia de uso (p. ej. logos de Entel y Claro, Microsoft Entra ID, banderas de país), descargada a `ic/` y registrada según el [flujo](flujo.md).
- Sin máximo de cajas. Si no cabe legible, dividir en varias figuras (o partes) sin eliminar ninguna relación ni componente aprobado. Una secuencia por imagen.
- Vistas lógicas: estilo de columnas por capa (D), banda de gobierno arriba y banda de Zero Trust/seguridad abajo. Despliegue, red, secuencia y data center usan el layout propio de su tipo.
- Bases Art. 40.4: figuras del cuerpo en vertical (carta/oficio), legibles a 7–8 pt efectivos al escalar al ancho de página; los anexos gráficos pueden ir en horizontal.
- Data center: gabinetes con listado U por U, revisando si queda una U libre; referencias: TikZ `sala-matriz` y `gabinete-complejo`.

## Identidad visual

- Tipografía: Source Sans 3 en el diagrama (Montserrat en títulos de diapositiva).
- Paleta: navy `#0B2252`, azul `#1E4FD0`, cian `#18C0E4` (oscuros `#18A9CC`, `#0E6E86`), fondo `#F4F7FB`, tinta `#12203F`, texto secundario `#2B3F66`, gris borde `#8A9BB8`, tintes `#E3ECFB` / `#E3F6FB`, nota `#EEF2F8`, rojo `#B42318`, ámbar `#B26A00`, verde `#1F7A45`.
- Líneas gruesas (≥ 3 px), espaciado generoso, nada suelto (todo nodo conectado salvo los que el texto no conecta).
- Contenedores: borde azul 3 px, esquinas 18. Cajas: borde 2 px, esquinas 10. Flujo principal en azul sólido; flujos de solo lectura o asíncronos en cian discontinuo; bloqueos en rojo.
- Proveedor sin ícono nativo ni imagen con fuente verificable: etiqueta de color con texto blanco (`tag:Lenovo:#E2231A`), nunca un logo inventado.
