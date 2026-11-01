# Proyecto PUCV y rúbrica por etapa

## Fuentes y estructura observada

- [Proyecto local: main.tex](../../../../informes/avance_seminario/archivos_latex/Informe-avance-tesis/main.tex).
- [Estilo pucv_inf_2024.sty](../../../../informes/avance_seminario/archivos_latex/Informe-avance-tesis/pucv_inf_2024.sty).
- [Rúbrica de contenido de Desarrollo de SW](../../../../rubricas/Rubrica-contenido-Desarrollo-de-SW.md): consultar también sus descripciones, no solo los títulos de esta tabla.

La raíz LaTeX es `informes/avance_seminario/archivos_latex/Informe-avance-tesis/`. Usa clase `report`, paquete `pucv_inf_2024` y archivos de contenido en `avance/`; `default/` es material base, no el informe que se debe completar. `main.tex` incorpora resumen, introducción, problema, objetivos, metodología de trabajo, plan, marco teórico, conclusión e IA. No contiene todos los capítulos para todas las etapas.

El estilo carga `graphicx`, `amsmath`, `caption`, `enumitem` y `biblatex` con `style=ieee, backend=biber`; `main.tex` añade `booktabs` y registra `references.bib`. No se ha establecido el motor de compilación de Overleaf. No copiar configuración de OneByte ni cambiar paquetes por conveniencia.

Los ejemplos del informe colocan el caption debajo de la figura, seguido del label; para tablas, el caption va arriba. Usan `[hbtp]` y columnas de párrafo cuando corresponde. La configuración base limita la subdivisión formal a tres niveles: capítulo, sección y subsección. Los títulos del autor que excedan esos niveles se conservan como rótulos internos diferenciados, sin añadir niveles de sección ni numeración propia. La existencia de otros comandos en LaTeX no acredita que satisfagan esa norma. Consulta la [configuración base](../../../../informes/avance_seminario/archivos_latex/Informe-avance-tesis/default/Configuracion_formato/Configuracion.tex) sin modificarla.

`longtable`, `tabularx`, `makecell` y `pdflscape` no figuran en el preámbulo y estilo inspeccionados. Verifica su estado antes de utilizarlos. Una tabla multipágina puede requerir añadir una dependencia al preámbulo como parte de la conversión; incorpora la mínima necesaria sin cambiar el estilo compartido.

## Cobertura macro

SA = Seminario avance; SF = Seminario final; PA = Proyecto avance; PF = Proyecto final. ✓ significa exigido; — significa sin marca en la rúbrica, no prohibido. Los números siguientes identifican puntos de rúbrica, **no números de capítulo**.

| Punto y título exacto | SA | SF | PA | PF | Representación LaTeX |
| --- | :---: | :---: | :---: | :---: | --- |
| 1. Resumen y palabras claves | ✓ | ✓ | ✓ | ✓ | Preliminar con `\chapter*` |
| 2. Introducción | ✓ | ✓ | ✓ | ✓ | Capítulo |
| 3. Descripción del problema | ✓ | ✓ | ✓ | ✓ | Capítulo |
| 4. Objetivos | ✓ | ✓ | ✓ | ✓ | Capítulo |
| 5. Metodología de trabajo | ✓ | ✓ | ✓ | ✓ | Capítulo |
| 6. Plan de trabajo | ✓ | ✓ | ✓ | ✓ | Capítulo |
| 7. Marco referencial | ✓ | ✓ | ✓ | ✓ | Capítulo |
| 8. Captura de necesidades, funcionalidades, restricciones | — | ✓ | ✓ | ✓ | Capítulo |
| 9. Especificación de requerimientos | — | ✓ | ✓ | ✓ | Capítulo |
| 10. Diseño de la aplicación/sistema | — | ✓ | ✓ | ✓ | Capítulo |
| 11. Desarrollo de la solución | — | ✓ | ✓ | ✓ | Capítulo |
| 12. Desarrollo del software | — | — | ✓ | ✓ | Capítulo |
| 13. Pruebas de software y validaciones con usuarios | — | — | ✓ | ✓ | Capítulo |
| 14. Producto software final | — | — | ✓ | ✓ | Capítulo |
| 15. Discusión de la propuesta y contribuciones | — | ✓ | ✓ | ✓ | Capítulo |
| 16. Limitaciones del proyecto | — | ✓ | ✓ | ✓ | Capítulo |
| 17. Conclusiones y trabajo futuro | ✓ | ✓ | ✓ | ✓ | Capítulo |
| 18. Referencias | ✓ | ✓ | ✓ | ✓ | `\printbibliography` |
| 19. Anexos | ✓ | ✓ | ✓ | ✓ | `\appendix` y `\chapter` |

Comprobación compacta: SA exige 1–7 y 17–19; SF, 1–11 y 15–19; PA y PF, 1–19, incluido «Producto software final» en avance de Proyecto. No rellenar etapas futuras ni afirmar experimentos realizados para completar casillas.

## Correspondencia y excepciones

- El cuerpo se organiza con `\chapter{Título}`, incorporado desde `main.tex`. Para nuevas secciones, usa paquetes `avance/seccion-N/seccion-N.tex`, con archivos propios en `tablas/` y recursos utilizados en `figuras/`. No reorganices los capítulos existentes como efecto lateral. Mantén el desglose del autor mediante niveles compatibles con la norma; no conviertas cada Markdown en un capítulo por defecto.
- Registrar equivalencias entre los títulos de rúbrica y los existentes: por ejemplo, el archivo `objetivos.tex` usa «Objetivo general y específicos». Revisar cobertura antes de proponer renombrados; no modificar títulos o alcance silenciosamente.
- Resumen: mantener el tratamiento preliminar `\chapter*{Resumen}` y la entrada de índice que corresponda. La rúbrica pide resumen y palabras claves en español e inglés. El `avance/resumen.tex` inspeccionado solo contiene español: reportar la ausencia del abstract, sin inventar traducción.
- Referencias: conservar `\printbibliography[heading=bibintoc, title={Referencias}]`; no crear un capítulo numerado adicional ni una lista bibliográfica manual.
- Anexos: usar `\appendix` una vez y luego capítulos de anexos, con contenido real y anonimización cuando corresponda. El `main.tex` inspeccionado no los incorpora: informar el faltante, no fabricar anexos.
- `avance/Ia` es contenido suplementario existente, no un punto adicional de la rúbrica ni sustituto de alguno. No eliminarlo como efecto lateral.
- La numeración automática depende de los capítulos realmente incluidos. No fijar contadores para hacer coincidir el punto 17 con un capítulo 17. La presencia de un encabezado no acredita la calidad ni la suficiencia de su contenido.

## Casos de referencia de la sección 8

La fuente localizada durante la auditoría está en [seccion-8](<../../../../informes/final_seminario/secciones/seccion-8/8. Captura de necesidades, funcionalidades, restricciones.md>). Esta ruta es una observación, no una selección permanente: compruébala al recibir un pedido.

- El origen contiene encabezados de cinco niveles, una instrucción inicial de rúbrica y marcas de avance. Conserva capítulo, sección y subsección como niveles formales y los títulos más profundos como rótulos internos, en el mismo orden. No modifiques las definiciones humanas, incluidas las de Q21.
- Mezcla imágenes Markdown con nombres abreviados e inserciones Obsidian desde raíz o con basename. Los recursos se distribuyen bajo `figuras/`; la resolución estrictamente relativa al Markdown no basta. Hay copias de pruebas que pueden producir coincidencias múltiples.
- La tabla 9 contiene 26 filas de hallazgos y tres columnas; la tabla 10, once filas y cinco columnas. Las tablas 7 y 8 incluyen celdas extensas y saltos internos. Son casos de control de fidelidad y composición, no una autorización para resumirlos.
- Las figuras originales 6a–g y 9a–s son referencias que deben mapearse a etiquetas y números automáticos, sin suponer subfiguras.
- Las cinco fuentes enumeradas al final del Markdown (Palomares, Ozkaya, protocolo de entrevistas, Franch y Zenodo 6754676) no tenían entradas correspondientes en el `references.bib` inspeccionado. Revisa la bibliografía actual, reutiliza coincidencias reales y añade las fuentes verificadas que falten durante la conversión, sin pedir una aprobación por entrada. No reutilices una clave por coincidencia de apellido.
- El punto 8 de la rúbrica no garantiza capítulo 8: en el `main.tex` inspeccionado, ubicar este contenido después del marco teórico produciría otro número. Respeta la numeración automática.

Estas observaciones describen la copia local inspeccionada; verifica su estado al convertir. Actualizar esta guía no resuelve las citas, añade paquetes ni convierte el capítulo.
