# Pendientes del paper GWO–DE–DTW

Resumen de continuidad del manuscrito `overleaf/gwo_de_dtw_paper/`.
Este documento registra trabajo pendiente; no autoriza correcciones adicionales ni nuevas campañas.

## Estado confirmado

- Overleaf actualizado al commit `0684d8b` (`docs: finalize discussion and clarify statistical evaluation`).
- PDF local: `overleaf/gwo_de_dtw_paper/mdpi_gwo_de_dtw.pdf`, 36 páginas en la última compilación.
- Última compilación exitosa, sin citas ni referencias sin resolver. Esto no certifica la exactitud de todos los metadatos ni reemplaza una revisión visual.
- El manuscrito incluye introducción, estado del arte, método, resultados, sensibilidad, protocolo estadístico, discusión, conclusiones, limitaciones y trabajo futuro.
- Keywords: seis, opción C elegida por el usuario.
- Autores: Lucas Erazo, Ricardo Soto, Pamela Hermosilla, José García, El-Ghazali Talbi y Emanuel Vega. José García sin ORCID; correspondencia de Emanuel Vega.
- Los repositorios de Overleaf son independientes del repositorio principal. Un push a GitHub no actualiza Overleaf y viceversa.

## Correcciones científicas expresamente pendientes

El usuario autorizó los puntos 2, 4, 6 y 7 del informe de coherencia y dejó el resto pendiente.

### P1. Formulación física y económica de HRES2

- [ ] Verificar el modelo efectivamente usado para las campañas antes de modificar ecuaciones.
- [ ] Explicitar cómo se escala la curva de una turbina de 5 MW a la capacidad eólica instalada optimizada.
- [ ] Distinguir potencia por turbina, capacidad total y, si corresponde, capacidad equivalente continua.
- [ ] Definir operacionalmente `E_delivered,annual` del denominador del LCOE: flujos horarios incluidos, acumulación anual y conversión MWh/kWh.
- [ ] Distinguir el objetivo LCOE de los indicadores adicionales de hidrógeno.

Ubicación: `sections/03_background/background.tex` del manuscrito. Contrastar con el modelo y los registros de campaña; no asumir que una omisión documental implica un error en los resultados calculados.

### P3. Parámetros compartidos y binarización

- [ ] Contrastar configuraciones de las campañas principales para identificar exactamente los parámetros compartidos entre MKP y HRES2.
- [ ] Corregir expresiones que sugieren parámetros numéricos distintos por dominio cuando lo distinto es la representación, decodificación o señal evaluada.
- [ ] Declarar inequívocamente el valor de G3 de la campaña principal, verificándolo en los registros.
- [ ] Aclarar que G3 permanece fijo entre perfiles y que el controlador modifica a, F y CR, no el offset de binarización.

Ubicaciones: `sections/03_background/background.tex`, `sections/04_proposed_approach/proposed_approach.tex`, configuración experimental y conclusiones.

### P5. Designación del control estadístico

- [ ] Aclarar, si se autoriza, que Binary-Complex se designó como control fijo para este reanálisis retrospectivo.
- [ ] No insinuar selección previa a las campañas ni preregistro sin evidencia documental.

Esto solo modifica la descripción de cuándo se eligió el control; no requiere cambiar pruebas ni resultados. El usuario pidió explicación, pero no autorizó aplicar este punto.

### P8. Motivación del contador de estancamiento

- [ ] Sustituir la afirmación absoluta de que un contador no distingue una pausa temporal de una meseta persistente.
- [ ] Describir su información real: tiempo desde la última mejora, sin representar por sí solo la magnitud y distribución de las mejoras dentro de una ventana.
- [ ] No atribuir a DTW una capacidad discriminativa superior demostrada: falta una ablación pertinente.

Ubicación: `sections/01_introduction/introduction.tex`.

## Fuentes y bibliografía

### Correcciones confirmadas pendientes de aplicar

- [ ] `derrac2011practical`: corregir el tercer autor. El PDF original identifica a Daniel Molina; el `.bib` del manuscrito todavía contiene a L. Sánchez.
- [ ] `thirunavukkarasu2023comprehensive`: corregir el tercer autor a Himadri Lala y completar el título: *A comprehensive review on optimization of hybrid renewable energy systems using various optimization techniques*. El `.bib` todavía contiene H. Sharma y un título truncado.
- [ ] Revisar la duplicación impresa «Proceedings of the Proceedings» en las referencias Keogh, Derbel, Żychowski y Pei. Ajustar `booktitle` según el comportamiento real de la plantilla y recompilar.
- [ ] Completar la identificación del artículo 129 de Talbi en ACM Computing Surveys; no confundir extensión de 32 páginas con identificador de artículo.
- [ ] Revisar puntuación del título de Modu, paréntesis de «Life Cycle Assessment (LCA)» en Nagapurkar y el rango `4921--4928` de Reijnen.

Ubicación principal: `overleaf/gwo_de_dtw_paper/references.bib`. Si se corrigen registros equivalentes en `papers/referencias.bib`, preservar claves y evitar DOI duplicados.

### Atribuciones que requieren evidencia original

- [ ] Establecer la procedencia exacta de la penalización `100 + 10*AGSR`; no atribuir esos coeficientes a una revisión sin un pasaje que los sostenga.
- [ ] Justificar el límite de exportación anual del 20 % como supuesto del escenario o identificar su fuente primaria. Exportar energía no equivale a depender de importaciones de la red.
- [ ] Documentar la procedencia del horizonte de 25 años, tasa de descuento de 4,35 % y parámetros de costos; distinguir fórmulas generales de valores adoptados.
- [ ] Distinguir controladores entrenados previamente de aprendizaje online; no afirmar que todo mecanismo aprendido requiere una fase offline antes de operar.
- [ ] Describir con cautela la conmutación GWO/DE del artículo de Wang: sus ecuaciones impresas y uso reportado de DE presentan una inconsistencia señalada en la ficha.
- [ ] Resolver la identidad bibliográfica de `talbi2016combining` contra el original correcto; no sustituir automáticamente una versión de 2016 por la revisión de 2013.

Los informes de fuentes basados en fichas no equivalen a una nueva lectura independiente de los originales. Una consulta externa bloqueada no prueba un DOI incorrecto.

## Maquetación y presentación

- [ ] Verificar visualmente el posible solapamiento de las fórmulas de eficiencia de carga/descarga de batería, observado en la página 7 de la versión revisada.
- [ ] Unificar separadores de miles y criterio de precisión de las ganancias entre tablas de campaña principal y sensibilidad, conservando los datos sin redondeo para los cálculos.
- [ ] Unificar presentación de p-valores en tablas y prosa; no cambiar probabilidades calculadas.
- [ ] Aclarar brevemente que la figura de arquitectura representa la ruta MKP con LB2 y reparación; HRES2 utiliza otra decodificación.
- [ ] Hacer inequívocos los encabezados «At best» y «Changes» de HRES2, sin volver a captions extensos.
- [ ] Revisar la densidad de la tabla de medias MKP sin reducir artificialmente la tipografía con `resizebox`.
- [ ] Revisar cortes de DOI/URL en bibliografía.
- [ ] Confirmar revista y configuración MDPI: quedan elementos genéricos de plantilla y advertencias de compilación.

Los márgenes estrechos señalados por un revisor NO son un defecto confirmado. Comparar con la plantilla MDPI antes de modificar geometría. Las páginas de los informes corresponden a un PDF anterior de 35 páginas; localizar los elementos por título o etiqueta en la versión actual de 36 páginas.

## Cambios ya resueltos: no reabrir sin nueva evidencia

- Abstract: retirado el cierre que atribuía beneficios generales al control adaptativo; no se establece ventaja de Binary-Complex sobre el mejor perfil fijo.
- Análisis global de 27 medias de instancia identificado como principal; análisis por tamaño identificados como secundarios.
- DTW/DDTW de HRES2: objetivos coincidentes no son confirmaciones independientes; se conserva la corrección conservadora de diez contrastes.
- Sensibilidad: índices 0, 15 y 29 verificados directamente contra capacidades y pesos, con factores 0,25, 0,50 y 0,75. No son réplicas de una misma dificultad ni se agrupan sus diferencias de ganancia en bruto.
- La comprobación de holguras es derivada de los datos, no una afirmación de haber verificado la documentación original del generador.
- No rechazo no se interpreta como equivalencia; alcanzar la referencia 59.965 no se presenta como óptimo certificado.
- La identidad DDTW respecto de una referencia constante limita ese componente de distancia, no toda la política Binary-Simple, que también usa contador y reinicios.

## Archivos locales y preservación

- No subir automáticamente `analysis/`, `figs/` sin seguimiento ni auxiliares de LaTeX al repositorio de Overleaf.
- La biblioteca `papers/07-hibridos-gwo-de/` contiene seis PDF y fichas. `papers/08-metodologia-estadistica/` contiene Derrac y las versiones editorial y arXiv de Carrasco; no eliminar originales sin autorización.
- Comprobar la integración de las carpetas 07 y 08 en los índices generales de la biblioteca. El registro histórico `downloads.json` de 07 puede seguir marcando como no disponibles PDF luego incorporados manualmente.
- Existen cambios locales de figuras, biblioteca y otros documentos que no pertenecen a este resumen. Preservarlos y no incluirlos en su commit por accidente.

## Próximo paso recomendado

Confirmar con el usuario el próximo bloque de correcciones. Priorizar la formulación HRES2 y los dos errores confirmados de autoría bibliográfica. Después, compilar y revisar las páginas afectadas. No reiniciar campañas ni reformular el análisis estadístico por defecto.
