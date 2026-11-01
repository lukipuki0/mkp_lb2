---
name: figuras-academicas
description: "Trigger: figuras académicas, gráficos de datos, diagramas, draw.io, Matplotlib, Plotly, arquitectura, flujo/metodología. Crea figuras reproducibles y trazables de apoyo para la tesis MASRE."
---

## Activation Contract

Crea diagramas de procesos, arquitectura, flujos y secuencias, y gráficos de datos o comparaciones respaldadas. Elige la herramienta por el contenido y el destino, no por decoración. No reestructures prosa ni inventes gráficos. Paquete independiente: no requiere otras skills.

## Hard Rules

- Define propósito y fuente; distingue evidencia, interpretación, propuesta y desconocido. Una vista no decide arquitectura ni valida MASRE.
- No inventes datos, conexiones, decisiones ni stack. Consulta vacíos sustantivos; resuelve estilo reversible sin preguntar.
- Por defecto entrega solo gráfico/diagrama y título opcional. Conserva etiquetas, ejes, unidades y leyendas necesarios para descifrarlo. Sin subtítulos, citas/fuentes, notas explicativas ni acotaciones de alcance incrustadas.
- Deja pie, citas, muestra, supuestos, límites y consejos en el README externo. El autor los incorporará al escrito; nunca los insertes automáticamente. Otras anotaciones internas requieren petición explícita o semántica indispensable.
- Usa etiquetas breves, jerarquía y agrupaciones fieles; evita decoración y repetición. Prioriza formas neutras e iconos solo por necesidad semántica; nunca inventes logos.
- Antes de cada generación, adaptación o recomendación, lee la [guía de selección](references/seleccion-de-graficos.md) y el [catálogo operativo de las infografías](references/catalogo-infografias.md). Elige familia y variante según pregunta y datos; registra el motivo en el README externo. Sugiere una alternativa si mejora la lectura; considera tabla o ninguna figura. No copies errores, datos simulados ni decoración de los ejemplos.
- Elige distribución según contenido y documento. Comprueba contraste, dirección, límites y legibilidad al tamaño final; grises cuando corresponda.
- Distingue elaboración propia, adaptación y reproducción. Usa citas bibliográficas verificadas `[1]` o `[1,2]`, sin páginas/secciones dentro. Sin respaldo, marca fuente pendiente; no impliques reanálisis ajeno.
- Guarda entregables junto al informe elegido, fuera de la skill, con slug estable. No presupongas LaTeX local: el informe formal está en Overleaf. No crees carpetas vacías.
- No instales, descargues, ejecutes código remoto ni publiques automáticamente.

## Decision Gates

- Densidad: abstrae/divide sin perder información; evita partes/mapas innecesarios.
- Evidencia insuficiente: explicita propuesta/desconocido; consulta decisiones sustantivas.
- Render ausente: editable como borrador y revisión visual pendiente, sin instalar.

## Execution Steps

1. Identifica informe, propósito y fuentes: [contenido](references/marca-y-contenido.md). Lee la [guía de selección de gráficos](references/seleccion-de-graficos.md) y el [catálogo de las infografías aportadas](references/catalogo-infografias.md), verifica estructura y denominadores, y elige familia y variante antes que la librería. Si falta una decisión indispensable, pregunta; si se pide consejo, recomienda con motivo y alternativa pertinente.
2. Selecciona herramienta según el [flujo](references/flujo.md): API local `D` y draw.io para diagramas; Matplotlib para gráficos estáticos de datos; Plotly cuando se necesite interactividad. Genera la fuente reproducible y prueba en scratch aislado. Consulta también el [README](README.md).
3. Renderiza con herramientas disponibles/autorizadas e inspecciona: [revisión](references/revision-y-diapositivas.md). XML válido no prueba calidad visual.

## Output Contract

Entrega `.drawio` para diagramas; para gráficos de datos, script reproducible y referencia a los datos de entrada. Incluye solo exportaciones efectivamente generadas. README externo con pie sugerido, fuentes, consejos, comprobaciones y pendientes. Declara pruebas visuales ausentes; no prometas PNG/PDF ni revisión humana inexistentes.
