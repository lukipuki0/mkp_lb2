---
name: diagramas-drawio-onebyte
description: "Trigger: diagramas, draw.io, OneByte, arquitectura y procesos. Genera diagramas editables por partes con identidad visual compartida."
---

## Activación

Carga esta skill al crear o rediseñar diagramas draw.io del proyecto o preparar su explicación por diapositivas.

## Reglas obligatorias

- Lee [marca y contenido](references/marca-y-contenido.md) antes de diseñar; conserva significado, nombres, tipografías y paleta. No inventes componentes ni conexiones.
- Lee [flujo de trabajo](references/flujo.md) antes de ejecutar. No instales ni descargues automáticamente; solicita autorización y usa un entorno aislado.
- Lee y aplica la [revisión y montaje](references/revision-y-diapositivas.md). Inspecciona los PNG antes de entregar; XML válido no demuestra calidad visual.
- Usa primero íconos nativos de draw.io ([icons.py](scripts/icons.py)). Si no existe uno nativo, el equipo autoriza imágenes de internet (logos, banderas, íconos de producto) de fuente oficial o con licencia de uso, descargadas a `ic/` y con su origen registrado; nunca enlazadas en caliente.
- Cada flecha y componente debe estar respaldado explícitamente por el texto fuente; las brechas se consultan, no se adivinan.
- No agrupes nodos ni pongas texto explicativo; si no cabe, divide en varias figuras sin quitar relaciones ni componentes aprobados.
- Figuras del cuerpo en vertical, legibles a 7–8 pt efectivos (Bases Art. 40.4); solo los anexos gráficos pueden ir en horizontal.
- Conserva el `.drawio` multipágina editable y las fuentes. Ubica los entregables bajo el informe/macrotarea verificada, no dentro de esta skill.

## Decisiones

| Caso | Acción |
| --- | --- |
| Cabe legible a 7–8 pt en vertical | Presenta completo en una figura. |
| No cabe legible | Divide en varias figuras; ninguna relación ni componente aprobado se elimina. |
| Varias secuencias | Una secuencia por imagen. |
| Vista lógica | Estilo de columnas por capa (D); gobierno arriba, Zero Trust/seguridad abajo. |
| Despliegue, red, secuencia o data center | Layout propio de su tipo, no columnas D. |
| Solo validar estructura | Ejecuta la muestra stdlib; declara pendiente el render y revisión visual. |
| Fuente incompleta o relación Jira desconocida | Solicita aclaración antes de producir el entregable. |

## Pasos

1. Identifica documento fuente, figura/sección, tarea e informe; confirma la ubicación según gobernanza.
2. Define partes, cajas y conexiones; usa la API `D` existente, sin crear otro generador.
3. Sigue el [README](README.md) para cargar, preparar y ejecutar; usa la muestra solo como demostración técnica, no como arquitectura real.
4. Renderiza en un entorno autorizado, inspecciona todos los PNG y corrige según la lista visual.
5. Registra fuentes, pendientes, resultados y revisión humana real.

## Contrato de salida

Entrega `.drawio`, PNG/PDF completos/partes/mapa y fuentes reproducibles; exporta PNG y verifica íconos antes de entregar; si hay presentación, incluye mapa, tarjetas y notas. Declara explícitamente cualquier render, inspección o revisión pendiente. Nombra `<Proyecto>-<Diagrama>-v3.drawio` e identifica diagramas nuevos y su sección fuente.

## Referencias

- [Marca y contenido](references/marca-y-contenido.md)
- [Flujo y API](references/flujo.md)
- [Revisión y diapositivas](references/revision-y-diapositivas.md)
