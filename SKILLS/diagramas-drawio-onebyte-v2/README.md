# Diagramas OneByte para el equipo

La fuente compartida es [SKILL.md](SKILL.md); `AGENTS.md` la indexa, pero no instala ni garantiza autodetección en todos los agentes. Aporta a [FP-292](https://asistec.atlassian.net/browse/FP-292), relacionado con [GitHub #117](https://github.com/TheVillegas/FEP/issues/117), sin cierre automático.

## Cargar y probar

Pide al agente: «Lee `.agents/skills/diagramas-drawio-onebyte/SKILL.md` y sus referencias; aplica OneByte al diagrama de la sección indicada, sin inventar componentes». Si tu herramienta tiene carga manual de skills, selecciona ese archivo. Todos los miembros usan la misma copia versionada.

La muestra solo necesita Python 3 y genera XML editable, sin instalar nada:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 .agents/skills/diagramas-drawio-onebyte/scripts/ejemplo.py RUTA_DE_SALIDA
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s .agents/skills/diagramas-drawio-onebyte/tests -v
```

Sustituye `RUTA_DE_SALIDA` por un directorio temporal o la ubicación de artefactos verificada. La muestra genera `Muestra-Diagrama-v3.drawio`, `page-full.drawio`, `page-p1.drawio`, `page-p2.drawio` y `meta.json`; no produce PNG.

## Preparar el render, solo con autorización

Desde la raíz del repositorio define `WORK` como una carpeta de trabajo relativa dentro de la macrotarea verificada. No ejecutes literalmente los marcadores. No versiones `.venv`, `node_modules`, descargas ni HTML temporal; usa una zona excluida previamente aprobada o un espacio temporal externo para el entorno y copia solo entregables revisados.

```bash
python3 -m venv "$WORK/.venv"
cd "$WORK"
.venv/bin/python -m pip install playwright pillow
mkdir -p ic out
```

Estos pasos son manuales y opcionales: requieren aprobación, conectividad y revisión de dependencias; no modifican las dependencias del proyecto. Chromium compatible debe estar disponible para Playwright; su instalación también requiere autorización separada. Instala localmente las fuentes Source Sans 3 y Montserrat si faltan, con aprobación.

Lee el [flujo detallado](references/flujo.md) para iconos, visor, API y renderizado. **Abre e inspecciona los PNG**: recortes, flechas, legibilidad, logos y superposiciones no se validan con XML ni con las pruebas.

## Entregables y límites

Conserva fuentes, `.drawio` multipágina, PDF, `full.png`, `part-<k>.png` y `map.png` (renómbralo `map-<fig>.png` al distinguir figuras). Las páginas individuales y `meta.json` son intermediarios útiles para reproducir el render. No se incluyen aquí diagramas generados ni slides finales.

Aplica la [política de ubicación](../../../docs/governance/artifact-location.md): Informe 3 bajo la macrotarea verificada en Jira; Informes 1/2 conservan sus rutas. No deduzcas la carpeta a partir de FP-292 ni publiques entornos de trabajo como entregables.

Origen: contenido de `diagramas-drawio-onebyte/SKILL.md` del ZIP aportado por el usuario. Se separaron referencias y scripts para la integración solicitada por el usuario; el ZIP no especifica licencia ni autoría y no se le atribuyen metadatos no verificados. El ejemplo `fig43.py` original tenía elipsis: se reemplaza por una muestra completa y ficticia. Los renderizadores externos, sus descargas y las restricciones del editor de slides no se ejercitan en esta integración.
