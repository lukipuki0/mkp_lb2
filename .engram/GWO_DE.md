# GWO-DE: copia curada de Engram

Actualizada el 2026-11-01. Fuente: observaciones Engram del proyecto `mkp_lb2` con tema GWO-DE. No es una exportación completa de la base de datos ni incluye credenciales, ids de sesión ni memorias de otros proyectos.

## Decisiones

- La propuesta vigente es GWO-DE. WOA-ABC quedó retirado. No tratarlo como alternativa.
- El diseño compartido vive en `gwo_de_dtw`. `gwo_de_sequential` es un paquete aparte: no importa ni edita `gwo_de_dtw`; reutiliza solo `dtw_stagnation.py` y tiene su propio parser OR-Library.
- El modo por defecto de `gwo_de_dtw` sigue siendo competitivo. `sequential` es opt-in.
- El selector se llama `DTW_KIRITO` / `--dtw-kirito`. El default es false. Exports nuevos usan `dtw_kirito`; un replay viejo puede decir `dtw_main`.
- Cuatro políticas de estancamiento, mismos perfiles explore/exploit (`a`, `F`, `CR`, `G3=0.1`), monitor DDTW, ventana 40, banda 4:
  - `binary_simple`: entra a explore cuando `D2 <= theta_c`; vuelve con `no_improve_len < plateau_max`; resetea la ventana al cambiar.
  - `binary_complex`: comportamiento original; entra por el fire del monitor; resetea la ventana. Tiene que seguir idéntico al híbrido base en esa política.
  - `continuous_simple`: intensidad `1 - clip(D2 / (theta_c * 2), 0, 1)`; no resetea.
  - `continuous_complex`: intensidad sigmoide; no resetea. `theta_delta` puede ser negativo.
- En continuas, `changed` significa que cambió el nombre del perfil, no que el monitor se haya reemplazado. El replay temporal solo resetea si la política es binaria.
- No mezclar este control con el de `DTW_optimization-main`: allí la ventana rueda y no se resetea, y la binarización es sigmoide, no LB2.

## Código y pruebas que Engram registró como hechas y sin push

- Políticas, controller, `run.py`, README y `tests/test_gwo_de_policies.py`.
- Campaña `gwo_de_dtw/campaign.py`: 9 archivos x 3 índices x 31 semillas x 4 políticas = 3348 corridas planificadas. Guarda traza y plots solo del mejor fitness por política; empate, semilla menor. `run.py` sigue siendo una semilla.
- Resumen en español `resumen.txt` por instancia y modo. El ranking es por mejor fitness dentro de la instancia, no entre instancias.
- Salida agrupada: una carpeta por invocación, cuatro subcarpetas de política, `convergence_comparison.png/pdf`. Un destino no vacío se rechaza antes de correr.
- Paquete `gwo_de_sequential/` con el mismo default de 3348 corridas. El dry-run validó el plan sin escribir. La campaña completa no se lanzó.

## Pilotos de una instancia

Instancia `mknapcb1` índice 0, semilla 1, DDTW, W40, banda 4, N30, T1000, LB2, G3=0.1, Kirito false, 120030 evaluaciones, salvo que se diga otra cosa.

Competitivo, cuatro políticas, fitness y cambios de etiqueta: binary_simple 24168/5, binary_complex 24243/11, continuous_simple 24094/7, continuous_complex 24144/1. No prueban superioridad.

Secuencial sobre esa misma semilla, fitness menor en las cuatro: 23916, 23963, 24076, 23958. Una semilla no prueba superioridad general.

Kirito contra el original, misma semilla: binary_simple 24168→23984 y 5→7 cambios; binary_complex 24243→24121 y 11→15; las continuas no cambiaron fitness. Más switches binarios y peor fitness en esa semilla. Sin claim estadístico.

## Rutas de resultados locales

- `gwo_de_dtw/results/mknapcb1_0_policies_s1_<policy>.json`
- `gwo_de_dtw/results/mknapcb1_0_policies_s1_original_grouped/`
- `gwo_de_dtw/results/mknapcb1_0_policies_s1_sequential/`
- `gwo_de_dtw/results/mknapcb1_0_policies_s1_dtw_main_<policy>.json` (nombre histórico; el job arrancó antes del rename)

## Pendiente al guardar esta copia

- La campaña de 3348 corridas no estaba lanzada.
- Estas notas no reemplazan leer el código de la rama antes de citar un número.
