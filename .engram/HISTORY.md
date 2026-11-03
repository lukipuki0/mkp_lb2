# Historia, procedencia y límites de la recuperación

## WOA–ABC (antecedente distinto de adaptive_mh_selection)

Memorias 619, 625, 626 y 627:

- Se trabajó tanto en `woa_abc/` como en la carpeta posteriormente aportada `woa_abc_version_vieja/`. No asumir que contienen la misma implementación ni que están presentes en todas las ramas.
- Cuatro variantes seleccionadas: M0 sin DTW, M2 con fire confirmado, M4 control continuo D2, M5 sigmoide del delta. M1 se excluyó por equivalencia con M2 tras la corrección del fire; M6/M8 se postergaron.
- Se corrigieron disparos confirmados y rearme: cada evento reinicia el monitor, no población, archivo del mejor, reloj global ni trials ABC. Los puntos iniciales de los gráficos no son disparos.
- En ese contexto el usuario pidió copiar literalmente el estilo del gráfico HRES2 y generar PDF, conservando datos reales. Esa preferencia histórica no reemplaza la aceptación posterior de PNG y PDF para probabilidades.
- Experimento histórico: `woa_abc_version_vieja/resultados/mkp/run_MKP_ddtw_20261021_171227_263707/`, cuatro variantes, mknapcb1 índice 0, semilla 42, población 30, 1000 iteraciones, ventana 70. Las cuatro obtuvieron 24343, brecha 0,155859 % contra 24381; tres adaptativas tuvieron 11 fires, M0 ninguno.
- Publicación histórica: `11661cc` y `8cc99f5` en `feat/binary-woa-abc-dtw`, con 148 pruebas aprobadas en esa entrega.
- Pendiente histórico separado: corrección del piso de theta_delta, retener perfil durante warm-up y cambio de pendiente MKP de 0,1 a 0,2 quedó bloqueada por fallo del ejecutor. No afirmar que se implementó por haberse publicado otros cambios. Confirmar estado actual antes de retomarla.

## Presentación de series temporales

Memorias 393 y 399:

- Se reorientó la exposición hacia usar la trayectoria del mejor objetivo para coordinar MH, dejando derivaciones DTW en apéndice.
- La figura conceptual representa ventana, rampa y meseta; no inventa resultados experimentales.
- Para HRES2 el monitor recibe -LCOE aunque la exposición muestre LCOE descendente; no copiar ese cambio de signo al MKP que maximiza beneficio.
- Publicación histórica de presentación: `cf6d809` en main. El archivo de seguimiento `odd/tasks/time-series-presentation.md` quedó con modificaciones locales adicionales; no incluirlas automáticamente en nuevas publicaciones.

## Plantilla de artículo (otra sesión, solo memoria recuperada)

Memoria 3531 informa que, por pedido del usuario, se conservaron solo los recursos Elsevier CAS de una columna en `6adf64c78fbcdfbb0016599d/`, retirando archivos de dos columnas. En esa observación no hubo compilación, commit ni push. No se inspeccionó el contenido de esa carpeta al crear esta memoria; cualquier trabajo posterior de la otra sesión puede haber actualizado ese estado.

## Procedencia de las notas recientes

Estas observaciones se recuperaron durante la conversación; los identificadores permiten consultar el detalle mediante Engram si sigue disponible:

| ID | Contenido |
|---:|---|
| 628 | Alcance visual: conservar panel (c), simplificar salidas nuevas |
| 629 | Mejoras del diagnóstico temporal y pruebas |
| 630 | Rutas aplanadas y compatibilidad con campañas anteriores |
| 631 | Aprobación del usuario del diseño temporal |
| 632–633 | Probabilidades por pool y verificación del exportador |
| 634 | Tabla OR-Library, gap y corrección de política del protocolo |
| 989 | Corrida DTW con ventana 20 |
| 3218 | Corrida DDTW con ventana 20 |
| 3307 | Mini-panel por MH y PDF |
| 3309 | Implementación de adaptive_exploratory (registro de progreso, no estado final) |
| 3313 | Resultado del piloto exploratory |
| 3321 | Piloto con tres MH por pool |
| 3517 | Commit/push final y fallo CRLF conocido |
| 3531 | Plantilla CAS de una columna en otra sesión |

Los registros de progreso antiguos pueden decir "pendiente" aunque una memoria posterior confirme finalización. Para publicación prevalece el resultado comprobado 3517: commit `1c87bdd`, rama publicada, sin merge.

## Preferencias operativas útiles

- Conversación en español; documentos del proyecto siguen su convención en español.
- Preservar resultados anteriores y cambios locales ajenos.
- Explicar probabilidades asignadas frente a uso realmente observado.
- No atribuir optimalidad a la tabla de mejores factibles.
- No presentar una sola semilla como evidencia estadística.
- No mezclar selección entre MH, control interno de parámetros e hibridación cooperativa: son mecanismos distintos.
- La propuesta GA–SA y la lista reconstruida de ideas DTW están documentadas en `PROJECT_MEMORY.md` como propuestas, no como implementaciones ni recuerdos históricos confirmados.
