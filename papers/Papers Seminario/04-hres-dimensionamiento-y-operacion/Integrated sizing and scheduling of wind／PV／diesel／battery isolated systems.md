# P030 — Dimensionar y programar un aislado con MILP, no con enjambre

**Título:** Integrated sizing and scheduling of wind/PV/diesel/battery isolated systems

**Cita:** `Malheiro_2015` — André Malheiro, Pedro M. Castro, Ricardo M. Lima y Ana Estanqueiro (2015). *Renewable Energy*, 83, 646–657. DOI: 10.1016/j.renene.2015.04.066.

**PDF:** [PDF](./Integrated%20sizing%20and%20scheduling%20of%20wind%EF%BC%8FPV%EF%BC%8Fdiesel%EF%BC%8Fbattery%20isolated%20systems.pdf) · **Bibliografía:** [referencias.bib](../../referencias.bib)

## Qué aporta

Formula en programación lineal entera mixta el dimensionamiento y la operación horaria de un sistema aislado eólico–FV–diésel–batería, minimizando el LCOE a 20 años. El solver es de programación matemática, no una metaheurística.

## Problema y método

El modelo cubre un año hora a hora, con recurso y demanda variables, y extrapola el costo con un factor de recuperación de capital. El caso usa irradiancia horaria de Lisboa (inclinación 35°) y perfiles de demanda semanal. La sensibilidad cambia la configuración (quitar un subsistema), el precio del diésel, el costo de tecnología, el clima (viento bueno/solar pobre y al revés) y el perfil de carga.

## Hallazgos del texto

En el escenario de referencia la solución más barata usa los cuatro subsistemas: 40 kW eólicos, 161,50 kW FV, 52,8 kW diésel y 558,4 kWh de baterías; 180 844 L de diésel; LCOE 0,2232 €/kWh; punto de equilibrio 4,7 años. Sin eólico, el LCOE sube a 0,2518 €/kWh. El diésel solo queda en 0,5586 €/kWh. Un perfil con más carga entre semana y menos el fin de semana (Load B, +18,2 % en días laborables, misma energía semanal) mantiene la potencia casi igual, sube la batería a 662,7 kWh y el LCOE un 3,5 %, a 0,2311 €/kWh, aunque baja el diésel a 174 086 L.

## Límites

| Origen | Límite |
| --- | --- |
| Autores | El horizonte operativo es un año representativo; el LCOE de 20 años extrapola ese año. No incluyen hidrógeno ni degradación explícita en los resultados citados. |
| Lectura crítica | Un MILP horario no es comparable, sin más, con un WOA que evalúa reglas de energía. La optimalidad es respecto del modelo lineal, no del sistema físico. No hay control adaptativo ni DTW. |

## Uso en el estado del arte

Referencia de dimensionamiento integrado con operación horaria y LCOE, para contrastar metaheurísticas cuando el modelo admite linealización.

Etiquetas secundarias: LCOE, Lisboa, diésel de respaldo, sensibilidad climática.

## Evidencia

- PDF p. 1: MILP, horizonte anual horario y LCOE a 20 años.
- PDF p. 9: Tabla 9, referencia 0,2232 €/kWh y configuraciones incompletas.
- PDF p. 11: Load B, +3,5 % de LCOE, y conclusiones.

## Procedencia del archivo

- El nombre de archivo sustituye solo caracteres prohibidos en Windows (/) por equivalentes de ancho completo. El título de la ficha permanece completo.
