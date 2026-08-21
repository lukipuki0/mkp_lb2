# 7. Conclusiones y Trabajo Futuro

Este artículo presentó un novedoso **Framework Híbrido Rotacional Adaptativo guiado por DTW/DDTW**, diseñado para superar las limitaciones de los esquemas de comunicación estáticos en optimización colaborativa mediante el monitoreo elástico en tiempo real de trayectorias de convergencia.

---

## 7.1. Hallazgos Principales y Contribuciones Científicas

Las principales conclusiones del estudio son:
1. **Convergencia y Calidad de Solución Superior:** En la suite IEEE CEC2022 ($D=10$), el framework superó consistentemente a las siete metaheurísticas individuales (PSO, GWO, WOA, EHO, ACO, ILS, SA), eludiendo trampas de mínimos locales en funciones compuestas multimodales (F11 y F12).
2. **Dimensionamiento Técnico-Económico Óptimo:** En la optimización del sistema renovable HRES2-H2 (8,760 horas), el framework alcanzó los menores costos nivelados ($LCOE = 0.267160 \text{ CNY/kWh}$, $LCOH = 17.6313 \text{ CNY/kg}$) con una **tasa de factibilidad del 100.0\%** bajo la restricción $AGSR \le 20.0\%$.
3. **Superioridad Estadística Demostrada:** Las pruebas no paramétricas confirmaron diferencias altamente significativas: la prueba de Wilcoxon rechazó la hipótesis nula ($p < 0.001$) frente a todos los competidores, y la prueba de Friedman ubicó a las variantes DDTW y DTW en los puestos 1.04 y 1.48 del ranking global.
4. **Mínima Sobrecarga Computacional:** El algoritmo de monitoreo DTW/DDTW consume menos del $0.5\%$ del tiempo total de cómputo gracias al acotamiento por ventana deslizante ($W=30, w=3$).

---

## 7.2. Implicaciones Teóricas y Prácticas

Desde el punto de vista teórico, el trabajo demuestra la validez del alineamiento elástico temporal (DTW/DDTW) como métrica para el control dinámico de metaheurísticas. Desde el punto de vista práctico, ofrece una herramienta robusta y precisa para la planificación de microredes renovables complejas con almacenamiento de hidrógeno.

---

## 7.3. Líneas de Investigación Futuras

Se identifican las siguientes oportunidades de investigación:
* **Extensión a Optimización Multiobjetivo (MOO):** Adaptar el monitor DTW para rastrear la cobertura del frente de Pareto y la hipervolometría en marcos como NSGA-III o MOEA/D.
* **Ejecución Paralela y Distribuida:** Implementar esquemas de comunicación asíncrona en clusters HPC para ejecución distribuida en tiempo real.
* **Auto-sintonización de Hiperparámetros:** Incorporar aprendizaje por refuerzo u optimización bayesiana online para ajustar dinámicamente los umbrales de percentiles ($P_{low}, P_{high}$).
