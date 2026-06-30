# Informe: Control Adaptativo de Parámetros Metaheurísticos vía DTW

Este informe detalla la propuesta técnica para convertir las metaheurísticas estáticas actuales en **Metaheurísticas Adaptativas**. 

## El Concepto
Actualmente, el monitor DTW solo se utiliza como un mecanismo de "aborto" (`fire=True -> break`). Sin embargo, el DTW es capaz de proveer un diagnóstico en tiempo real (el estado `delta`) sobre qué tan estancado está el algoritmo.
El objetivo de esta mejora es utilizar ese estado para modificar los hiperparámetros fundamentales de las metaheurísticas "al vuelo", dándoles la capacidad de defenderse del estancamiento antes de ser abortadas.

## Relación de Estados del DTW
Basado en la lógica actual (evaluando el margen `delta` contra los umbrales), el algoritmo atraviesa 4 estados matemáticos:
1. **Explorar mucho:** Estancamiento severo (Línea plana crítica).
2. **Explorar poco:** Estancamiento leve.
3. **Explotar poco:** Progreso constante y saludable.
4. **Explotar mucho:** Mejora gigante repentina.

---

## Propuestas de Implementación por Algoritmo

### 1. PSO (Particle Swarm Optimization) Adaptativo
**Mecanismo:** Ajustar la inercia (`w`) y los coeficientes (`c1`, `c2`).
*   **Estado "Explorar mucho":** Aumentar la inercia drásticamente (ej. `w = 0.90`) para forzar a las partículas a ganar velocidad y romper su trayectoria. Reducir el coeficiente social (`c2`) para que ignoren temporalmente al líder (que está atrapado en el óptimo local).
*   **Estado "Explotar mucho":** Bajar la inercia (ej. `w = 0.40`) para frenar las partículas y obligarlas a hacer una búsqueda local fina en la zona prometedora.

### 2. GA (Algoritmo Genético) Adaptativo
**Mecanismo:** Ajustar la tasa de mutación (`mutation_rate`).
*   **Estado "Explorar mucho":** Cuando la población pierde diversidad (todos se parecen), la tasa de mutación se dispara de su valor normal (`0.04`) a un nivel crítico (ej. `0.25` o `0.30`). Esto inyecta material genético radicalmente nuevo para descubrir nuevas áreas del espacio de soluciones.
*   **Estado "Explotar poco":** Reducir la mutación a niveles mínimos (ej. `0.01`) para que el cruce puro se encargue de perfeccionar los cromosomas actuales.

### 3. SA (Simulated Annealing) Adaptativo
**Mecanismo:** Re-calentamiento dinámico (Re-heating).
*   **Estado "Explorar mucho":** En lugar de dejar que la Temperatura siga bajando irremediablemente según su factor `alpha`, la temperatura sufre un "shock térmico" (se multiplica por 1.5 o 2.0). Esto ablanda el sistema, permitiendo que la probabilidad de aceptar peores soluciones vuelva a subir temporalmente, logrando que el algoritmo trepe fuera del cráter del óptimo local.

### 4. GWO (Grey Wolf Optimizer) Adaptativo
**Mecanismo:** Modulación de la variable de cerco `a`.
*   **Estado "Explorar mucho":** El parámetro `a` dicta si los lobos atacan (explotación) o buscan presa (exploración). En lugar de decrecer linealmente a 0, si se detecta estancamiento, el valor de `a` rebota temporalmente por encima de 1.0 (ej. `a = 1.5`), forzando a los lobos alfa, beta y delta a separarse y buscar en otras áreas del bosque.

---

## Estrategia de Código Recomendada para Implementarlo
Para llevar esto a cualquier algoritmo de la carpeta `mh/` (ej. `pso.py`), los pasos técnicos a codificar serían:

1. Evaluar el `estado` basándose en `delta` y `theta_delta` dentro de la lógica del monitor DTW.
2. Usar un bloque `if/elif` que asigne nuevos valores temporales a variables locales como `inercia_dinamica` o `mutacion_dinamica` basándose en el estado.
3. Usar esas variables dinámicas en las fórmulas matemáticas de la siguiente iteración, en lugar de los `params.*` fijos pasados al inicio.
4. (Opcional) Reintroducir la función `adaptar_G_por_dtw` para que la binarización LB2 también cambie en sintonía con la metaheurística.
