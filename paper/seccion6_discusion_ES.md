# 6. Discusión y Análisis de Sensibilidad

Esta sección proporciona un análisis profundo de los hallazgos empíricos, enfocándose en los mecanismos subyacentes que impulsan el rendimiento del **Framework Híbrido Rotacional Adaptativo guiado por DTW/DDTW**.

---

## 6.1. Dinámica de Colaboración Adaptativa

Las metaheurísticas colaborativas y de relevo tradicionales dependen de intervalos de comunicación fijos (ej. conmutación cada $K$ iteraciones). Este enfoque estático sufre dos deficiencias fundamentales:
1. **Conmutaciones Prematuras:** Cambiar de solver mientras la metaheurística activa aún genera mejoras significativas desperdicia la inercia exploratoria.
2. **Atrapamiento Prolongado:** Mantener un algoritmo cuando su trayectoria ya se ha estancado desperdicia presupuesto computacional ($NFE$).

El framework propuesto resuelve esta restricción mediante el monitoreo elástico de trayectorias a través de Dynamic Time Warping (DTW) y Derivative DTW (DDTW). Al calcular alineamientos temporales no lineales entre la ventana reciente ($Q$) y una trayectoria monótona de referencia ($R$), el monitor cuantifica los índices de pendiente $D_1$ y $D_2$. Los percentiles adaptativos ($P_{low}, P_{high}$) se escalan dinámicamente según el historial, garantizando que la rotación de solvers se active solo ante un verdadero estancamiento numérico.

En los diagramas de Gantt, se observa que en etapas tempranas ($t < 200$), los algoritmos poblacionales ($\mathcal{P}_{pop}$) ejecutan épocas más largas ($\Delta t \approx 60 - 120$ iteraciones) para cobertura macro-topológica. En explotación tardía ($t > 500$), el monitor detecta la pérdida de gradiente velozmente, activando alternancias ágiles hacia los métodos de trayectoria ($\mathcal{P}_{traj}$) con épocas más cortas ($\Delta t \approx 20 - 45$ iteraciones) para realizar descenso localizado.

---

## 6.2. Rol de la Inyección Elitista de Memoria

Al activar una rotación del solver $\mathcal{M}_A$ al solver $\mathcal{M}_B$, la transferencia de conocimiento es crítica. El operador de memoria elitista inyecta el vector de la mejor solución global ($\mathbf{x}_{best}$) en el estado inicial de $\mathcal{M}_B$:
* **Para algoritmos poblacionales ($\mathcal{P}_{pop}$):** $\mathbf{x}_{best}$ reemplaza al individuo de peor fitness, preservando la diversidad poblacional y anclando la búsqueda cerca de la cuenca prometedora.
* **Para métodos de trayectoria ($\mathcal{P}_{traj}$):** $\mathbf{x}_{best}$ actúa como punto de partida directo, evitando reinicios aleatorios y permitiendo intensificación local inmediata.

Esto elimina la penalización de re-exploración asociada a los cambios de solver.

---

## 6.3. Análisis de Sensibilidad Paramétrica

El rendimiento del monitor DTW/DDTW depende de tres parámetros clave: el tamaño de ventana $W$, el ancho de banda Sakoe-Chiba $w$, y los percentiles históricos ($P_{low}, P_{high}$).

### 6.3.1. Impacto del Tamaño de Ventana $W$
Se evaluaron ventanas $W \in \{15, 30, 50\}$:
* **Ventanas pequeñas ($W = 15$):** Sensibles al ruido estocástico, generando falsos positivos de estancamiento.
* **Ventanas grandes ($W = 50$):** Introducen retardo en la detección, retrasando conmutaciones necesarias entre 15 y 20 iteraciones.
* **Configuración por defecto ($W = 30$):** Ofrece el balance óptimo entre capacidad de respuesta y filtrado de ruido estocástico.

### 6.3.2. Ancho de Banda Sakoe-Chiba $w$
Restringe el camino de alineamiento $|i - j| \le w$:
* **Alineamiento no restringido ($w = \infty$):** Genera distorsiones temporales excesivas.
* **Configuración $w = 3$:** Limita los desplazamientos a pequeños desfases locales, preservando la tendencia real del gradiente con mínima carga computacional.

### 6.3.3. Umbrales de Percentiles ($P_{low}, P_{high}$)
La configuración $P_{low} = 30.0\%$ y $P_{high} = 70.0\%$ crea una envolvente auto-ajustable que elimina la necesidad de sintonización manual específica por problema.

---

## 6.4. Sobrecarga Computacional y Amenazas a la Validez

### 6.4.1. Complejidad Algorítmica
La complejidad de alineamiento DTW/DDTW sobre la ventana $W$ con banda $w$ es $\mathcal{O}(W \cdot w)$. Al ser $W=30$ y $w=3$ constantes pequeñas e independientes de la dimensión $D$ o iteraciones $T_{max}$, el tiempo consumido por el monitor es inferior al $0.5\%$ del tiempo total de ejecución.

### 6.4.2. Amenazas a la Validez
* **Validez Interna:** Mitigada mediante 31 corridas independientes por algoritmo con semillas fijas derivadas de la semilla maestra 42.
* **Validez Externa:** Mitigada mediante pruebas en la suite matemática CEC2022 (F1–F12) y en la simulación física de 8,760 horas del sistema HRES2-H2.
