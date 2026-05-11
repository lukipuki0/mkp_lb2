# Contexto de Implementación: LB2_MKP.ipynb

Este archivo resume la estrategia de autoadaptación implementada en el notebook `LB2_MKP.ipynb` para el algoritmo PSO utilizando métricas de Dynamic Time Warping (DTW).

## El Problema Original
En las versiones anteriores (Original y V1-V8), el `StagnationMonitor` usaba las métricas del DTW únicamente para generar una señal binaria (`fire=True` o `False`). Cuando se detectaba estancamiento, se disparaban intervenciones agresivas, actuando de forma discontinua (como un interruptor).

## La Nueva Estrategia: Modulación Continua (4 Estados)
En lugar de esperar pasivamente a que se confirme el estancamiento absoluto, la nueva "Versión Autoadaptativa" modula los parámetros hiperheurísticos (`inercia`, `coeficiente_cognitivo`, `coeficiente_social`) en cada iteración.

Se utiliza el indicador **Delta ($\Delta$)**, calculado como:
`\Delta = D_1 - D_2`
Donde:
- **$D_1$** (Distancia a la rampa): Qué tan lejos estamos del progreso ideal.
- **$D_2$** (Distancia a constante): Qué tan lejos estamos de un valle plano (estancamiento).

### Los 4 Estados de Operación

Al comparar $\Delta$ contra el umbral estadístico del monitor ($\theta_\Delta$), definimos 4 estados:

1. **Explorar mucho** ($\Delta > \theta_\Delta$)
   - *Diagnóstico:* Estancamiento severo (curva plana).
   - *Acción:* `inercia = 0.9` (salto largo), `c1 = 2.5` (confianza individual alta), `c2 = 0.5` (ignorar al enjambre estancado).

2. **Explorar poco** ($0 \le \Delta \le \theta_\Delta$)
   - *Diagnóstico:* Estancamiento leve (desaceleración del progreso).
   - *Acción:* `inercia = 0.75`, `c1 = 2.2`, `c2 = 1.0` (liberación gradual del enjambre).

3. **Explotar poco** ($-\theta_\Delta \le \Delta < 0$)
   - *Diagnóstico:* Mejora constante (estado base o "warm-up").
   - *Acción:* `inercia = 0.65`, `c1 = 2.0`, `c2 = 2.0` (equilibrio estándar de PSO).

4. **Explotar mucho** ($\Delta < -\theta_\Delta$)
   - *Diagnóstico:* Mejora repentina fuerte (caída libre hacia el óptimo).
   - *Acción:* `inercia = 0.4` (frenado), `c1 = 1.0`, `c2 = 2.8` (colapso gravitacional hacia el nuevo óptimo global).

## Consideraciones sobre la Ventana de Warm-Up

El algoritmo DTW requiere juntar datos antes de empezar a medir distancias fiables. Este periodo inicial se define por el parámetro `window` del `StagnationConfig`.

**¿Qué implica achicar o agrandar el `window`?**

- **Agrandar la ventana (ej. 40 iteraciones):** 
  - *Ventaja:* El cálculo de la tendencia es mucho más robusto frente a "ruido" de pequeñas variaciones. Reduce drásticamente los falsos positivos.
  - *Desventaja:* El sensor es "lento y pesado". Tarda muchas iteraciones en darse cuenta que se estancó o que mejoró, haciendo que el PSO reaccione tarde (inercia estadística).
  - *Warm-up largo:* Durante 40 iteraciones, el PSO operará "a ciegas" en estado base (Explotar poco) antes de que el monitor de la señal de `ready`.

- **Achicar la ventana (ej. 10 iteraciones):** 
  - *Ventaja:* El sensor es extremadamente ágil. Ante el primer signo de aplanamiento, cambiará a "Explorar poco". El periodo de warm-up es muy cortito (10 iteraciones).
  - *Desventaja:* Hipersensibilidad. El algoritmo se pondrá muy nervioso, oscilando caóticamente entre exploración y explotación, lo que a veces le impedirá converger apropiadamente en un buen mínimo local antes de "saltar" lejos de nuevo.

La calibración de esta ventana (actualmente en `20`) es un balance entre reflejos rápidos (agilidad) y diagnóstico certero (estabilidad).
