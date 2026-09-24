# Cambios pendientes en paper_final.tex

Revisión realizada sobre la versión actual de paper/paper_final.tex.

Este archivo incluye únicamente cambios de contenido. Se excluyen la portada, las figuras, las rutas de imágenes y el material final de la plantilla. No se modificó paper_final.tex.

## 1. Resumen: afirmación demasiado general

### Buscar

~~~latex
significantly outperforming standalone baselines
~~~

### Reemplazar por

~~~latex
showing significant improvements over several standalone baselines
~~~

### Por qué

En MKP y CEC2022 también existen empates y derrotas frente a algunos métodos. La redacción actual parece afirmar que el framework supera significativamente a todos los baselines.

## 2. Resumen: frase gramaticalmente incorrecta

### Buscar

~~~latex
alignment provides an effective, less sensitive to changes in objective scale mechanism for coordinating
collaborative metaheuristics.
~~~

### Reemplazar por

~~~latex
alignment provides an effective mechanism that is less sensitive to changes in objective scale for coordinating
collaborative metaheuristics.
~~~

### Por qué

La expresión less sensitive to changes in objective scale está colocada entre el artículo y el sustantivo mechanism, lo que produce una construcción incorrecta en inglés.

## 3. Introducción: no todos los dominios alternan obligatoriamente los dos pools

### Buscar

~~~latex
Rather than independently adapting individual control parameters, each solver operates between exploration- and exploitation-oriented search phases.
~~~

### Reemplazar por

~~~latex
Rather than independently adapting individual control parameters, the controller coordinates exploration- and exploitation-oriented search behavior through the solver pools enabled for each problem domain.
~~~

### Buscar

~~~latex
The pool comprises population-based solvers (PSO, GWO, WOA, EHO, ACO, and GA) and trajectory-based solvers (SA, TS, ILS, and VNS), whose complementary search behaviors are alternated by the adaptive controller.
~~~

### Reemplazar por

~~~latex
The complete portfolio comprises population-based solvers (PSO, GWO, WOA, EHO, ACO, and GA) and trajectory-based solvers (SA, TS, ILS, and VNS). The controller selects from the solver pools enabled for the corresponding problem domain.
~~~

### Por qué

CEC2022 utiliza únicamente el pool poblacional. Por tanto, el artículo no debe afirmar que en todos los experimentos se alternan obligatoriamente los pools poblacional y de trayectoria.

## 4. Contribución de la introducción: corregir la alternancia universal

### Buscar

~~~latex
\item A collaborative rotational architecture that alternates
      population-based exploration ($\mathcal{P}_{pop}$) and
      trajectory-based exploitation ($\mathcal{P}_{traj}$) pools upon
      each validated stagnation event, transferring accumulated search
      progress through an elitist memory injection protocol
      ($\mathbf{x}_{best}^*$).
~~~

### Reemplazar por

~~~latex
\item A collaborative rotational architecture that selects solvers from the
      population-based exploration ($\mathcal{P}_{pop}$) and
      trajectory-based exploitation ($\mathcal{P}_{traj}$) pools according
      to the problem domain and transfers accumulated search progress through
      an elitist memory injection protocol ($\mathbf{x}_{best}^*$).
~~~

### Por qué

La contribución debe ser válida para los tres dominios, incluido CEC2022, donde no se usa el pool de trayectoria.

## 5. Descripción general del framework: espacio y alternancia

### Buscar

~~~latex
configured population($\mathcal{P}_{pop}$)
~~~

### Reemplazar por

~~~latex
configured population ($\mathcal{P}_{pop}$)
~~~

### Buscar

~~~latex
alternates the pool type ($\mathcal{P}_{pop} \leftrightarrow \mathcal{P}_{traj}$),
~~~

### Reemplazar por

~~~latex
alternates between the configured pools when both pools are enabled,
~~~

### Por qué

Falta un espacio antes del paréntesis y la segunda frase vuelve a presentar la alternancia entre ambos pools como una regla universal.

## 6. Descripción del cambio de solver: afirmación causal excesiva y puntuación

### Buscar

~~~latex
which inevitably lead to either premature switching \cite{talbi2016combining}, (interrupting productive search trajectories) or delayed switching (wasting computational evaluations on stalled regions)\cite{akbulut2026artificial}
~~~

### Reemplazar por

~~~latex
which may lead either to premature switching (interrupting productive search trajectories) \cite{talbi2016combining} or to delayed switching (wasting computational evaluations on stalled regions) \cite{akbulut2026artificial}
~~~

### Por qué

Inevitably es una afirmación absoluta que no queda demostrada. Además, la coma antes del paréntesis y la cita pegada al texto están mal ubicadas.

### Buscar

~~~latex
fine grained exploitation
~~~

### Reemplazar por

~~~latex
fine-grained exploitation
~~~

### Por qué

Fine-grained funciona como adjetivo compuesto y debe llevar guion.

## 7. Monitor: explicar el signo común para maximización y minimización

### Buscar

~~~latex
The stagnation detection engine operates on the sliding temporal sequence of best-so-far objective values recorded by the active solver across the most recent $W$ iterations:
~~~

### Reemplazar por

~~~latex
To use a common progress convention across domains, let $q_t=f_t$ for maximization problems and $q_t=-f_t$ for minimization problems. In the following formulation, $x_t$ denotes this monitored score. The stagnation detection engine operates on the sliding temporal sequence of best-so-far monitored scores recorded by the active solver across the most recent $W$ iterations:
~~~

### Por qué

El monitor espera una trayectoria creciente: el código envía el objetivo directamente en maximización y con signo negativo en minimización. Sin esta explicación, la rampa creciente contradice aparentemente los casos CEC2022 y HRES2--H2.

## 8. Pendiente mínima: la ecuación no coincide con el código

### Buscar

~~~latex
where $s_{min}$ is adaptively computed from the instantaneous dynamic range of the window:
\begin{equation}
s_{min} = 0.01 \cdot \frac{|x_W - x_1|}{W}
\label{eq:adaptive_slope}
\end{equation}
~~~

### Reemplazar por

~~~latex
where $s_{min}$ is the configured minimum slope. When the configured value is zero, an adaptive value is computed from the instantaneous dynamic range:
\begin{equation}
s_{min} =
\begin{cases}
s_{\mathrm{cfg}}, & s_{\mathrm{cfg}} > 0,\\
0.01 \cdot \dfrac{\max(1,|x_W-x_1|)}{W}, & s_{\mathrm{cfg}} = 0.
\end{cases}
\label{eq:adaptive_slope}
\end{equation}
~~~

### Por qué

En dtw_stagnation.py se usa el valor configurado cuando es distinto de cero. El cálculo automático se activa solo cuando el valor configurado es cero y emplea max(1, rango).

## 9. Umbrales adaptativos: frase duplicada

### Buscar

~~~latex
Static thresholding is inherently brittle when dealing with multi-domain benchmarks comprising disparate fitness landscapes (e.g., MKP with fitness in the order of $10^4$, CEC2022 in the order of $10^2$, and HRES2-H2 in sub-unit ranges). To improve portability across different objective scales, the framework maintains three cumulative memory buffers:, the framework maintains three cumulative memory buffers: $\mathcal{H}_{D_1}$, $\mathcal{H}_{D_2}$, and $\mathcal{H}_{\Delta}$, which store historical values of $D_1$, $D_2$, and $\Delta$ recorded during the current epoch.
~~~

### Reemplazar por

~~~latex
Static thresholding is inherently brittle when dealing with multi-domain benchmarks comprising disparate fitness landscapes (e.g., MKP with fitness in the order of $10^4$, CEC2022 in the order of $10^2$, and HRES2-H2 in sub-unit ranges). To improve portability across different objective scales, the framework maintains three cumulative memory buffers: $\mathcal{H}_{D_1}$, $\mathcal{H}_{D_2}$, and $\mathcal{H}_{\Delta}$, which store historical values of $D_1$, $D_2$, and $\Delta$ recorded during the current epoch.
~~~

### Por qué

La expresión the framework maintains three cumulative memory buffers aparece dos veces y además quedó un signo de dos puntos antes de una coma.

## 10. Inyección: la estrategia de tres niveles no es común a todas las implementaciones

### Buscar y quitar completo

~~~latex
\item \textbf{Injection into Population Solvers ($M \in \mathcal{P}_{pop}$):} The successor population of size $N_{pop}$ is initialized through a three-tier hybrid strategy:
\begin{align}
\mathbf{x}_1^{(0)} &= \mathbf{x}_{best}^* \label{eq:elite_inj} \\
\mathbf{x}_i^{(0)} &= \mathbf{x}_{best}^* + \mathcal{N}\left(\mathbf{0}, \sigma^2 \mathbf{I}\right) \odot (\mathbf{U} - \mathbf{L}), \quad \forall i \in \left\{2, \dots, \left\lfloor \frac{N_{pop}}{2} \right\rfloor \right\} \label{eq:gauss_inj} \\
\mathbf{x}_j^{(0)} &\sim \mathcal{U}(\mathbf{L}, \mathbf{U}), \quad \forall j \in \left\{\left\lfloor \frac{N_{pop}}{2} \right\rfloor + 1, \dots, N_{pop}\right\} \label{eq:unif_inj}
\end{align}
where $\sigma = 0.05$, and $[\mathbf{L}, \mathbf{U}]$ denotes the problem bounding box. This guarantees strong local exploitation around $\mathbf{x}_{best}^*$ while maintaining background exploratory dispersion.
~~~

### Poner en su lugar

~~~latex
\item \textbf{Injection into Population Solvers ($M \in \mathcal{P}_{pop}$):} At a solver switch, the incumbent elite solution is transferred to the successor solver. The remaining candidates are initialized according to the domain-specific injection policy configured for the corresponding benchmark.
~~~

### Por qué

La inyección gaussiana para la mitad de la población no representa de forma general las implementaciones de MKP, CEC2022 y HRES2--H2. Al quitar las ecuaciones también deben desaparecer las etiquetas eq:elite_inj, eq:gauss_inj y eq:unif_inj, salvo que estén citadas en otra parte.

## 11. Semillas y pruebas estadísticas de HRES2--H2

### Cambio compatible con el código y los datos actuales

### Buscar

~~~latex
\item \textbf{Deterministic Seed Control:} Stochastic initializations were generated using a deterministic master seed sequence, ensuring that all competing methods were evaluated under identical initial sampling conditions.
~~~

### Reemplazar por

~~~latex
\item \textbf{Seed Control:} MKP and CEC2022 used matched deterministic run seeds. In HRES2--H2, the DTW--DDTW comparison used matched seeds, whereas the standalone baseline samples were treated as independent.
~~~

### Buscar

~~~latex
\item \textit{Wilcoxon Signed-Rank Test:} Conducted for paired comparisons between the proposed DTW/DDTW framework and the baseline metaheuristics at a significance level of $\alpha = 0.05$. The run index was used as the pairing factor, and the Holm step-down procedure was applied within each family of simultaneous comparisons. Results are reported as win/tie/loss outcomes from the perspective of the proposed framework.
~~~

### Reemplazar por

~~~latex
\item \textit{Pairwise Non-Parametric Tests:} The two-sided Wilcoxon signed-rank test was used for matched comparisons in MKP and CEC2022 and for the matched DTW--DDTW comparison in HRES2--H2. The two-sided Mann--Whitney $U$ test was used for the independent HRES2--H2 hybrid--baseline comparisons. The Holm step-down procedure was applied within each family of simultaneous comparisons. Results are reported as win/tie/loss outcomes from the perspective of the proposed framework.
~~~

### Por qué

En HRES2-H2/benchmark_hres2.py se reinicia RANDOM_SEED + r dentro del ciclo del framework híbrido, pero no dentro de cada corrida de cada baseline. Además, el artículo declara que el archivo no conserva la correspondencia por corrida de los baselines. Por eso esos datos no se pueden tratar como muestras emparejadas.

### Si quieres afirmar que HRES usó las mismas semillas

No cambies solamente Mann--Whitney por Wilcoxon en el LaTeX. Primero debes volver a ejecutar cada baseline reiniciando la misma semilla para cada índice de corrida, conservar los 31 valores individuales por método y recalcular Wilcoxon y la corrección de Holm. Después deben sustituirse también los valores p y los resultados W/T/L de la tabla estadística.

## 12. Contribución estadística: evitar la afirmación de superioridad universal

### Buscar

~~~latex
\item A rigorous non-parametric inferential protocol based on
      Wilcoxon, Mann--Whitney, and Friedman tests with Holm correction
      over 31 independent runs, confirming statistically significant
      advantages over standalone metaheuristics in all three domains.
~~~

### Reemplazar por

~~~latex
\item A non-parametric inferential protocol based on Wilcoxon,
      Mann--Whitney, and Friedman tests with Holm correction over
      31 stochastic runs, quantifying wins, ties, and losses against
      standalone metaheuristics across the three domains.
~~~

### Por qué

Los resultados no demuestran una ventaja significativa contra cada baseline en cada problema. La nueva frase describe exactamente lo que hacen las pruebas.

## 13. Configuración DTW/DDTW: frase mal formada y parámetros HRES no documentados

### Buscar

~~~latex
For each budget regime, both alignment modalities standard Dynamic Time Warping (DTW, evaluating raw fitness amplitudes) and Derivative Dynamic Time Warping (DDTW, evaluating first-order finite differences) were systematically executed.
~~~

### Reemplazar por

~~~latex
For each budget regime, both alignment modalities---standard Dynamic Time Warping (DTW), which evaluates raw fitness amplitudes, and Derivative Dynamic Time Warping (DDTW), which evaluates first-order finite differences---were systematically executed.
~~~

### Por qué

Faltan signos que separen la explicación de las dos modalidades.

### Añadir después de la tabla de parámetros, si los resultados publicados proceden de los scripts actuales

~~~latex
For HRES2--H2, the DTW and DDTW implementations used $W=40$, $K_{\max}=15$, $s_{\min}=0$ (automatic mode), adaptive thresholds with $P_{low}=30\%$ and $P_{high}=70\%$, and patience values $P=8$ and $P=3$, respectively.
~~~

### Por qué

La tabla general indica $s_{\min}=0.1$ y $P=8$ para 1,000 iteraciones. Sin embargo, run_hres2.py configura DTW con $s_{\min}=0$ y $P=8$, mientras benchmark_hres2.py configura DDTW con $s_{\min}=0$ y $P=3$. Antes de pegar la frase, confirma que esos dos scripts generaron exactamente los resultados del artículo.

## 14. MKP: el número de switches no demuestra que todos fueran productivos

### Buscar

~~~latex
confirming that the adaptive monitor continues to detect productive stagnation events throughout extended search horizons without premature saturation
~~~

### Reemplazar por

~~~latex
showing that the adaptive monitor continued to trigger solver switches during the extended search horizon; the switch counts alone do not establish that every switch was productive or prevented premature saturation
~~~

### Por qué

El número de switches demuestra que hubo activaciones, pero no demuestra por sí mismo que cada activación fuera productiva ni que evitara saturación prematura.

## 15. HRES2--H2: cambiar optimal por best-found

### Buscar

~~~latex
outlines the optimal physical component sizing and annual green hydrogen production
~~~

### Reemplazar por

~~~latex
outlines the best-found physical component sizing and annual green hydrogen production
~~~

### Buscar

~~~latex
\caption{Optimal physical system component sizing and annual green hydrogen production for the HRES2-H2 microgrid.}
~~~

### Reemplazar por

~~~latex
\caption{Best-found physical system component sizing and annual green hydrogen production for the HRES2-H2 microgrid.}
~~~

### Por qué

Una metaheurística permite reportar la mejor solución encontrada, pero no demuestra optimalidad global.

### Buscar

~~~latex
\textbf{LCOH}
~~~

### Reemplazar por

~~~latex
\textbf{Mean LCOH}
~~~

### Por qué

El párrafo de discusión identifica esos valores como medias; el encabezado debe indicar el estadístico reportado.

## 16. HRES2--H2: nueve baselines en las tablas, pero el texto dice diez

Las tablas actuales contienen nueve baselines: GWO, VNS, TS, ACO, ILS, PSO, WOA, EHO y SA. No contienen una fila para ABC. Por consistencia con las tablas actuales, realiza estos reemplazos:

### Reemplazos directos

~~~latex
ten HRES2--H2 comparisons
~~~

por:

~~~latex
nine HRES2--H2 comparisons
~~~

~~~latex
over the ten comparisons in each
~~~

por:

~~~latex
over the nine comparisons in each
~~~

~~~latex
all 20 distributional contrasts
~~~

por:

~~~latex
all 18 distributional contrasts
~~~

~~~latex
10/0/0
~~~

por:

~~~latex
9/0/0
~~~

~~~latex
over the ten independent-sample comparisons
~~~

por:

~~~latex
over the nine independent-sample comparisons
~~~

~~~latex
all ten standalone methods
~~~

por:

~~~latex
all nine standalone methods
~~~

~~~latex
Against the ten standalone methods
~~~

por:

~~~latex
Against the nine standalone methods
~~~

### Por qué

El texto cuenta diez comparaciones, pero las dos tablas HRES muestran solo nueve métodos independientes. El código de benchmark_hres2.py sí enumera ABC como décimo baseline. Si tienes resultados válidos de ABC y quieres conservar el número diez, no hagas estos reemplazos: añade la fila ABC con sus resultados reales en ambas tablas y verifica nuevamente los valores p de Holm. No se deben inventar los valores faltantes.

## Cambios que ya aparecen correctamente aplicados

- El párrafo de CEC2022 ya usa una sola vez best objective values y ahora dice paired Monte Carlo runs.
- La formulación de HRES2--H2 ya indica que se minimiza LCOE y que LCOH es una métrica secundaria.
- La conversión binaria de MKP ya describe la regla LB2; la ecuación sigmoidal anterior ya no está.
- El framework ya enumera diez metaheurísticas.
- La terminación de cada época ya contempla el límite configurado de iteraciones y la detección temprana de estancamiento.
- El historial adaptativo ya se limita a la época actual.
- La condición de cambio ya utiliza el valor configurado de paciencia $P$.
- El nombre del sistema HRES2--H2, la capacidad total fija, el coeficiente fotovoltaico y la definición de AGSR ya fueron corregidos.
- Las expresiones scale-invariant y strict morphological shape-invariance ya no aparecen en el contenido revisado.
- La expresión without manual recalibration permanece únicamente en la parte final que se excluyó de esta revisión por indicación del autor.
