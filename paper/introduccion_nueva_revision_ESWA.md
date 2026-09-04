# Introducción revisada para ESWA

Esta es una copia revisada de `introduccion nueva.md`. No modifica el archivo
original. La estructura sigue la lógica de `a.md` —contexto, brecha, propuesta,
configuración experimental y contribuciones—, pero el contenido se ha ajustado
al proyecto de este repositorio.

## English version (manuscript text)

```latex
\section{Introduction}
\label{sec:introduction}

Population-based and trajectory-based metaheuristics are widely used to address complex optimization problems because they can explore large search spaces without requiring derivative information. Their performance depends on maintaining a suitable balance between diversification and intensification: excessive diversification may delay convergence, whereas excessive intensification may cause premature convergence and loss of search diversity \cite{blum2003metaheuristics,talbi2009metaheuristics}. This balance is problem- and run-dependent, since different stages of an optimization process may benefit from different search behaviors \cite{vega2021learning}. The No Free Lunch theorem further indicates that no single optimizer can be expected to dominate across all problem classes \cite{wolpert1997no}. These observations motivate adaptive and hybrid strategies that use information generated during the search to modify the behavior or allocation of optimization procedures online \cite{talbi2002taxonomy,raidl2006unified}.

Existing adaptive approaches use, among other signals, successful parameter histories \cite{song2019review}, estimates of the evolutionary state, diversity measures, stagnation counters, or learning-based feedback. Hybrid metaheuristics additionally combine population-based exploration with trajectory-based intensification \cite{crainic2010parallel,alba2005parallel}, but their coordination raises a central control question: when should the active solver terminate its current search phase and transfer control to another solver? \cite{talbi2021machine_tax} Fixed iteration budgets and manually selected stagnation thresholds provide simple control rules, but their behavior can depend strongly on the objective scale, landscape structure, and algorithm being monitored \cite{elhashash2025hybrid,pei2024learning}. The temporal evolution of the incumbent fitness therefore provides a useful, solver-independent signal for distinguishing ongoing progress from a persistent plateau. Nevertheless, explicit shape-based analysis of this trajectory remains insufficiently explored as a mechanism for coordinating heterogeneous metaheuristics.

Accordingly, this work proposes a trajectory-driven adaptive configuration-control framework in which Dynamic Time Warping (DTW) \cite{berndt1994using} and Derivative Dynamic Time Warping (DDTW) \cite{keogh2001derivative} are employed to analyze patterns in the recent best-so-far fitness trajectory. Rather than independently adapting individual control parameters, each solver operates between exploration- and exploitation-oriented search phases. Information derived from the DTW/DDTW-based trajectory analysis is then used to regulate the transition between these phases: upon detecting a persistent plateau, the orchestrator terminates the current solver epoch and transfers the best feasible solution to the next solver in a rotational pool. The pool comprises population-based solvers (PSO, GWO, WOA, EHO, ACO, and GA) and trajectory-based solvers (SA, TS, ILS, and VNS), whose complementary search behaviors are alternated by the adaptive controller. In this way, DTW/DDTW acts as a search-monitoring mechanism within a broader adaptive controller, allowing solver transitions to respond to patterns observed in the evolving fitness trajectory.

The framework is evaluated in three complementary settings. First, the discrete study considers nine representative Chu--Beasley Multidimensional Knapsack Problem (MKP) instances, spanning $n \in \{100,250,500\}$ items and $m \in \{5,10,30\}$ capacity constraints \cite{chu1998genetic}. Second, the continuous study uses the repository's $D=10$ implementation of the 12-function CEC2022-based benchmark, covering unimodal, basic multimodal, hybrid, and composition landscapes \cite{kumar2022problem}. Third, the engineering study addresses mixed continuous-discrete sizing of a grid-connected wind--photovoltaic--electrolyzer--battery (WPEB) system for green-hydrogen production, using an annual 8,760-hour dispatch simulation. In this case, LCOE is the primary optimization objective, while LCOH, hydrogen production, and the Annual Green Hydrogen Surplus Ratio (AGSR) are reported as additional performance and feasibility indicators; feasible designs must satisfy $\mathrm{AGSR} \leq 0.20$ \cite{li2024capacity,modu2023systematic}. Each campaign uses 31 runs, with corresponding executions treated as paired, and reports descriptive and non-parametric inferential analyses.

The main contributions of this work are as follows:

\begin{itemize}
    \item A trajectory-driven adaptive rotational framework that uses DTW/DDTW analysis of the incumbent fitness trajectory to coordinate transitions between complementary population-based and trajectory-based solvers.

    \item A stagnation-monitoring procedure based on synthetic progress and plateau references, historical percentile thresholds, and a confirmation streak, providing a common control principle for heterogeneous objective functions.

    \item A cross-domain evaluation over discrete MKP instances, a continuous CEC2022-based implementation, and a mixed-variable WPEB engineering case, with the campaign-specific parameters and computational budgets explicitly reported.

    \item A controlled comparison of the complete framework with standalone population-based and trajectory-based baselines, together with a direct assessment of the relative behavior of DTW and DDTW. The analysis emphasizes competitiveness, stability, and practical feasibility rather than unconditional superiority on every problem.
\end{itemize}

The remainder of the article presents the framework and problem formulations, describes the experimental design and statistical protocol, reports the results for the three domains, and discusses the switching behavior, computational-budget effects, limitations, and directions for future validation.
```

## Versión en español de referencia

```latex
\section{Introducción}
\label{sec:introduccion}

Las metaheurísticas poblacionales y de trayectoria se utilizan ampliamente para abordar problemas de optimización complejos, ya que permiten explorar espacios de búsqueda extensos sin requerir información derivativa. Su desempeño depende de mantener un equilibrio adecuado entre diversificación e intensificación: una diversificación excesiva puede retrasar la convergencia, mientras que una intensificación excesiva puede causar convergencia prematura y pérdida de diversidad de búsqueda \cite{blum2003metaheuristics,talbi2009metaheuristics}. Este equilibrio depende del problema y de la ejecución, puesto que distintas etapas del proceso de optimización pueden beneficiarse de comportamientos de búsqueda diferentes \cite{vega2021learning}. Además, el Teorema del No Free Lunch indica que no es esperable que un único optimizador domine en todas las clases de problemas \cite{wolpert1997no}. Estas observaciones motivan estrategias adaptativas e híbridas que utilicen la información generada durante la búsqueda para modificar en línea el comportamiento o la asignación de los procedimientos de optimización \cite{talbi2002taxonomy,raidl2006unified}.

Los enfoques adaptativos existentes utilizan, entre otras señales, historiales de parámetros exitosos \cite{song2019review}, estimaciones del estado evolutivo, medidas de diversidad, contadores de estancamiento o retroalimentación basada en aprendizaje. Las metaheurísticas híbridas combinan adicionalmente exploración poblacional con intensificación mediante trayectorias \cite{crainic2010parallel,alba2005parallel}, pero su coordinación plantea una pregunta central de control: ¿cuándo debe el solver activo terminar su fase de búsqueda y transferir el control a otro solver? \cite{talbi2021machine_tax} Los presupuestos fijos de iteraciones y los umbrales de estancamiento seleccionados manualmente proporcionan reglas sencillas, pero su comportamiento puede depender fuertemente de la escala del objetivo, la estructura del paisaje y el algoritmo monitoreado \cite{elhashash2025hybrid,pei2024learning}. Por ello, la evolución temporal de la aptitud incumbente constituye una señal útil e independiente del solver para distinguir entre progreso activo y una meseta persistente. Sin embargo, el análisis explícito de la forma de esta trayectoria todavía se ha explorado de manera limitada como mecanismo de coordinación de metaheurísticas heterogéneas.

En consecuencia, este trabajo propone un marco de control adaptativo de configuración guiado por la trayectoria en el que se emplean el Alineamiento Temporal Dinámico (DTW) \cite{berndt1994using} y el Alineamiento Temporal Dinámico Derivativo (DDTW) \cite{keogh2001derivative} para analizar patrones en la trayectoria reciente del mejor valor de aptitud obtenido hasta el momento. En lugar de adaptar de forma independiente parámetros de control individuales, cada solver opera entre fases de búsqueda orientadas a la exploración y a la explotación. La información derivada del análisis de trayectoria basado en DTW/DDTW se utiliza para regular la transición entre estas fases: al detectar una meseta persistente, el orquestador finaliza la época del solver actual y transfiere la mejor solución factible al siguiente solver en un pool rotacional. Dicho pool comprende solvers poblacionales (PSO, GWO, WOA, EHO, ACO y GA) y solvers de trayectoria (SA, TS, ILS y VNS), cuyos comportamientos de búsqueda complementarios son alternados por el controlador adaptativo. De esta manera, DTW/DDTW actúa como un mecanismo de monitoreo de la búsqueda dentro de un controlador adaptativo más amplio, permitiendo que las transiciones de solver respondan a patrones observados en la trayectoria de aptitud en evolución.

El marco se evalúa en tres escenarios complementarios. Primero, el estudio discreto considera nueve instancias representativas del Problema de la Mochila Multidimensional (MKP) de Chu--Beasley, con $n \in \{100,250,500\}$ ítems y $m \in \{5,10,30\}$ restricciones de capacidad \cite{chu1998genetic}. Segundo, el estudio continuo utiliza la implementación del repositorio con $D=10$ del benchmark basado en las 12 funciones CEC2022, que abarca paisajes unimodales, multimodales básicos, híbridos y de composición \cite{kumar2022problem}. Tercero, el estudio de ingeniería aborda el dimensionamiento mixto continuo-discreto de un sistema conectado a la red Eólica--Fotovoltaica--Electrolizador--Batería (WPEB) para producir hidrógeno verde, mediante una simulación anual de despacho de 8.760 horas. En este caso, el LCOE es el objetivo principal de optimización, mientras que el LCOH, la producción de hidrógeno y el Annual Green Hydrogen Surplus Ratio (AGSR) se reportan como indicadores adicionales de desempeño y factibilidad; los diseños factibles deben satisfacer $\mathrm{AGSR} \leq 0.20$ \cite{li2024capacity,modu2023systematic}. Cada campaña utiliza 31 corridas, tratando las ejecuciones correspondientes como pareadas, y reporta análisis descriptivos e inferenciales no paramétricos.

Las principales contribuciones de este trabajo son las siguientes:

\begin{itemize}
    \item Un marco rotacional adaptativo guiado por trayectoria que utiliza el análisis DTW/DDTW de la trayectoria de aptitud incumbente para coordinar transiciones entre solvers poblacionales y de trayectoria complementarios.

    \item Un procedimiento de monitoreo de estancamiento basado en referencias sintéticas de progreso y meseta, umbrales históricos por percentiles y una racha de confirmación, que proporciona un principio de control común para funciones objetivo heterogéneas.

    \item Una evaluación multidominio sobre instancias discretas MKP, una implementación basada en CEC2022 para optimización continua y un caso de ingeniería WPEB con variables mixtas, reportando explícitamente los parámetros y presupuestos computacionales de cada campaña.

    \item Una comparación controlada del marco completo frente a líneas base poblacionales y de trayectoria ejecutadas de forma independiente, junto con una evaluación directa del comportamiento relativo de DTW y DDTW. El análisis enfatiza competitividad, estabilidad y factibilidad práctica, no una superioridad incondicional en todos los problemas.
\end{itemize}

El resto del artículo presenta el marco y las formulaciones de los problemas, describe el diseño experimental y el protocolo estadístico, reporta los resultados de los tres dominios y discute el comportamiento de las conmutaciones, los efectos del presupuesto computacional, las limitaciones y las líneas de validación futura.
```

## Observaciones para la versión final del manuscrito

1. `a.md` se utilizó únicamente como modelo de estructura; su contenido corresponde a otro experimento y no se incorporó.
2. La versión original decía que HRES2-H2 minimiza LCOE y LCOH. En el código, LCOE es el objetivo y LCOH se calcula y reporta como métrica secundaria.
3. Se reemplazó “scale-invariant” por “scale-adaptive” porque el monitor usa percentiles históricos, pero no normaliza explícitamente las distancias DTW/DDTW.
4. Se evitó afirmar que la misma configuración funciona sin recalibración: los benchmarks tienen parámetros distintos, especialmente en ventana y paciencia.
5. La implementación continua debe revisarse antes del envío: `section6_discussion_EN.tex` reconoce que usa datos de fallback y que algunas definiciones no coinciden completamente con la implementación oficial CEC2022. Por eso la copia dice “CEC2022-based implementation”.
6. La introducción original citaba varias claves sin `\\bibitem` correspondiente; la copia utiliza únicamente claves presentes en `paper/referencias_bibliograficas.md`.
