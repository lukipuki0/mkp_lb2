# Sección 1: Introducción (Manuscrito Extendido - Enfoque en Metaheurísticas Colaborativas)

---

## 1. INTRODUCCIÓN (Introduction)

### 1.1. Contexto, Motivación y Marco de Referencia
La resolución computacional de problemas de optimización de alta complejidad constituye uno de los pilares fundamentales de la investigación operacional, la ingeniería de sistemas y la inteligencia artificial moderna. En la práctica, estos problemas abarcan desde dimensiones discretas y combinatorias de gran escala —como el Problema de la Mochila Multidimensional (*Multidimensional Knapsack Problem*, MKP)— hasta espacios continuos no lineales altamente no convexos (tales como la suite de funciones benchmark CEC2022), alcanzando aplicaciones industriales del mundo real como el dimensionamiento tecno-económico y despacho horario de Microredes Híbridas de Energía Renovable e Hidrógeno Verde (HRES2-H2).

Durante las últimas tres décadas, los algoritmos metaheurísticos han demostrado una notable capacidad para encontrar soluciones sustancialmente cercanas al óptimo global en tiempos computacionales razonables ([Talbi, 2009](https://doi.org/10.1002/9780470496916); [Blum & Roli, 2003](https://doi.org/10.1145/937503.937505)). Sin embargo, el **Teorema del No Free Lunch** (*No Free Lunch Theorem*, NFL) ([Wolpert & Macready, 1997](https://doi.org/10.1109/4235.585893)) demuestra que ningún algoritmo de optimización único es capaz de superar a todos los demás cuando su rendimiento se promedia sobre el conjunto de todos los problemas posibles.

Para superar esta restricción intrínseca, el paradigma de las **Metaheurísticas Colaborativas (Collaborative / Cooperative Metaheuristics - CM)** ([Crainic & Toulouse, 2010](https://doi.org/10.1007/978-1-4419-1665-5_17); [El-Abd & Kamel, 2005](https://doi.org/10.1007/11546241_73); [Alba, 2005](https://doi.org/10.1007/b106656)) propone la integración cooperativa de múltiples algoritmos o agentes de búsqueda autónomos (solvers heterogéneos) que intercambian información diagnóstica y memorias de soluciones para lograr un rendimiento global superior al de cualquier algoritmo individual por separado.

---

### 1.2. Planteamiento del Problema (Problem Statement & Limitations)

A pesar de los importantes avances en el desarrollo de algoritmos metaheurísticos colaborativos, las aproximaciones convencionales presentes en la literatura actual sufren tres limitaciones estructurales severas:

#### A. Dilema del Protocolo de Comunicación ("Cuándo y Cómo Colaborar")
Los marcos colaborativos tradicionales utilizan reglas de intercambio rígidas o frecuencias de comunicación constantes (por ejemplo, migrar o intercambiar la mejor solución cada $K$ iteraciones fijas). 
* Si la frecuencia de comunicación es demasiado alta ($K$ pequeño), la sobrecomunicación destruye la diversidad local de los solvers y genera una convergencia prematura hacia subóptimos.
* Si la frecuencia es demasiado baja ($K$ grande), los algoritmos permanecen atrapados en mesetas estancadas consumiendo evaluaciones de la función objetivo sin aportar mejoras.

#### B. Estancamiento Prematuro y Pérdida de Diversidad Genotípica
Las metaheurísticas poblacionales (ej. PSO, GWO, WOA, EHO, ACO, ABC) exhiben una convergencia acelerada en las fases iniciales, pero sufren pérdida de diversidad a medida que la población evoluciona ([Črepinšek et al., 2013](https://doi.org/10.1145/2480741.2480752)). Sin un orquestador adaptativo capaz de detectar el momento exacto del estancamiento, los solvers individuales desperdician recursos computacionales en regiones sin potencial.

#### C. Rigidez Multidominio en los Pools Colaborativos
La abrumadora mayoría de los marcos cooperativos se diseñan a la medida (*ad-hoc*) para un único dominio de problemas (exclusivamente combinatorio discreto o únicamente continuo). Faltan arquitecturas adaptativas capaces de ajustar sus *pools* de algoritmos colaboradores (poblacionales o de trayectoria) según el tipo de problema (MKP, CEC2022 o HRES2-H2) manteniendo un motor de control unificado.

---

### 1.3. Propuesta de Solución: Framework Colaborativo Adaptativo Guiado por DTW

Para resolver de manera integral las deficiencias identificadas, este trabajo propone un novedoso **Framework Metaheurístico Colaborativo Adaptativo gobernado por un Monitor de Estancamiento basado en Dynamic Time Warping (DTW / DDTW)**.

La arquitectura metodológica se fundamenta en tres pilares interconectados:

```mermaid
graph LR
    SubGraph1[Monitor de Estancamiento DTW/DDTW] -->|Disparo de Conmutación Colaborativa| SubGraph2[Orquestador Adaptativo de Pools]
    SubGraph2 -->|Inyección Elitista de Memoria (x_best)| SubGraph3[Pool Activo de Solvers Colaboradores]
    SubGraph3 -->|Serie Temporal de Convergencia| SubGraph1
```

1. **Monitoreo Continuo de Convergencia por Alineamiento Temporal (DTW/DDTW):**  
   En lugar de emplear reglas estáticas, el algoritmo evalúa continuamente la distancia elástica entre el perfil de fitness reciente y una trayectoria ideal de mejora constante utilizando Dynamic Time Warping ([Berndt & Clifford, 1994](https://www.aaai.org/Papers/Workshops/1994/WS-94-03/WS94-03-031.pdf)) y DDTW ([Keogh & Pazzani, 2001](https://doi.org/10.1145/502512.502515)). La adaptación dinámica de umbrales mediante percentiles históricos ($P_{low}$ y $P_{high}$) permite detectar con alta precisión el estancamiento real y activar la comunicación colaborativa solo cuando es estrictamente necesario.

2. **Orquestación Adaptativa de Pools Colaborativos:**  
   El motor orquestador organiza los solvers según la naturaleza del problema:
   - *Dominio Discreto (MKP) e Industrial (HRES2-H2):* Alternancia cooperativa entre un *Pool Poblacional* (PSO, GWO, WOA, EHO, ACO, ABC) para exploración global y un *Pool de Trayectoria* (ILS, SA) para intensificación local.
   - *Dominio Continuo (CEC2022):* Cooperación adaptativa entre múltiples solvers exclusivamente poblacionales (PSO, GWO, WOA, EHO, ACO, ABC).

3. **Inyección Elitista de Memoria y Transferencia de Conocimiento:**  
   En cada conmutación colaborativa, la mejor solución global alcanzada ($x_{best}$) se inyecta directamente como semilla en la población o vector inicial del nuevo algoritmo activado, maximizando la aceleración cooperativa sin destruir la capacidad de escape.

---

### 1.4. Principales Contribuciones Científicas

Las contribuciones originales de esta investigación se resumen en cuatro puntos principales:

1. **Novedoso Protocolo Colaborativo Adaptativo Basado en DTW/DDTW:**  
   Se introduce por primera vez un mecanismo de comunicación entre metaheurísticas gobernado por el alineamiento elástico de series temporales de fitness, sustituyendo las frecuencias estáticas por una conmutación adaptativa precisa.

2. **Esquema de Inyección Elitista de Memoria entre Solvers:**  
   Se formula un protocolo de transferencia que reutiliza la memoria de búsqueda acumulada ($x_{best}$) para acelerar el reinicio de algoritmos heterogéneos en reposo.

3. **Validación Experimental Multidominio Unificada:**  
   Se demuestra la versatilidad e invariancia del framework aplicándolo sin modificaciones en el motor de control en tres dominios complejos:
   - *Combinatorio Discreto:* Benchmark MKP de Chu & Beasley (Mknapcb).
   - *Continuo Benchmark:* Suite estándar IEEE CEC2022 (F1–F12).
   - *Ingeniería Real:* Dimensionamiento y despacho horario (8,760 h) del sistema HRES2-H2, optimizando LCOE, LCOH y AGSR.

4. **Rigurosa Validación Estadística Inferencial No Paramétrica:**  
   Toda la evaluación experimental cuenta con respaldo inferencial conforme a los estándares de la comunidad ([Derrac et al., 2011](https://doi.org/10.1016/j.swevo.2011.02.002); [García et al., 2010](https://doi.org/10.1007/s10489-008-0159-9)), incluyendo pruebas de normalidad (Shapiro-Wilk), comparación por pares (Wilcoxon Signed-Rank, Mann-Whitney U) y ranking global no paramétrico (Friedman Rank Test).

---

### 1.5. Organización del Documento (Roadmap)

El resto de este artículo se estructura de la siguiente manera:
- **Sección 2 (Estado del Arte y Trabajos Relacionados):** Revisa las taxonomías de metaheurísticas colaborativas e híbridas, métodos de control de comunicación, optimización de sistemas HRES2-H2 y pruebas estadísticas.
- **Sección 3 (Metodología del Framework Colaborativo DTW):** Detalla el algoritmo del monitor DTW/DDTW, los protocolos de pools y el flujo de inyección elitista de memoria.
- **Sección 4 (Formulación Matemática de los Problemas Benchmark):** Describe las ecuaciones y restricciones de MKP, CEC2022 y la simulación físico-económica del sistema HRES2-H2.
- **Sección 5 (Resultados Experimentales y Análisis Inferencial):** Presenta las comparativas tabulares, diagramas de convergencia, boxplots, mapas Gantt de switches y tablas de $p$-valores de Wilcoxon/Friedman.
- **Sección 6 (Conclusiones y Trabajo Futuro):** Sintetiza los hallazgos clave y plantea futuras extensiones del paradigma colaborativo adaptativo.
