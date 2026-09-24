# Discusión

## Síntesis de los hallazgos

Los experimentos entregan evidencia consistente de que el framework cooperativo
gobernado por DTW/DDTW puede aprovechar un portafolio de metaheurísticas en
problemas discretos, continuos y de ingeniería energética. El principal
beneficio no es dominar siempre a cada algoritmo individual, sino reducir la
dependencia de elegir correctamente un único solucionador desde el comienzo.

Esta distinción es importante. En el problema multidimensional de la mochila
(MKP), el framework acumuló entre 40 y 41 victorias significativas en 63
comparaciones con configuraciones base, pero también tuvo entre 3 y 8 derrotas
significativas, según la configuración. En CEC2022, DTW y DDTW obtuvieron los
mejores rangos agregados de las seis alternativas evaluadas, aunque algunos
métodos especializados fueron mejores en funciones concretas. En HRES2--H2,
ambas variantes alcanzaron un LCOE medio menor que las diez metaheurísticas
individuales, y las comparaciones de distribuciones permanecieron significativas
después de la corrección de Holm.

En conjunto, estos resultados respaldan al framework como un controlador de
portafolio eficaz. No respaldan, en cambio, la afirmación más fuerte de que el
framework deba superar a todos sus algoritmos componentes en todos los
paisajes de búsqueda.

| Dominio | Evidencia principal | Interpretación | Límite del resultado |
|---|---|---|---|
| MKP | DTW--3K obtuvo rango medio 1.22; aumentar el presupuesto redujo el gap medio en 19.3% para DTW y 21.5% para DDTW. | Un presupuesto mayor permite aprovechar mejor la dinámica de cooperación; DTW--3K fue la configuración más fuerte en este conjunto. | Nueve instancias seleccionadas y sin control de rotación fija o sin monitor. |
| CEC2022 | Rangos medios 2.323 (DTW) y 2.358 (DDTW); registros W/T/L de 38/16/6 y 37/19/4. | Ambas variantes son competitivas en paisajes heterogéneos; ninguna modalidad domina de manera uniforme. | Dimensión 10, 12 funciones y cinco metaheurísticas base. |
| HRES2--H2 | LCOE medio 0.267160 CNY/kWh para ambas variantes; comparación directa DTW--DDTW con (p=0.593). | El portafolio llega de forma estable a una región de alta calidad, pero no se detecta una diferencia entre DTW y DDTW. | Un único año sintético fijo y muestras independientes para comparar contra las bases. |

## Adaptación del portafolio entre dominios

Los tres dominios difieren en representación, escala del objetivo, restricciones y
geometría del paisaje. MKP es un problema discreto de maximización con
restricciones, mesetas y vecindades combinatorias. CEC2022 contiene funciones
continuas desplazadas, rotadas, híbridas y de composición. HRES2--H2 es un
problema de dimensionamiento tecnoeconómico de baja dimensión, pero cada
evaluación incluye una simulación de despacho de 8.760 horas y restricciones
estrechas.

Que el framework mantenga un comportamiento competitivo en los tres escenarios
es más informativo que observar una mejora en una sola familia de benchmarks.
La regla de conmutación usa la forma de la trayectoria y umbrales relativos a
la propia ejecución. Esto parece favorecer su portabilidad entre escalas de
fitness. Sin embargo, el diseño actual no aísla el efecto de los percentiles
adaptativos frente al resto de los componentes; por tanto, esta explicación es
plausible, pero no una atribución causal demostrada.

Los registros de victorias, empates y derrotas también muestran dónde es más
útil la cooperación. En MKP, el híbrido superó a PSO en las nueve instancias y
tuvo un comportamiento generalmente fuerte frente a EHO, ACO, GWO y WOA. GA
fue el competidor más difícil: obtuvo tres derrotas significativas frente a
cada campaña DTW y tres o cuatro frente a las campañas DDTW. En otras palabras,
la rotación no elimina el valor de un especialista bien adaptado a una
instancia.

En CEC2022, la ventaja más consistente del híbrido apareció frente a WOA,
mientras que ACO fue la base más competitiva. Estos patrones sugieren que el
framework aporta más cuando sus algoritmos tienen comportamientos de búsqueda
complementarios, y menos cuando uno de ellos ya está muy bien alineado con el
problema.

## MKP: efecto del presupuesto y configuración DTW--3K

Los resultados de MKP ofrecen la separación más clara entre las cuatro
configuraciones del framework. El test global de Friedman fue significativo
(χ²_F = 24,0667; p = 2,419 × 10⁻⁵), y las seis comparaciones pareadas
entre configuraciones permanecieron significativas después de Holm. DTW--3K
obtuvo el mejor rango medio (1.22) y el menor gap normalizado promedio
(1.805%). Le siguieron DDTW--3K (rango 1.78; gap 1.876%), DTW--1K (rango 3.11;
gap 2.237%) y DDTW--1K (rango 3.89; gap 2.391%). DTW--3K tuvo el menor gap
medio en siete de las nueve instancias; DDTW--3K fue mejor en `mknapcb2` y
`mknapcb9`.

El efecto del presupuesto es estadística y prácticamente relevante. Pasar de
1.000 a 3.000 iteraciones redujo el gap promedio en 19.3% para DTW y 21.5%
para DDTW. Esto indica que el híbrido todavía no había agotado su dinámica útil
con 1.000 iteraciones. El número medio de cambios también aumentó: de 7.85 a
26.70 para DTW y de 16.43 a 26.96 para DDTW.

No obstante, frecuencia de cambio y calidad de la solución no deben confundirse.
Los logs muestran que hubo más rotaciones con un presupuesto mayor, pero no
demuestran que cada rotación haya producido una mejora posterior.

En el presupuesto 1K, DDTW cambió aproximadamente el doble de veces que DTW,
pero obtuvo un gap medio mayor. Una explicación posible es que la representación
derivativa sea más sensible a cambios cortos alrededor de mesetas discretas y
active más rotaciones sin mejorar necesariamente el incumbente. DTW conserva la
información del nivel y puede beneficiarse de mesetas largas separadas por saltos
discretos. Esta es una hipótesis de mecanismo, no una conclusión causal. Para
evaluarla se necesitan análisis evento por evento que relacionen cada cambio con
la mejora posterior, además de variar de forma controlada la ventana y la
paciencia.

El mejor desempeño con 3K también implica un costo computacional mayor. Los
tiempos medios fueron 306.94 s para DTW--1K, 756.11 s para DTW--3K, 256.43 s
para DDTW--1K y 682.21 s para DDTW--3K. Estos tiempos combinan presupuesto de
evaluaciones, costo de los solucionadores, condiciones de hardware y monitoreo;
no aíslan el costo de DTW o DDTW. Un estudio de overhead justo debe comparar las
mismas trayectorias de los solucionadores con y sin monitor, usando hardware y
número de evaluaciones controlados.

## CEC2022: competitivo, pero no universal

En CEC2022, el framework generaliza más allá del dominio discreto. DTW y DDTW
obtuvieron los mejores rangos medios agregados entre las seis alternativas:
2.323 y 2.358, respectivamente. Después de Holm, sus registros frente a las
bases fueron 38/16/6 para DTW y 37/19/4 para DDTW en 60 comparaciones.

Los doce tests de Friedman por función fueron significativos en ambas campañas,
lo que confirma que la elección del algoritmo afectó los valores observados.
Sin embargo, la presencia de derrotas y empates muestra que ninguna variante
dominó todo el conjunto.

La dependencia del paisaje también aparece en los promedios descriptivos. DTW
obtuvo el mejor promedio híbrido en F6, F7, F8 y F11; DDTW lo obtuvo en F10 y
F12. En algunas funciones, una metaheurística individual conservó el mejor
promedio. Los mejores runs muestran el rendimiento alcanzable, pero son más
sensibles a la aleatoriedad que las distribuciones de 31 corridas; por ello,
los rangos y las comparaciones pareadas son evidencia más robusta.

La comparación directa DTW--DDTW no fue significativa en ninguna de las doce
funciones después de Holm. DTW tuvo el promedio sin redondear menor en siete
funciones y un gap normalizado medio levemente menor (10.281% frente a
10.501%), pero esto no permite declarar superior a ninguna modalidad. Del mismo
modo, un resultado no significativo no demuestra equivalencia. Para afirmar
intercambiabilidad se necesitaría un diseño de equivalencia o no inferioridad
con un margen práctico definido antes del experimento.

La actividad de cambio fue bastante menor en CEC2022 que en MKP--3K y
HRES2--H2: 6.20 cambios medios para DTW y 6.01 para DDTW, con un rango de 4 a
10. Los conteos brutos no son directamente comparables como tasas porque los
presupuestos y espacios de búsqueda son distintos. Aun así, su variación entre
funciones y dominios es compatible con un disparador dependiente de los datos,
no con una rotación puramente calendárica.

## HRES2--H2: evidencia estadística y relevancia de ingeniería

El caso energético modela un sistema conectado a red compuesto por generación
eólica, fotovoltaica, electrolizador y batería. Cada evaluación simula un año
de 8.760 horas. El vector de decisión tiene cuatro dimensiones mixtas: potencia
eólica, número entero de módulos PEM, potencia de batería y duración discreta
del almacenamiento.

Las dos variantes convergieron prácticamente a la misma región económica. El
LCOE medio fue 0.267160 CNY/kWh y el mejor LCOE 0.267159 CNY/kWh. El LCOH medio
fue 17.631382 CNY/kg para DTW y 17.631312 CNY/kg para DDTW. El mejor diseño
reportado en ambas campañas fue aproximadamente:

- 174.47 MW de potencia eólica;
- 25.53 MW de potencia fotovoltaica;
- 70 MW de electrolizador (14 módulos de 5 MW);
- batería de 50 MW con 4 horas de duración;
- producción anual de hidrógeno cercana a 8,86 × 10⁶ kg.

La restricción AGSR quedó activa o muy cerca de su máximo de 20%. Por tanto,
el resultado no debe describirse como un diseño con un margen amplio de
confiabilidad; el optimizador aprovechó prácticamente todo el excedente de red
permitido por la formulación.

Frente a las diez metaheurísticas individuales, los promedios de LCOE de ambas
variantes híbridas fueron menores. La reanálisis con muestras independientes,
test de Mann--Whitney y corrección de Holm mantuvo significativas las veinte
comparaciones. El registro fue 10/0/0 (victorias/empates/derrotas) desde la
perspectiva del framework en cada variante.

La magnitud de la mejora no fue homogénea. Para DTW, la reducción relativa del
LCOE medio varió aproximadamente entre 0.009% frente a ILS y 6.73% frente a SA.
Para DDTW, varió entre 0.009% frente a ILS y 7.90% frente a SA. La significancia
estadística no implica la misma importancia de ingeniería: frente a ILS y TS la
ventaja absoluta es muy pequeña, mientras que frente a SA es más apreciable.

La comparación directa con las mismas semillas produjo una diferencia media de
solo 7,419 × 10⁻⁷ CNY/kWh y un Wilcoxon bilateral de p = 0,593. Los
datos no detectan una diferencia de LCOE entre DTW y DDTW en este caso. Tampoco
establecen equivalencia formal, especialmente porque los valores se archivaron
con seis decimales y la dispersión entre corridas fue extremadamente pequeña.

La interpretación debe respetar además la frontera del sistema implementado.
El modelo no incluye tanque de almacenamiento de hidrógeno, reconversión con
celda de combustible ni una demanda mínima de H₂. Por ello, estos resultados
validan el optimizador para el problema específico de dimensionamiento y
despacho eólico--PV--electrolizador--batería, no para cualquier microrred de
hidrógeno.

Las conclusiones de ingeniería más robustas exigirán múltiples años
meteorológicos y de precios, escenarios de demanda, trayectorias de costos de
tecnología y restricciones de incertidumbre o confiabilidad.

## Qué puede inferirse de la dinámica de cambios

El número medio de cambios varió de manera importante entre dominios y
presupuestos: 7.85 y 16.43 para DTW--1K y DDTW--1K en MKP; 26.70 y 26.96 con
3K; 6.20 y 6.01 en CEC2022; y 21.00 y 21.68 en HRES2--H2. Esta heterogeneidad
es compatible con un controlador cuyo disparo depende de la trayectoria
reciente, y no con una cantidad fija de iteraciones entre cambios.

La similitud entre DTW y DDTW en CEC2022 y HRES2--H2 sugiere que ambas
representaciones identificaron estructuras de estancamiento comparables en esos
experimentos. La separación mayor observada en MKP--1K apunta, en cambio, a que
la representación de la trayectoria tiene un efecto más visible sobre mesetas
discretas.

Los gráficos de convergencia y de la señal DTW/DDTW son útiles para explicar el
mecanismo: muestran cómo una meseta coincide con una señal compatible con
estancamiento y cómo se producen los cambios de solucionador. Sin embargo,
corresponden a corridas seleccionadas, no a una trayectoria promedio de las 31
corridas. Deben interpretarse como ilustraciones del mecanismo, no como prueba
de que la misma secuencia temporal ocurra siempre.

Los conteos de cambios siguen siendo evidencia diagnóstica, no causal. No
indican por sí solos qué solucionador fue abandonado o activado, si aumentó la
diversidad, cuánto duró el beneficio ni si la mejora habría ocurrido sin el
cambio. Un análisis causal debería comparar la política observada con rotación
periódica, rotación aleatoria, un portafolio sin cambios y los mismos
solucionadores independientes bajo presupuestos equivalentes. También debería
estimar la variación de la tasa de mejora antes y después de cada disparo.

## Rol de la memoria elitista compartida

El framework transfiere la mejor solución encontrada hasta el momento entre
rotaciones. Este diseño protege el incumbente y permite cambiar la dinámica de
búsqueda sin perder el progreso acumulado. Es una explicación plausible de por
qué la cooperación puede diversificar la búsqueda sin sacrificar la calidad
alcanzada, especialmente en MKP y HRES2--H2.

Existe, no obstante, un compromiso. Una inyección elitista demasiado agresiva
puede reducir la diversidad efectiva si todos los solucionadores se
reinicializan demasiado cerca del mismo incumbente. Los resultados muestran el
rendimiento del framework completo, pero no separan los efectos del monitor, la
rotación, la memoria y la diversidad del portafolio. Para atribuir el beneficio
a cada componente se requiere una ablación factorial con y sin memoria, con y
sin disparo adaptativo y con distintos niveles de diversidad.

## Fiabilidad estadística y amenazas a la validez

El protocolo inferencial fortalece las conclusiones en varios aspectos:

- cada configuración usa 31 corridas estocásticas;
- se emplean pruebas basadas en rangos cuando se rechaza la normalidad;
- se usan diseños pareados cuando existen semillas comunes documentadas;
- la corrección de Holm controla el error familiar dentro de las familias de
  comparaciones declaradas;
- la síntesis combina gaps normalizados, rangos por problema y registros
  W/T/L, en lugar de mezclar escalas incompatibles.

Persisten varias limitaciones. Primero, la cobertura es finita: MKP usa una
instancia de cada una de nueve familias, CEC2022 solo se evalúa en dimensión 10
y HRES2--H2 representa un único escenario anual sintético. Segundo, no se varía
sistemáticamente la ventana, la banda de Sakoe--Chiba, los percentiles, el
umbral de falta de mejora ni la paciencia.

Tercero, no todos los archivos de salida conservan las observaciones por corrida
de las metaheurísticas base. En HRES2--H2 se pueden revisar los resúmenes y el
reanálisis Mann--Whitney, pero no reconstruir todos los tamaños de efecto desde
los archivos disponibles. Además, las comparaciones híbrido--base no tienen el
mismo emparejamiento por semilla que la comparación directa DTW--DDTW.

Cuarto, el análisis informa significancia y dirección, pero todavía no presenta
intervalos de confianza bootstrap ni tamaños de efecto no paramétricos
estandarizados para todas las comparaciones. Finalmente, seleccionar el mejor
diseño observado no equivale a validar el método en instancias no vistas ni
constituye una demostración de óptimo global.

La agenda de validación debería, por tanto, incluir:

1. archivar todos los resultados por corrida;
2. preregistrar las familias de comparación y los márgenes de relevancia
   práctica;
3. reportar tamaños de efecto e intervalos de confianza;
4. probar más instancias MKP y más dimensiones CEC;
5. evaluar HRES2--H2 con múltiples escenarios meteorológicos, económicos y de
   demanda;
6. realizar ablaciones controladas de monitor, memoria y rotación;
7. utilizar un conjunto externo o una configuración anidada para separar
   generalización de ajuste específico al benchmark.

## Implicaciones prácticas

Para el escenario MKP estudiado, DTW--3K es la configuración más defendible si
se acepta el costo computacional adicional. Para CEC2022 y HRES2--H2, la
evidencia actual no justifica una preferencia universal entre DTW y DDTW. DTW,
basado en niveles, es competitivo y puede ser más simple de interpretar; DDTW
resulta atractivo cuando la velocidad de mejora es más informativa que el nivel
absoluto del fitness.

En ambos casos, la selección debe validarse en instancias representativas y
considerar explícitamente el costo de evaluar la función objetivo. En HRES2--H2
ese costo es especialmente relevante porque cada candidato implica recorrer un
año horario completo.

## Conclusión de la discusión

La conclusión más sólida es que la rotación de un portafolio activada por la
trayectoria puede ofrecer un rendimiento confiable en dominios distintos sin
exigir que una única metaheurística sea óptima en todos ellos. La evidencia es
más decisiva en la comparación presupuestaria de MKP y en la ventaja
distribucional del híbrido para el caso HRES2--H2. En CEC2022 la conclusión es
más matizada: el framework obtiene el mejor rendimiento agregado, pero los
métodos especializados siguen siendo competitivos en funciones individuales.

Por tanto, la contribución debe formularse como un mecanismo adaptativo de
portafolio con fortaleza empírica multidominio, no como una garantía de
dominancia universal. En HRES2--H2, DTW y DDTW alcanzan prácticamente el mismo
desempeño económico en el escenario probado; esto apoya su comparabilidad, pero
no constituye una prueba formal de equivalencia.

## Archivos de respaldo

- `paper/discussion_final.tex`: discusión original en inglés.
- `paper/analisis_estadistico_ES.tex`: análisis estadístico en español.
- `HRES2-H2/wpeb_model.py`: modelo WPEB y simulación horaria.
- `dtw_stagnation.py`: implementación de DTW/DDTW y regla de estancamiento.
- `resultados finales/run_dtw_hres2/` y `resultados finales/run_ddtw_hres2/`:
  resultados, gráficos y resúmenes de las campañas HRES2--H2.
