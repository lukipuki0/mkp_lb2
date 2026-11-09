# The role of hydrogen in the optimal design of off-grid hybrid renewable energy systems

Ficha de lectura para el estado del arte. PDF: [PDF](./The%20role%20of%20hydrogen%20in%20the%20optimal%20design%20of%20off-grid%20hybrid%20renewable%20energy%20systems.pdf). Entrada bibliográfica: [referencias.bib](../../referencias.bib). Clave BibTeX: `Marocco_2022`.

Etiquetas secundarias: multiobjetivo por ε-constraint, PSO, islas Froan, batería frente a hidrógeno. Sección primaria: sistemas híbridos renovables, dimensionamiento y operación.

## Qué aporta

Marocco, Ferrero, Lanzini y Santarelli (2022) cuantifican el papel del hidrógeno en el diseño óptimo de un HRES aislado. Optimizan LCOE y emisiones de CO₂ con ε-restricción y PSO. PSO es el solucionador, no una metaheurística híbrida ni un controlador aprendido.

## Problema, método y contribución

El caso son las islas Froan (Noruega), en el proyecto REMOTE. Las combinaciones incluyen fotovoltaica, eólica, baterías, hidrógeno y diésel. El hidrógeno usa electrolizador, tanque y pila, con curvas de carga parcial, costes según tamaño y vida útil según uso acumulado. El marco (p. 2) toma demanda, meteorología, costes y restricciones de fiabilidad, almacenamiento, emisiones y tamaño. La ε-restricción genera frentes coste-emisiones; no es un ranking tipo NSGA-II.

## Resultados que el texto sostiene

El resumen (p. 1) afirma que el almacenamiento es clave para reducir la dependencia fósil y que el hidrógeno evita sobredimensionar baterías y renovables por su almacenamiento de largo plazo. En el caso extremo sin diésel, la configuración más barata con baterías e hidrógeno tiene un LCOE de 0,41 €/kWh, alrededor de un 35 % menor que el sistema que solo usa baterías. La conclusión (p. 10) matiza el otro extremo: si el diésel no está limitado, la opción más barata es FV, eólica, baterías y grupo diésel, porque el diésel evita baterías enormes y las baterías reducen el combustible. El hidrógeno no sustituye al diésel en todo el frente; cambia el coste cuando se exige alta penetración renovable o se elimina el fósil.

## Límites

Un archipiélago y un conjunto de costes. Las curvas de carga parcial mejoran el realismo, pero siguen siendo modelos, no medidas de campo de Froan.

Lectura crítica: el 0,41 €/kWh y el 35 % son de ese caso sin diésel, no transferibles a Kerkennah o a Uttarakhand. El paper no dice que haga falta un GA híbrido o DTW para llegar a esa conclusión. Sirve para separar el rol de largo plazo del hidrógeno del rol de corto plazo de la batería, y para no tratar el multiobjetivo como sinónimo de algoritmo evolutivo multiobjetivo.

## Evidencia

Título, autores, *Journal of Energy Storage* 46 (2022) 103893 y DOI en la p. 1. LCOE 0,41 €/kWh en el resumen, p. 1. Marco en la p. 2. Conclusión en la p. 10.
