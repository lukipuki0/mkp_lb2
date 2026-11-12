# P035 — Revisión PRISMA de la optimización de sistemas híbridos renovables

**Título:** Modern Optimization Technologies in Hybrid Renewable Energy Systems: A Systematic Review of Research Gaps and Prospects for Decisions

**Cita:** `Korovushkin_2025` — Vitalii Korovushkin, Sergii Boichenko, Artem Artyukhov, Kamila Ćwik, Diana Wróblewska y Grzegorz Jankowski (2025). *Energies*, 18(17), 4727. DOI: 10.3390/en18174727.

**PDF:** [PDF](./Modern%20Optimization%20Technologies%20in%20Hybrid%20Renewable%20Energy%20Systems%EF%BC%9A%20A%20Systematic%20Review%20of%20Research%20Gaps%20and%20Prospects%20for%20Decisions.pdf) · **Bibliografía:** [referencias.bib](../referencias.bib)

## Qué aporta

Revisión sistemática, con PRISMA 2020, del espacio de optimización de sistemas híbridos renovables (HRES): arquitecturas, formulaciones, objetivos y métodos. No propone ni prueba un algoritmo.

## Problema y método

El diseño y la operación de un HRES mezclan componentes, recurso estocástico y varios objetivos. La búsqueda cubre Scopus, IEEE Xplore y Web of Science, en inglés, con última consulta el 26 de junio de 2025 y ventana aproximada 2014–mediados de 2025. Las cadenas incluyen LCOE, NPC, LPSP, LOLP, EENS y emisiones. El texto formaliza el problema y separa hallazgos de una agenda de investigación.

## Hallazgos del texto

La literatura pasa de un solo objetivo económico o técnico a formulaciones multiobjetivo que añaden ambiente, criterios sociales y resiliencia. NSGA-II y MOPSO son herramientas habituales de frente de Pareto. Los solucionadores más citados siguen siendo metaheurísticas (GA, PSO y variantes de enjambre) por la no linealidad y las variables mixtas. Aparecen híbridos de metaheurística con solver clásico o con sustituto de aprendizaje, redes para pronosticar sol, viento o carga, y refuerzo profundo emergente para la gestión de energía. Los autores separan ese uso de las redes —pronóstico o sustituto— de la hibridación de buscadores. HOMER se critica por optimizar en la práctica un solo objetivo. La imagen de cierre es un “valle de la muerte” entre la simulación y un proyecto financiable y validado en campo.

## Límites

| Origen | Límite |
| --- | --- |
| Autores | Inglés solamente y tres bases; lo declaran límite de cobertura global. La revisión no valida en campo los algoritmos que resume. |
| Lectura crítica | Un hueco que la revisión nombre (software, validación, co-optimización) no es automáticamente el hueco de este seminario. No hay DTW. El refuerzo para EMS no debe contarse como híbrido de metaheurísticas. |

## Uso en el estado del arte

Mapa 2015–2025 para situar a Sun, Malheiro, Tezer y Lorestani: qué objetivo optimizan y con qué clase de solver. Las redes de pronóstico quedan en otra casilla.

Etiquetas secundarias: PRISMA, NSGA-II, HOMER, EMS, valle simulación–campo.

## Evidencia

- PDF p. 1: PRISMA, periodo y giro hacia multiobjetivo.
- PDF p. 5: Tabla 1, bases y fecha de búsqueda.
- PDF p. 36: conclusiones sobre solvers, redes de pronóstico y límite del trabajo.

## Procedencia del archivo

- El nombre de archivo sustituye solo caracteres prohibidos en Windows (:) por equivalentes de ancho completo. El título de la ficha permanece completo.
