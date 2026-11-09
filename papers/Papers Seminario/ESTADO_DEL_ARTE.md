# Estado del arte de trabajo sobre el corpus del seminario

Síntesis para orientar la lectura de 75 estudios con PDF, no un capítulo listo para entregar y no una revisión sistemática de toda la literatura. Cada afirmación de fondo remite a una ficha y a páginas físicas del PDF. El inventario fila por fila está en [README.md](./README.md).

## Cómo usarla

1. Fije el eje (control de la búsqueda, medición, sizing, hidrógeno o caso complementario).
2. Lea la comparación de esa sección y abra la ficha antes de copiar una cifra.
3. Si el dato no está en la ficha, no lo trate como demostrado por este corpus.

| Recurso | Uso |
| --- | --- |
| [README.md](./README.md) | 75 estudios, 4 copias y recorridos cortos |
| [catalogo.json](./catalogo.json) | Rutas, hashes, claves y advertencias |
| [referencias.bib](../referencias.bib) | 81 claves; manda sobre el año del nombre de archivo |

La bibliografía tiene seis entradas originales sin PDF en esta carpeta: `Aydin_2025`, `Biedenkapp_2022`, `Doerr_2021`, `Guo_2024`, `Ming_2024` y `Sun_2021`. No se resumen aquí. La lista histórica de exportaciones Excel sigue en [papers_que_sirven.md](../papers_que_sirven.md); no sustituye a las fichas.

## Qué demuestra el corpus y qué no

Los PDF sostienen diferencias de diseño, de señal y de banco experimental entre trabajos concretos. No sostienen un ranking de calidad ni un orden de LCOE entre plantas distintas. Tampoco sostienen que una combinación “no exista” en la literatura: la ausencia en esta carpeta es una pregunta de búsqueda, no un hueco demostrado.

Conexión posible con un proyecto de control híbrido aplicado a energía, no demostrada por ningún PDF de la carpeta: usar la taxonomía de híbridos para nombrar el controlador, y usar los casos de sizing solo como especificación de objetivos de planta. Ninguna ficha evalúa ese controlador sobre un modelo de microrred.

## Incertidumbres bibliográficas

| Pieza | Qué quedó verificado | Qué no hay que afirmar |
| --- | --- | --- |
| P005 `Beheshti_2013` | Título, autores y revista en el PDF, p. 1. Crossref por título y autor no devolvió el artículo | Un DOI. El campo queda nulo |
| P052 `Blalock_2020` | Proceedings MLSys 2020 en el PDF, p. 1. El preprint arXiv:2003.03033 tiene DOI de preprint 10.48550/arXiv.2003.03033, verificado aparte | Un DOI de proceedings. No se afirma que ningún registro futuro lo publique; solo que no está verificado |
| P038 `IEA_2021` | DOI 10.1787/c8328405-en por metadatos OECD/Crossref (título, subtítulo, autor, fecha). No está impreso en el PDF | Que el 10.1787 de la p. 223 sea este informe: es una referencia NEA |
| `Ding_2025` | Clave retenida. El PDF indica vol. 31, n.º 1, febrero de 2026, pp. 101-124, copyright 2026. Crossref sigue en 2025-08. La entrada bibliográfica usa año 2026 | Otro DOI u otra clave. El año del registro y el del número impreso no se colapsan |
| P043 `CortesLeiss_2023` | Portada: Cortés Leiss, Maxwell y Oré Mónago. DOI impreso 10.25613/42F2-EW17 | La lista DataCite (Leiss y White-Nockleby, sin Maxwell ni Oré Mónago). Crossref devolvió 404. Se citan los autores de portada |
| P027 `Makhmudova_2024` | Portada: Makhmudova y Almufti, con el DOI impreso | Los autores que Crossref añade (Yunusa, Hameed) y que no están en la portada |

Copias exactas, mismo SHA-256, una sola cita: P019 de P018 (`Dau_2018`), P047 de P046 (`Sakoe_1978`) y P077 de P010 (`Vega_2024`). P076 y P002 (`Vega_2021`) son el mismo artículo por título y DOI, pero sus PDF tienen distinto SHA-256; también comparten una sola cita. En P010/P077 el PDF escribe El-Gazhali; el correo del autor es el-ghazali.talbi@univ-lille.fr. Otras advertencias de ficha, no de identidad: P009 es pre-proof; en P016 las cifras de LDP de la conclusión no son legibles y la ficha usa el resumen; en P028 las tablas no se recuperan del extracto; en P036 la prosa de la p. 20 y la Tabla 16 no cuadran en el LPSP del mínimo ACS.

## 1. Fundamentos, taxonomía de híbridos y NFL

Wolpert y Macready muestran que una ventaja promediada sobre todas las funciones del marco se compensa en otra clase: estar bien adaptado es una relación algoritmo–problema, no una propiedad absoluta del algoritmo ([nota](01-fundamentos-e-hibridacion/No%20Free%20Lunch%20Theorems%20for%20Optimization.md), pp. 1-2 y 16). Beheshti y Shamsuddin reseñan poblacionales y cierran en la misma idea pedagógica: no hay dominancia universal ([nota](01-fundamentos-e-hibridacion/A%20Review%20of%20Population-based%20Meta-Heuristic%20Algorithm.md), pp. 1 y 26-28). Eso no elige un híbrido.

Talbi (2002) clasifica el diseño, no la implementación: bajo nivel si un método sustituye una función interna de otro; alto nivel si siguen autocontenidos; relay si van en tubería; teamwork si cooperan ([nota](01-fundamentos-e-hibridacion/A%20Taxonomy%20of%20Hybrid%20Metaheuristics.md), pp. 1-3). La conclusión observa que muchos híbridos reseñados acoplan poblacionales con búsqueda local y que la mayoría son secuenciales (p. 15). En 2013 el mismo autor extiende el vocabulario a metaheurística con programación matemática, con restricciones y con aprendizaje ([nota](01-fundamentos-e-hibridacion/Combining%20metaheuristics%20with%20mathematical%20programming%2C%20constraint%20programming%20and%20machine%20learning.md), pp. 1-2). El grano sigue siendo de encuesta: la p. 43 habla de paisajes de alta dimensión, no de un banco nuevo.

Hiperheurística no es lo mismo que híbrido. Dokeroglu, Kucukyilmaz y Talbi la definen como selección, generación o secuencia de heurísticas, sin recorrer ella misma el espacio de soluciones, y proponen cuatro ejes sin experimento propio ([nota](01-fundamentos-e-hibridacion/Hyper-heuristics%EF%BC%9A%20A%20survey%20and%20taxonomy.md), pp. 1-2 y 12). Drake, Kheiri, Özcan y Burke se limitan a selección posterior a Burke et al. (2013), con barrera de dominio, y critican la evaluación de CHeSC: el presupuesto de 10 minutos nominales no iguala llamadas entre dominios ([nota](01-fundamentos-e-hibridacion/Recent%20advances%20in%20selection%20hyper-heuristics.md), pp. 1-2 y 21).

Métodos de hibridación sin política aprendida, y por eso no intercambiables con LB2: Lale encadena GWO, WOA y AOA por etapas en 10 funciones CEC 2019 ([nota](01-fundamentos-e-hibridacion/GWO-WOA-AOA%EF%BC%9A%20Multistage%20Hybrid%20Metaheuristic%20Optimization%20Approach.md), pp. 1 y 5-7); El-Hashash acopla GA o PSO con Newton o gradiente conjugado, pero las tablas no son legibles en el extracto ([nota](01-fundamentos-e-hibridacion/Hybrid%20Optimization%EF%BC%9A%20Combining%20Global%20and%20Local%20Search%20Algorithms%20for%20Efficient%20Non-Convex%20Optimization.md), pp. 1 y 8-11); Akbulut injerta operadores y momento dentro de ABC, en pre-proof ([nota](01-fundamentos-e-hibridacion/Artificial%20Bee%20Colony%20with%20Momentum-Guided%20Search%20and%20Starfish%20Exploration%20for%20Complex%20and%20Multimodal%20Problems.md), pp. 1-2 y 31-32); Makhmudova y Almufti catalogan marcos multiobjetivo de ingeniería, con autores de portada distintos de Crossref ([nota](01-fundamentos-e-hibridacion/Hybrid%20Metaheuristic%20Frameworks%20for%20Multi-Objective%20Engineering%20Optimization%20Problems.md), pp. 1 y 11-12); Araujo y Batista cambian la migración de un evolutivo distribuido según diversidad, no el operador de movimiento ([nota](01-fundamentos-e-hibridacion/A%20diversity-driven%20migration%20strategy%20for%20distributed%20evolutionary%20algorithms.md), p. 1 y p. 14); Jiang y coautores hacen que GEP busque la ecuación del operador mientras DE evalúa la función ([nota](01-fundamentos-e-hibridacion/A%20framework%20for%20designing%20of%20genetic%20operators%20automatically%20based%20on%20gene%20expression%20programming%20and%20differential%20evolution.md), pp. 1, 5 y 13).

| Trabajo | Qué clasifica o construye | Qué no hace |
| --- | --- | --- |
| `Wolpert_1997` | Teorema sobre promedios de funciones | No recomienda un solver |
| `Talbi_2002` | Alto/bajo nivel y relay/teamwork | No mide calidad |
| `Talbi_2013` | Híbrido con matemática, CP o ML | No entrena un modelo |
| `Dokeroglu_2024`, `Drake_2020` | Hiperheurística de selección y sus límites de evaluación | No coronan una familia con un efecto propio |
| `Lale_2025`, `El_Hashash_2025`, `Akbulut_2026` | Híbrido de operadores o de etapas | No leen trazas para elegir la fase |

Conexión posible, no resultado: para clasificar un balanceador por regresión hay que describir primero qué componentes siguen siendo autocontenidos, cuáles sustituyen funciones internas y cómo cooperan; no alcanza con saber que usa regresión ([nota](01-fundamentos-e-hibridacion/A%20Taxonomy%20of%20Hybrid%20Metaheuristics.md), lectura de la ficha sobre pp. 1-3). La taxonomía no dice cuál clase es mejor.

## 2. Acoplamiento aprendizaje–metaheurística

Hay varias flechas y no son el mismo diseño.

Song, Triguero y Özcan separan cuatro interacciones: aprendizaje que mejora aprendizaje, optimización que mejora aprendizaje, aprendizaje que mejora optimización y optimización que mejora optimización ([nota](02-aprendizaje-y-control-adaptativo/A%20review%20on%20the%20self%20and%20dual%20interactions%20between%20machine%20learning%20and%20optimisation.md), p. 1 y pp. 16-18). Azevedo, Rocha y Pereira revisan híbridos para clustering y clasificación: a menudo un enjambre fija centros o hiperparámetros; el objeto no es diseñar la metaheurística combinatoria ([nota](02-aprendizaje-y-control-adaptativo/Hybrid%20approaches%20to%20optimization%20and%20machine%20learning%20methods%EF%BC%9A%20a%20systematic%20literature%20review.md), p. 1 y pp. 33-35). Abou Houran y coautores son ese segundo uso: Coati ajusta una CNN-LSTM de pronóstico FV/viento ([nota](06-multiobjetivo-y-aplicaciones-complementarias/COA-CNN-LSTM%EF%BC%9A%20Coati%20optimization%20algorithm-based%20hybrid%20deep%20learning%20model%20for%20PV%EF%BC%8Fwind%20power%20forecasting%20in%20smart%20grid%20appl%E2%80%A6.md), pp. 1 y 16).

La flecha inversa, aprendizaje al servicio de la búsqueda, está en tres encuestas que tampoco experimentan: Talbi (2021) ordena por componente —problema, bajo nivel, alto nivel— y separa offline de online ([nota](02-aprendizaje-y-control-adaptativo/Machine%20Learning%20into%20Metaheuristics%EF%BC%9A%20A%20Survey%20and%20Taxonomy.md), pp. 1 y 21); Karimi-Mamaghan y coautores detallan inicialización, operadores, vecindad y patrones de buenas soluciones en combinatoria ([nota](02-aprendizaje-y-control-adaptativo/Machine%20learning%20at%20the%20service%20of%20meta-heuristics%20for%20solving%20combinatorial%20optimization%20problems%EF%BC%9A%20A%20state-of-the-art.md), pp. 1-4); Bolufé-Röhler y Tamayo-Vera dan una taxonomía funcional para no mezclar un modelo que elige operadores con uno que elige algoritmos ([nota](02-aprendizaje-y-control-adaptativo/Machine%20Learning%20for%20Enhancing%20Metaheuristics%20in%20Global%20Optimization%EF%BC%9A%20A%20Comprehensive%20Review.md), pp. 1 y 31). Li y coautores reseñan solo hiperheurísticas cuyo selector es RL, valor o política, y listan falta de teoría de convergencia ([nota](02-aprendizaje-y-control-adaptativo/A%20review%20of%20reinforcement%20learning%20based%20hyper-heuristics.md), pp. 1, 6-8 y 17-20). Calvet y coautores definen otra pieza: la learnheuristic reescribe entradas dinámicas del problema, no los operadores; el ejemplo numérico es un heurístico de juguete, costos 896,86 frente a 791,26, no una política descubierta ([nota](02-aprendizaje-y-control-adaptativo/Learnheuristics%EF%BC%9A%20hybridizing%20metaheuristics%20with%20machine%20learning%20for%20optimization%20with%20dynamic%20inputs.md), pp. 1, 13 y 14).

Conexión posible: clasificar un controlador con esas taxonomías antes de compararlo con un WOA de sizing. La carpeta no demuestra que una clase de la taxonomía domine a las demás.

## 3. Control con política entrenada y control sin ella

La diferencia operable es si hay un modelo que se ajusta con experiencia —aunque sea una recta en línea— o una regla fija, una teoría o una ventana sin parámetros aprendidos.

| Trabajo | Nivel de decisión | Señal | Entrenamiento | Banco que el texto usa | Límite que la ficha marca |
| --- | --- | --- | --- | --- | --- |
| `Vega_2021` | Mezcla de seis esquemas de SHO | Fitness | Regresión lineal en línea, cada umbral β | 15 funciones | No es sizing ni series ([nota](02-aprendizaje-y-control-adaptativo/A%20Learning-Based%20Hybrid%20Framework%20for%20Dynamic%20Balancing%20of%20Exploration-Exploitation%EF%BC%9A%20Combining%20Regression%20Analysis%20and%20Metaheuristics.md), pp. 1, 5-6, 15, 20-21) |
| `Vega_2022` | Tamaño de población | Datos de la búsqueda | Varios modelos lineales | Continuo y MKP | No es el mismo artículo que LB2 ([nota](02-aprendizaje-y-control-adaptativo/Combining%20a%20Population-Based%20Approach%20with%20Multiple%20Linear%20Models%20for%20Continuous%20and%20Discrete%20Optimization%20Problems.md), pp. 1, 20, 24-26) |
| `Vega_2024` | Tamaño de población | Datos dinámicos | Componente de aprendizaje y selector probabilístico | Discretos | Copia P077, no segundo estudio ([nota](02-aprendizaje-y-control-adaptativo/Autonomous%20Parameter%20Balance%20in%20Population-Based%20Approaches%EF%BC%9A%20A%20Self-Adaptive%20Learning-Based%20Strategy.md), pp. 1, 4-5, 29) |
| `Vega_2022a` | Población y parada | Métricas de la corrida | Regresión lineal múltiple | 24 funciones y MKP; base SHO o CSA | Superioridad en MKP solo en instancias que el texto lista ([nota](02-aprendizaje-y-control-adaptativo/Predicting%20population%20size%20and%20termination%20criteria%20in%20metaheuristics%EF%BC%9A%20A%20case%20study%20based%20on%20spotted%20hyena%20optimizer%20and%20crow%20search%20a%E2%80%A6.md), pp. 1-3 y 17-18) |
| `Crawford_2021` | Operadores de binarización | Búsqueda en curso | Q-table | Set covering con WOA y SCA | No da un porcentaje único de mejora ([nota](02-aprendizaje-y-control-adaptativo/Q-Learnheuristics%EF%BC%9A%20Towards%20Data-Driven%20Balanced%20Metaheuristics.md), pp. 1 y 22) |
| `Durgut_2022` | Tres operadores binarios de ABC | Estado agrupado | Q-learning; transfiere la tabla entre corridas del mismo caso | 30 SUKP | La transferencia mejora tiempo más que calidad ([nota](02-aprendizaje-y-control-adaptativo/Transfer%20Learning%20for%20Operator%20Selection%EF%BC%9A%20A%20Reinforcement%20Learning%20Approach.md), pp. 1, 8, 13, 15) |
| `Hao_2025` | Topología del PSO | Búsqueda | Q-learning | 29 funciones CEC2017 | La partícula no la mueve la red ([nota](02-aprendizaje-y-control-adaptativo/Q-Learning-Based%20Multi-Strategy%20Topology%20Particle%20Swarm%20Optimization%20Algorithm.md), pp. 1 y 22-23) |
| `Seyyedabbasi_2021` | Explorar o explotar | Comparación actual/previa | Q-table con premio o castigo | 30 funciones y un brazo | Tres híbridos separados, no un portafolio GWO/WOA ([nota](02-aprendizaje-y-control-adaptativo/Hybrid%20algorithms%20based%20on%20combining%20reinforcement%20learning%20and%20metaheuristic%20methods%20to%20solve%20global%20optimization%20problems.md), pp. 1 y 19) |
| `Maleki_2025` | Actualización de líderes de GWO | Experiencia compartida | RL multiagente | 23 clásicas y CEC2022 | 17/23 y 10/12 son del resumen, no recontados en la ficha ([nota](02-aprendizaje-y-control-adaptativo/Enhancing%20leadership-based%20metaheuristics%20using%20reinforcement%20learning%EF%BC%9A%20A%20case%20study%20in%20grey%20wolf%20optimizer.md), pp. 1 y 4) |
| `Meidani_2022` | Regla HHO, DE o WOA por tramos | Mejora de fitness | Bandit UCB | Paisajes de tipos distintos | Rechaza rasgos de paisaje ([nota](02-aprendizaje-y-control-adaptativo/MAB-OS%EF%BC%9A%20Multi-Armed%20Bandits%20Metaheuristic%20Optimizer%20Selection.md), pp. 1, 2 y 4) |
| `Pei_2024` | Operador, mezclando dos módulos | Offline y online | El módulo online aprende; el peso entre módulos se ajusta | CEC 2017, BBOB 2023, CVRPTW | La experiencia offline puede engañar; un α fijo gana en parte del banco ([nota](02-aprendizaje-y-control-adaptativo/Learning%20from%20Offline%20and%20Online%20Experiences%EF%BC%9A%20A%20Hybrid%20Adaptive%20Operator%20Selection%20Framework.md), pp. 1, 7 y 8) |
| `Kalatzantonakis_2025` | Vecindario de VNS | Trazas etiquetadas | Clasificadores offline; sin recompensa al desplegar | 84 CVRP | PR-AUC 0,94 y la ganancia >30 % dependen de su protocolo ([nota](02-aprendizaje-y-control-adaptativo/An%20offline%20data-driven%20process%20for%20learning%20operator%20selection%20from%20metaheuristic%20search%20traces.md), pp. 1, 4 y 16) |
| `Zhang_2023` | Instante de cambio de fase hacia CMA-ES | Proceso de decisión | Q-learning y DQN | CEC 2014 y 2018 | El agente no optimiza la función ([nota](02-aprendizaje-y-control-adaptativo/Controlling%20Sequential%20Hybrid%20Evolutionary%20Algorithm%20by%20Q-Learning.md), pp. 2-3 y 20) |
| `Ding_2025` | Pareja mutación y (F, CR), 12 acciones | Paisaje de fitness y estado de DE | PPO distribuido | CEC2013/2017 no vistos y rover 60-D | Clave 2025, número impreso 2026; no se copian errores medios ausentes en el pasaje ([nota](02-aprendizaje-y-control-adaptativo/Differential%20Evolution%20with%20Joint%20Adaptation%20of%20Mutation%20Strategies%20and%20Control%20Parameters%20via%20Distributed%20Proximal%20Policy%20Optimization.md), pp. 1, 4, 8, 16 y 20) |
| `Zhou_2024` | Parámetros continuos de un MOEA | Rasgos, distribución y grid-IGD, no el frente exacto | RL profundo con transferencia | 4 DTLZ y 8 WFG, 30 corridas | Si el base ya falla, el envoltorio no lo arregla ([nota](02-aprendizaje-y-control-adaptativo/Parameter%20Control%20Framework%20for%20Multiobjective%20Evolutionary%20Computation%20Based%20on%20Deep%20Reinforcement%20Learning.md), pp. 1, 12 y 14) |
| `Kallestad_2023` | Destrucción e inserción dentro de ALNS | Diferencia de objetivo, entre otras | RL profundo, sin imitar un experto | Cuatro problemas combinatorios | La recompensa sigue pendiente de más prueba, según los autores ([nota](02-aprendizaje-y-control-adaptativo/A%20general%20deep%20reinforcement%20learning%20hyperheuristic%20framework%20for%20solving%20combinatorial%20optimization%20problems.md), pp. 1, 3, 5, 11 y 16) |
| `Carrasco_2026` | Cuál de siete metaheurísticas sigue, con población compartida | Estado de 14 dimensiones | PPO | MKP y set covering | No fusiona operadores. En cb9 la selección queda por delante de round-robin y aleatoria; en set covering queda primera entre once estrategias, en un caso donde la ficha dice que ningún método individual domina ([nota](02-aprendizaje-y-control-adaptativo/Towards%20Autonomous%20Bio-Inspired%20Optimization%EF%BC%9A%20Deep%20Reinforcement%20Learning%20for%20Adaptive%20Metaheuristic%20Orchestration.md), pp. 1 y 24-25) |
| `deCarvalho_2021` | Cuál de cinco MOEA | Definida por cada hiperheurística | Cuatro selectores previos, no un solo DRL | 18 problemas reales multiobjetivo | La ventaja reclamada es cruzada, no de un MOEA en un problema ([nota](06-multiobjetivo-y-aplicaciones-complementarias/Comparative%20Analysis%20of%20Selection%20Hyper-Heuristics%20for%20Real-World%20Multi-Objective%20Optimization%20Problems.md), pp. 1 y 17) |
| `Cai_2023` | Transferencias entre dos poblaciones de DE | Búsqueda previa | Ningún modelo que prediga el operador | 29 funciones | Control explícito, no aprendizaje de política ([nota](02-aprendizaje-y-control-adaptativo/Toward%20explicit%20control%20between%20exploration%20and%20exploitation%20in%20evolutionary%20algorithms%EF%BC%9A%20A%20case%20study%20of%20differential%20evolution.md), pp. 1 y 17) |
| `Lehre_2023` | Tasa de mutación codificada en el genotipo | Teoría, no traza empírica | Autoadaptación, sin red | Dynamic Substring Matching | Paisaje binario artificial; no habla de HRES ni de DTW ([nota](03-dtw-y-analisis-de-busqueda/Self-adaptation%20Can%20Help%20Evolutionary%20Algorithms%20Track%20Dynamic%20Optima.md), pp. 2 y 9 del PDF) |
| `CarlesBou_2023` | Índice de la mutación polinomial en NSGA-II | Estado de la búsqueda multiobjetivo | Regla autoadaptativa, sin red | Problemas donde el resumen supera el 75 % en HV, IGD+ y el tercer indicador | No selecciona algoritmos ([nota](06-multiobjetivo-y-aplicaciones-complementarias/Self-adaptive%20polynomial%20mutation%20in%20NSGA-II.md), pp. 1 y 13) |
| `Fu_2021` | `DE/rand/1` o crow search dentro de MOEA/D | Variación de GD y MS en 7 iteraciones | Regla de ventana, no entrenada | Bancos y 5 motores, población 200 | Caso de mantenimiento, no de energía ([nota](06-multiobjetivo-y-aplicaciones-complementarias/An%20adaptive%20hybrid%20evolutionary%20algorithm%20and%20its%20application%20in%20aeroengine%20maintenance%20scheduling%20problem.md), pp. 1, 4 y 8-10) |

`Doerr_2021`, `Sun_2021`, `Guo_2024`, `Ming_2024`, `Biedenkapp_2022` y `Aydin_2025` están en la bibliografía y en la lista histórica, sin ficha local. No se usan como evidencia de página.

Conexión posible: un proyecto puede situar la regresión en línea (LB2 y sucesores) frente a PPO o Q-learning por nivel de decisión y por costo de entrenar, y frente a TRADE o a la mutación autoadaptativa por ausencia de política. Esta carpeta no contiene la comparación cruzada de esos métodos en un mismo presupuesto.

## 4. DTW, paisajes y medición de exploración

No son equivalentes. Medir una serie, describir un paisaje y contar cuencas responden preguntas distintas.

Sakoe y Chiba optimizan el emparejamiento por programación dinámica para alinear habla: la forma simétrica reconoce mejor que la asimétrica y la restricción de pendiente evita alabeos extremos ([nota](03-dtw-y-analisis-de-busqueda/Dynamic%20Programming%20Algorithm%20Optimization%20for%20Spoken%20Word%20Recognition.md), pp. 1-2 y 7 del PDF). Dau y coautores muestran que el único parámetro práctico de DTW, el ancho w, decide clasificación y clustering, y que el mejor w no se transfiere entre tareas ni entre tamaños del mismo conjunto; el w de clasificación no sirve para clustering ([nota](03-dtw-y-analisis-de-busqueda/Optimizing%20dynamic%20time%20warping%E2%80%99s%20window%20width%20for%20time%20series%20data%20mining%20applications.md), p. 1, pp. 10-12 y 43). P019 es el mismo archivo.

Malan reseña el paso del análisis de paisajes desde herramienta teórica a soporte para explicar algoritmos, predecir desempeño y configurarlos, e incluye paisajes multiobjetivo, de restricciones, dinámicos y de error de redes ([nota](03-dtw-y-analisis-de-busqueda/A%20Survey%20of%20Advances%20in%20Landscape%20Analysis%20for%20Optimisation.md), pp. 1, 10 y 13). Ella misma remite a la sensibilidad al muestreo; no propone una medida nueva. Ravber y coautores definen exploración como caer en otra cuenca que el padre, y explotación como permanecer en la misma, sin precalcular las cuencas; distinguen exploración exitosa, fallida, engañosa y rechazo exitoso en seis problemas continuos y en Lennard-Jones ([nota](03-dtw-y-analisis-de-busqueda/Measuring%20successful%2C%20failed%20and%20deceptive%20exploration%20of%20evolutionary%20algorithms%20based%20on%20attraction%20basins.md), pp. 1 y 39). Lehre y Qin demuestran, en un paisaje dinámico artificial, que una tasa de mutación fija pierde el óptimo móvil con alta probabilidad y que la autoadaptación lo sigue ([nota](03-dtw-y-analisis-de-busqueda/Self-adaptation%20Can%20Help%20Evolutionary%20Algorithms%20Track%20Dynamic%20Optima.md), pp. 2 y 9). Eso no es una distancia entre series.

| Eje | Objeto que mide | Condición | No usar como |
| --- | --- | --- | --- |
| `Sakoe_1978`, `Dau_2018` | Alineación temporal y ancho de warping | Habla; luego >100 conjuntos de minería de series | Medida de exploración de un evolutivo |
| `Malan_2021` | Estructura del paisaje y uso del análisis | Encuesta posterior a 2013 | Sustituto de DTW |
| `Ravber_2026` | Cuenca del hijo respecto del padre | Continuo y Lennard-Jones; seis algoritmos | Optimizador nuevo o warping |
| `Lehre_2023` | Seguimiento de un óptimo que se mueve | Prueba en DSM binario | Resultado empírico de HRES |

Conexión posible: si el proyecto registra trayectorias de fitness, DTW compararía formas de serie; las cuencas de Ravber exigirían poder asignar cuenca, lo que un sizing de caja negra no entrega. Ningún PDF aplica las dos medidas al mismo objeto. Meidani rechaza explícitamente los rasgos de paisaje para su bandit ([nota](02-aprendizaje-y-control-adaptativo/MAB-OS%EF%BC%9A%20Multi-Armed%20Bandits%20Metaheuristic%20Optimizer%20Selection.md), p. 2): es una elección de diseño, no una prueba de que el análisis de paisajes sea inútil.

## 5. Dimensionamiento y operación de HRES

Hay tres familias en la sección 04, más casos de capacidad en la sección 05. Una encuesta no es un solver. Un MILP no es un enjambre. Un modelo probabilístico o una simulación SAM no optimizan la misma decisión.

Thirunavukkarasu, Sawle y Lala ordenan métodos clásicos, de un algoritmo, híbridos y herramientas, y declaran preferencia por IA e híbridos frente a clásicos ([nota](04-hres-dimensionamiento-y-operacion/A%20comprehensive%20review%20on%20optimization%20of%20hybrid%20renewable%20energy%20systems%20using%20various%20optimization%20techniques.md), p. 1 y pp. 27-28). Korovushkin y coautores hacen una revisión PRISMA hasta el 26 de junio de 2025 y describen el giro a multiobjetivo; NSGA-II, MOPSO, GA y PSO aparecen como herramientas citadas, no como ganadores de un experimento propio ([nota](04-hres-dimensionamiento-y-operacion/Modern%20Optimization%20Technologies%20in%20Hybrid%20Renewable%20Energy%20Systems%EF%BC%9A%20A%20Systematic%20Review%20of%20Research%20Gaps%20and%20Prospects%20for%20Decisions.md), pp. 1, 5 y 36). Khan, Yu y Waseem organizan el HRES con electrolizador, pila y tanque; no miden un LCOE nuevo ([nota](04-hres-dimensionamiento-y-operacion/Review%20on%20recent%20optimization%20strategies%20for%20hybrid%20renewable%20energy%20system%20with%20hydrogen%20technologies%EF%BC%9A%20State%20of%20the%20art%2C%20trends%20and%20fu%E2%80%A6.md), pp. 1-3, 36 y 40).

| Trabajo | Decisión | Método | Objetivo que el texto optimiza o revisa | Condición experimental | Por qué no se ordena con los demás |
| --- | --- | --- | --- | --- | --- |
| `FerrerMarti_2013` | Dónde generar y cómo agrupar consumos | MILP | Inversión inicial que cubre demanda | Comunidades de Perú | No reporta LCOE ([nota](04-hres-dimensionamiento-y-operacion/A%20MILP%20model%20to%20design%20hybrid%20wind%E2%80%93photovoltaic%20isolated%20rural%20electrification%20projects%20in%20developing%20countries.md), pp. 1, 4-5 y 7) |
| `Malheiro_2015` | Sizing y despacho horario | MILP | LCOE a 20 años | Lisboa, año horario; referencia 0,2232 €/kWh con los cuatro subsistemas | Moneda, diésel y clima propios ([nota](04-hres-dimensionamiento-y-operacion/Integrated%20sizing%20and%20scheduling%20of%20wind%EF%BC%8FPV%EF%BC%8Fdiesel%EF%BC%8Fbattery%20isolated%20systems.md), pp. 1, 9 y 11) |
| `Gupta_2010` | Despacho y penetración FV | MILP en serie temporal | Coste unitario de generación | Aldeas de Tehri Garhwal; tabla 13 en Rs/kWh | No es metaheurística ([nota](04-hres-dimensionamiento-y-operacion/Steady-state%20modelling%20of%20hybrid%20energy%20system%20for%20off%20grid%20electrification%20of%20cluster%20of%20villages.md), pp. 1-2, 9 y 15) |
| `Tina_2006` | Desempeño de largo plazo, no un despacho optimizado | Convolución de densidades | EIR / EENS | Diferencia 0,5–1,9 % frente a simulación; sin baterías en la conclusión de dimensionamiento | Modelo analítico, no solver de capacidades ([nota](04-hres-dimensionamiento-y-operacion/Hybrid%20solar%EF%BC%8Fwind%20power%20system%20probabilistic%20modelling%20for%20long-term%20performance%20assessment.md), pp. 1 y 9) |
| `Smaoui_2015` | Seis variables de FV/eólico/hidrógeno para desalación | Búsqueda iterativa | Menor inversión que cubre la demanda de agua | Kerkennah | No es metaheurística; no publicar su inversión como LCOE ([nota](04-hres-dimensionamiento-y-operacion/Optimal%20sizing%20of%20stand-alone%20photovoltaic%EF%BC%8Fwind%EF%BC%8Fhydrogen%20hybrid%20system%20supplying%20a%20desalination%20unit.md), pp. 1 y 12-14) |
| `Kamal_2023` | Tamaño de microrred rural con recursos locales | DE, comparado con PSO y GA | COE y tamaño; reportan TNPC | Uttarakhand; resumen 0,14 USD/kWh y déficit nulo en el caso | Sin hidrógeno ([nota](04-hres-dimensionamiento-y-operacion/Optimal%20sizing%20of%20standalone%20rural%20microgrid%20for%20sustainable%20electrification%20with%20renewable%20energy%20resources.md), pp. 1 y 19) |
| `Sun_2022` | Unidades de biorresiduo, FV, electrolizador, tanque y pila | WOA | TNPC sujeto a LPSP | Ardabil, 20 años; almacenamiento de hidrógeno, no batería | COE del caso no se compara con Lisboa ([nota](04-hres-dimensionamiento-y-operacion/Designing%20framework%20of%20hybrid%20photovoltaic-biowaste%20energy%20system%20with%20hydrogen%20storage%20considering%20economic%20and%20technical%20indices%20usin%E2%80%A6.md), pp. 1, 19 y 26-28) |
| `Naderipour_2021` | Potencias y tanque de un HRES remoto | WOA; PSO comparador | NPC sujeto a probabilidad de déficit; reportan COE | Tres climas de Irán; siete combinaciones | LDP de conclusión ilegible; la ficha usa el resumen ([nota](04-hres-dimensionamiento-y-operacion/Comparative%20evaluation%20of%20hybrid%20photovoltaic%2C%20wind%2C%20tidal%20and%20fuel%20cell%20clean%20system%20design%20for%20different%20regions%20with%20remote%20applic%E2%80%A6.md), pp. 1 y 18-19) |
| `Tezer_2025` | Tamaño y gestión juntos | i-NSGA-II frente a MOPSO | ACS, LPSP y energía desperdiciada | Tres climas; tabla de Çanakkale | No usar la p. 20 y la Tabla 16 como si coincidieran ([nota](04-hres-dimensionamiento-y-operacion/Multi-objective%20optimization%20of%20hybrid%20renewable%20energy%20systems%20with%20green%20hydrogen%20integration%20and%20hybrid%20storage%20strategies.md), pp. 1, 20 y 21) |
| `Lorestani_2019` | Microrred eléctrica y térmica | E-PSO y reglas de despacho | Costo anual total | Rafsanjan; vertidos 3 % y 6 % en la configuración completa | Incluye calor y gas, no solo electricidad ([nota](04-hres-dimensionamiento-y-operacion/Optimal%20sizing%20and%20techno-economic%20analysis%20of%20energy-%20and%20cost-efficient%20standalone%20multi-carrier%20microgrid.md), pp. 1, 13 y 14) |
| `Nagapurkar_2019` | Diseño de microrred y, aparte, LCA | GA | LCOE; la huella es análisis, no el objetivo del GA | Tucson, Lubbock, Dickinson; resumen 0,32–0,42 USD/kWh | El rango mezcla ciudades y tamaños ([nota](04-hres-dimensionamiento-y-operacion/Techno-economic%20optimization%20and%20environmental%20Life%20Cycle%20Assessment%20%28LCA%29%20of%20microgrids%20located%20in%20the%20US%20using%20genetic%20algorithm.md), pp. 1 y 16) |
| `Marocco_2022` | Diseño aislado con o sin diésel | PSO y ε-restricción | LCOE y CO₂ | Islas Froan; 0,41 €/kWh es el extremo sin diésel con batería e hidrógeno | No es el LCOE de las otras configuraciones del artículo ([nota](04-hres-dimensionamiento-y-operacion/The%20role%20of%20hydrogen%20in%20the%20optimal%20design%20of%20off-grid%20hybrid%20renewable%20energy%20systems.md), pp. 1, 2 y 10) |
| `Su_2023` | Capacidades de planta de hidrógeno en red | NSGA-III | LCOH, reducción de carbono y energía abandonada | Caso de 20 000 t/año; 27,80 CNY/kg a IRR 8 % en el resumen | LCOH, no LCOE ([nota](05-hidrogeno-y-contexto-energetico/Capacity%20configuration%20optimization%20for%20green%20hydrogen%20generation%20driven%20by%20solar-wind%20hybrid%20power%20based%20on%20comprehensive%20performance%E2%80%A6.md), pp. 1 y 6-7) |
| `Li_2024a` | WPEB utility-scale de lazo abierto | Gradiente sobre dominio factible de 8760 h | LCOE, con restricciones de balance y venta | Por encima de 10 MW; cifras de highlights del caso W190-P10-E95-B30 | No es enjambre ni lazo residencial cerrado ([nota](05-hidrogeno-y-contexto-energetico/Capacity%20optimization%20of%20a%20wind-photovoltaic-electrolysis-battery%20%28WPEB%29%20hybrid%20energy%20system%20for%20power%20and%20hydrogen%20generation.md), p. 1) |
| `Mokhtara_2021` | PV, batería e hidrógeno de campus, más tranvía | PSO y HOMER; ε-constraint | COE, con LPSP, NRU y LHSP | Ouargla; 0,016 y 0,1 USD/kWh en dos lecturas del texto | El segundo escenario cambia porque hay demanda de hidrógeno ([nota](05-hidrogeno-y-contexto-energetico/Design%20optimization%20of%20grid-connected%20PV-Hydrogen%20for%20energy%20prosumers%20considering%20sector-coupling%20paradigm%EF%BC%9A%20Case%20study%20of%20a%20university%20buil%E2%80%A6.md), pp. 1-2, 14 y 16) |

Modelado energético no es optimización. Tina compara una forma cerrada con simulación. Moraga-Contreras y coautores simulan FV residencial en Arica con SAM, sin metaheurística ([nota](05-hidrogeno-y-contexto-energetico/Evolution%20of%20Solar%20Energy%20in%20Chile%EF%BC%9A%20Residential%20Opportunities%20in%20Arica%20and%20Parinacota.md), pp. 1, 21 y 22). La IEA describe una trayectoria de escenario con World Energy Outlook y Energy Technology Perspectives, no un algoritmo de sizing ([nota](05-hidrogeno-y-contexto-energetico/Net%20Zero%20by%202050%EF%BC%9A%20A%20Roadmap%20for%20the%20Global%20Energy%20Sector.md), pp. 4-6).

Conexión posible: el polo exacto (Ferrer-Martí, Malheiro, Gupta) sirve de contraste de formulación, no de competidor de un balanceador de exploración. Un metaheurístico de estos casos (WOA, DE, GA, PSO, NSGA) optimiza la planta con parámetros fijos del solver. Ningún PDF muestra que cambiar en línea esos parámetros mejore el MILP o el LCOE del caso.

## 6. Economía, confiabilidad, emisiones y método multiobjetivo

Los objetivos se repiten —costo, déficit, emisiones— pero el método para tratarlos no. Mavrotas implementa AUGMECON: ε-constraint con holguras para evitar Pareto débil, en programación matemática, con ejemplo que no es energético ([nota](06-multiobjetivo-y-aplicaciones-complementarias/Effective%20implementation%20of%20the%20%CE%B5-constraint%20method%20in%20Multi-Objective%20Mathematical%20Programming%20problems.md), pp. 1 y 9-11). Mokhtara y Marocco usan ε-constraint para volver un solo objetivo o para barrer emisiones, con solvers distintos (sección 5). Maneengam genera el frente de un ruteo multimodal con ε-constraint adaptativo y lo ordena con TOPSIS; el caso es el golfo de Tailandia, no una microrred ([nota](06-multiobjetivo-y-aplicaciones-complementarias/Multi-Objective%20Optimization%20of%20the%20Multimodal%20Routing%20Problem%20Using%20the%20Adaptive%20%CE%B5-Constraint%20Method%20and%20Modified%20TOPSIS%20w%E2%80%A6.md), pp. 1, 14 y 16). de Carvalho compara selectores de MOEA completas en problemas reales ([nota](06-multiobjetivo-y-aplicaciones-complementarias/Comparative%20Analysis%20of%20Selection%20Hyper-Heuristics%20for%20Real-World%20Multi-Objective%20Optimization%20Problems.md), pp. 1 y 17). Carles-Bou solo autoadapta la mutación de NSGA-II ([nota](06-multiobjetivo-y-aplicaciones-complementarias/Self-adaptive%20polynomial%20mutation%20in%20NSGA-II.md), pp. 1 y 13). Zhou envuelve cuatro MOEA con una red de parámetros ([nota](02-aprendizaje-y-control-adaptativo/Parameter%20Control%20Framework%20for%20Multiobjective%20Evolutionary%20Computation%20Based%20on%20Deep%20Reinforcement%20Learning.md), pp. 1 y 12). Fu alterna operadores dentro de MOEA/D para mantenimiento de motores ([nota](06-multiobjetivo-y-aplicaciones-complementarias/An%20adaptive%20hybrid%20evolutionary%20algorithm%20and%20its%20application%20in%20aeroengine%20maintenance%20scheduling%20problem.md), pp. 1 y 8-10).

En planta, Tezer optimiza ACS, LPSP y desperdicio a la vez; Nagapurkar separa el LCOE del GA y la huella de ciclo de vida; Su separa LCOH, carbono y vertido. Esas tripletas no son la misma métrica con otro nombre.

Conexión posible: si el proyecto necesita un frente y no un escalar, AUGMECON y los ε-constraint de los casos energéticos son referencias de formulación, no de control de operadores. No hay en la carpeta un estudio que aplique el mismo ε-constraint a un HRES y a un problema combinatorio con el mismo protocolo.

## 7. Hidrógeno: tecnología, cadena, casos y contexto

Conviene no llamar “paper de hidrógeno” a cuatro cosas distintas.

Tecnología y rutas, sin sizing: Qureshi y coautores comparan electrólisis y sitúan PEM con cifras compiladas en la Tabla 1, no medidas en el artículo ([nota](05-hidrogeno-y-contexto-energetico/Sustainable%20energy%20prospects%EF%BC%9A%20Advancements%20in%20green%20hydrogen%20production%20through%20proton%20exchange%20membrane%20water%20electrolysis.md), pp. 1-2 y 20). Olabi y coautores ordenan colores, costos citados y ODS; 120 MJ/kg y la cuota del 4 % son literatura secundaria ([nota](05-hidrogeno-y-contexto-energetico/Green%20hydrogen%EF%BC%9A%20Pathways%2C%20roadmap%2C%20and%20role%20in%20achieving%20sustainable%20development%20goals.md), pp. 1-2 y 19-20).

Cadena y gestión, no una vivienda: Riera, Lima y Knio separan diseño de cadena y diseño de proceso; el patrón que describen para cadena determinista de un objetivo es MILP, no enjambre ([nota](05-hidrogeno-y-contexto-energetico/A%20review%20of%20hydrogen%20production%20and%20supply%20chain%20modeling%20and%20optimization.md), pp. 1-3 y 17-19). Modu y coautores, de 746 registros a 157 más 19, separan dimensionar equipos de la estrategia de gestión energética ([nota](05-hidrogeno-y-contexto-energetico/A%20systematic%20review%20of%20hybrid%20renewable%20energy%20systems%20with%20hydrogen%20storage%EF%BC%9A%20Sizing%2C%20optimization%2C%20and%20energy%20management%20strategy.md), pp. 1, 4 y 15). Khan reseña el HRES-H₂ sin caso propio ([nota](04-hres-dimensionamiento-y-operacion/Review%20on%20recent%20optimization%20strategies%20for%20hybrid%20renewable%20energy%20system%20with%20hydrogen%20technologies%EF%BC%9A%20State%20of%20the%20art%2C%20trends%20and%20fu%E2%80%A6.md), pp. 1-2 y 40).

Casos de capacidad: Su, Li, Mokhtara, y en la sección 04 Sun, Smaoui, Tezer, Naderipour, Marocco y Nagapurkar. Ya están en la tabla de la sección 5, con solvers y métricas que no se alinean.

Contexto de escenario y de política: la IEA, en el NZE, escribe 530 Mt de hidrógeno y 13 % de la demanda final en las páginas que la ficha marca; son salidas de escenario, no de un optimizador de seminario ([nota](05-hidrogeno-y-contexto-energetico/Net%20Zero%20by%202050%EF%BC%9A%20A%20Roadmap%20for%20the%20Global%20Energy%20Sector.md), pp. 14-15, 76 y 109-110). Cortés Leiss, Maxwell y Oré Mónago preguntan si Chile puede armar hubs competitivos, con foco en Antofagasta, sin experimento de electrólisis ([nota](05-hidrogeno-y-contexto-energetico/Paving%20the%20Road%20for%20Competitive%20Green%20Hydrogen%20Hubs%EF%BC%9A%20Does%20Chile%20Have%20a%20Chance%EF%BC%9F.md), pp. 1-4 y 6). Moraga-Contreras sitúa la meta chilena de al menos 70 % de electricidad renovable en 2050 y un FV residencial en Arica; no dimensiona hidrógeno ([nota](05-hidrogeno-y-contexto-energetico/Evolution%20of%20Solar%20Energy%20in%20Chile%EF%BC%9A%20Residential%20Opportunities%20in%20Arica%20and%20Parinacota.md), pp. 1 y 21-22).

Conexión posible: un capítulo de contexto puede citar IEA, Baker Institute y Arica como marco, y Modu o Riera como mapa de decisiones, sin presentarlos como evidencia de que un controlador de metaheurísticas mejore un hub. La ficha de Khan advierte que documentar híbridos ya publicados para HRES-H₂ no autoriza a decir que falte hibridación.

## 8. Casos complementarios de aprendizaje

Blalock, Gonzalez Ortiz, Frankle y Guttag meta-analizan 81 papers de poda y concluyen que el campo no se compara a sí mismo por falta de bancos y métricas; ShrinkBench es su respuesta metodológica ([nota](06-multiobjetivo-y-aplicaciones-complementarias/What%20is%20the%20State%20of%20Neural%20Network%20Pruning%EF%BC%9F.md), pp. 1-2 y 18). El paralelo útil es ese, no una analogía técnica con GWO. Ahmad y coautores recorren ML probabilístico en redes, materiales y pronóstico; el rango de ahorro del resumen no es un experimento de la revisión ([nota](06-multiobjetivo-y-aplicaciones-complementarias/Data-driven%20probabilistic%20machine%20learning%20in%20sustainable%20smart%20energy%EF%BC%8Fsmart%20energy%20systems%EF%BC%9A%20Key%20developments%2C%20challenges%2C%20and%20f%E2%80%A6.md), pp. 1 y 31-32). El pronóstico de Abou Houran y el ruteo de Maneengam muestran metaheurística o ε-constraint fuera del sizing (secciones 2 y 6).

Conexión posible: el argumento de protocolo común de Blalock sirve para no coronar un controlador con bancos incomparables. No es un resultado sobre energía.

## Preguntas para seguir buscando

No son huecos demostrados. Son preguntas que esta carpeta no cierra.

- ¿Fuera de estos PDF, algún controlador entrenado usa como banco el mismo modelo horario de un MILP de sizing, y no CEC? Aquí no aparece.
- ¿DTW sobre trayectorias de fitness aporta una decisión de operador que las cuencas, o una regla sin modelo como TRADE, no aportan? Los ejes no se cruzan en el corpus.
- ¿PPO (Carrasco, Ding, Kallestad) se amortiza frente a la regresión en línea de Vega en un presupuesto común de evaluaciones? No hay ese enfrentamiento.
- ¿Los costos de Ardabil, Froan, Lisboa, Kerkennah, Ouargla y las ciudades de EE. UU. son ordenables? No, por moneda, horizonte, componentes y restricción. La pregunta abierta es otra: qué protocolo haría comparable un subconjunto, no cuál paper “gana”.
- ¿El DOI de proceedings de Blalock o el de Beheshti está en un registro no consultado? La ficha los deja sin verificar; no afirma inexistencia.

## Esquema de capítulo propuesto

Borrador de estructura, no texto de tesis.

1. Alcance de la lectura: 75 estudios, cuatro copias, seis claves sin PDF, incertidumbres de la tabla inicial.
2. Qué se puede afirmar de un algoritmo antes del problema: NFL y vocabulario de híbridos e hiperheurísticas.
3. Dónde entra el aprendizaje: flecha hacia la búsqueda, hacia el modelo, o hacia las entradas dinámicas.
4. Control con política entrenada y control sin ella, con la matriz de nivel, señal y banco.
5. Cómo se mide la búsqueda, en eje distinto del warping de series.
6. Formular un HRES: exacto, analítico, metaheurístico y modelado, sin tabla de LCOE cruzada.
7. Hidrógeno como tecnología, como cadena, como caso de capacidad y como contexto chileno o global.
8. Multiobjetivo: método (ε-constraint, MOEA, selector) separado del indicador (costo, déficit, emisiones).
9. Qué queda como conexión posible con el proyecto y qué preguntas siguen abiertas.
10. Matriz viva. Cada paper nuevo entra por la plantilla de abajo, no por un puesto en un ranking.

## Matriz para fichas futuras

Una fila por estudio. No rellenar “mejor” ni “hueco de la literatura”. Copiar cifras solo con la página física del PDF.

| Campo | Qué anotar |
| --- | --- |
| ID y clave | Del catálogo; si no hay PDF, no inventar ficha |
| Tipo | Fundamento, encuesta, método o caso |
| Nivel de decisión | Operador, parámetro, fase, algoritmo completo, capacidad de planta, política pública, ninguno |
| Señal | Fitness, diversidad, paisaje, cuenca, serie temporal, demanda, ninguna |
| Entrenamiento | Ninguno, regla fija, regresión en línea, tabular, supervisado offline, bandit, DRL |
| Objetivo y restricciones | Como los nombra el PDF, sin traducir LCOH a LCOE |
| Método o solver | MILP, GA, WOA, PPO, escenario, revisión |
| Condición experimental | Banco, sitio, horizonte, moneda, número de corridas |
| Páginas | Físicas del PDF usado |
| Límite | El que declara el autor y el de lectura crítica de la ficha |
| Relación con el proyecto | No aplica, o conexión posible no demostrada |

Ejemplo ya leído, no un modelo a imitar: `Vega_2021` es método; decide la mezcla de seis esquemas; señal fitness; regresión lineal en línea; banco de 15 funciones; no aplica a planta; evidencia en pp. 1, 5-6, 15 y 20-21 ([nota](02-aprendizaje-y-control-adaptativo/A%20Learning-Based%20Hybrid%20Framework%20for%20Dynamic%20Balancing%20of%20Exploration-Exploitation%EF%BC%9A%20Combining%20Regression%20Analysis%20and%20Metaheuristics.md)).

## Límites de esta síntesis

Cubre los ejes presentes en los 75 estudios suministrados. No recontó tablas que las fichas dejaron sin cifra. No actualizó Crossref. No trata las cuatro copias como evidencia adicional. Si un número de esta página y el de la ficha discrepan, manda la ficha.
