---
name: antislop-academic-writing
description: "Trigger: escritura académica, redacción académica, academic writing, antislop académico, sintetizar en tablas, citas IEEE, referencias cruzadas. Reducir carga cognitiva y redacción artificial con prosa concisa, tablas útiles y evidencia trazable."
---

## Activation Contract

Aplicá a tesis, informes y artículos. Reducí carga cognitiva y prosa genérica o artificial. Conservá idioma, terminología y voz; no cambies alcance, metodología ni decisiones sin autorización.

## Hard Rules

- Organizá según el argumento: títulos informativos, párrafos enfocados, agrupación temática y detalles después de sus antecedentes. No impongas respuesta primero, ruta rápida, checklist ni estructuras repetitivas.
- Escribí con concisión, actores claros y ritmo variado. Eliminá relleno, grandilocuencia, anuncios, repeticiones y conclusiones genéricas. Evitá rayas largas, negritas mecánicas y listas forzadas; preservá matices.
- Redactá de manera directa y concluyente cuando el análisis permita sostener una afirmación. Presentá lo que se identifica, se interpreta o se propone sin acumular condicionales ni calificar sistemáticamente todo como candidato, provisional o pendiente. Reservá esas expresiones para incertidumbres o estados que tengan una función concreta en el argumento.
- No traslades al texto académico las reglas de colaboración con el agente. Evitá fórmulas como "Todas siguen siendo propuestas sujetas a aceptación humana" o recordatorios de que el autor debe revisar, aceptar o aprobar lo escrito: esa revisión forma parte del trabajo y no es una limitación de la investigación. Conservá las cautelas que respondan a límites reales de la evidencia, sin inventar resultados, validaciones ni decisiones de alcance. Esto no elimina la revisión humana cuando sea una característica del sistema estudiado o una exigencia metodológica.
- Usá comillas dobles rectas (`"texto"`) para citas textuales y expresiones entrecomilladas, y paréntesis comunes (`(aclaración)`) para incisos. Mantené la simbología clásica por defecto; usá otra solo cuando exista una causa justificada, como una exigencia editorial, una convención de la disciplina o un significado técnico que deba preservarse. Sin esa justificación, no uses comillas angulares ni tipográficas, ni delimitadores ornamentales. Mantené una puntuación natural y sobria: preferí puntos, comas, dos puntos y paréntesis cuando correspondan, sin acumular incisos o símbolos como recurso de estilo. Conservá los corchetes de las referencias bibliográficas (`[1]`, `[1,2]`) y los símbolos con significado técnico; no los sustituyas por comillas o paréntesis.
- No inventes datos ni referencias. No conviertas percepciones en causalidad ni atribuyas trabajo empírico ajeno como propio. Priorizá fuentes originales; declará consulta indirecta si solo comprobaste secundarias.
- Separá y conectá fuente → hallazgo → interpretación → propuesta. Explicitá supuestos y límites para verificar el razonamiento sin reconstruir antecedentes dispersos.
- Toda tabla, figura, gráfico, diagrama, algoritmo, ecuación numerada u otro elemento independiente debe estar referido explícitamente en al menos un párrafo del cuerpo que lo introduce o interpreta. Usá su identificador y explicá qué aporta; el pie, el título, el índice o la inserción de la imagen no cuentan como referencia en prosa. Preferí una mención anterior o próxima a su aparición, sin repetirla por rutina. Aplicá las [reglas y ejemplos de referencias cruzadas](references/referencias-cruzadas.md).
- Citá junto a la afirmación con corchetes exclusivamente numéricos: `[1]`, `[1,2]`. No incluyas secciones, páginas, figuras, años ni autores dentro de ellos. Reutilizá números de la bibliografía global; en documentos nuevos, numerá por primera aparición, incluidas tablas.
- Cada número debe corresponder a una fuente con datos comprobados. Sin respaldo, marcá `PENDIENTE: fuente` en borradores; nunca inventes números ni entradas bibliográficas.

## Decision Gates

Elegí la cita según el contexto; preferí 1 y 2, sin cuotas ni rotación mecánica. Los siguientes son patrones esquemáticos: los campos entre llaves y números ilustrativos requieren respaldo real antes de usarse.

1. **Afirmación integrada:** "{Afirmación respaldada} [1], como {aspectos documentados}". Priorizá la continuidad del argumento.
2. **Autores sin apellidos:** "Los autores [1] proponen {propuesta documentada}". Usá también hallan o analizan según la fuente; no fuerces esta fórmula en cada oración.
3. **Autores nombrados, excepcionalmente:** "{Apellido real} y colaboradores [1] muestran en la tabla {identificador} de su estudio {evidencia pertinente}". Reservalo para discutir explícitamente una sección, tabla o figura concreta de otra obra que sustente el argumento; identificá ese elemento en prosa, fuera del corchete.

Usá tablas solo para comparar, reconocer diferencias o sintetizar; conservá párrafos para razonamiento y matices. Ante evidencia ausente o contradictoria, señalá vacíos o discrepancias sin conjeturas.

Considerá figuras cuando permitan comprender relaciones, estructuras, secuencias o flujos mejor que la prosa, o aporten apoyo visual necesario. No las sugieras por decoración ni para duplicar información.

- Antes de proponer una figura, leé y aplicá los criterios de la skill [figuras-academicas](../figuras-academicas/SKILL.md). Consultarla no autoriza por sí solo a generar archivos.
- Si el pedido no incluye generarlas, dejá un pendiente de borrador: `PENDIENTE - Figura: {descripción}. Ubicación: {lugar en el texto}. Propósito: {aporte al argumento}. Fuentes: {respaldo disponible y faltantes}`. No presentes la figura pendiente como existente ni inventes numeración o referencias cruzadas.
- Si el pedido incluye crearlas, seguí el procedimiento de esa skill en el mismo flujo, sin exigir una invocación manual adicional. Respetá sus entregables y límites; no inventes datos, relaciones ni decisiones para completar la figura.

## Execution Steps

1. Leé fragmento, contexto y fuentes; identificá propósito, evidencia, destinos de referencias internas y pendientes.
2. Elegí prosa, tablas, figuras o una combinación, según el argumento, sin cuotas ni alternancia impuesta. Para figuras, aplicá la consulta y la distinción entre propuesta y generación de Decision Gates.
3. Dale a cada tabla propósito y título descriptivo, encabezados breves, filas comparables y dimensiones equivalentes; mantené celdas concisas. Atribuí por fila o celda, o indicá una fuente común inequívoca. Interpretá sin repetir su contenido en prosa.
4. Verificá respaldo, correspondencia cita-referencia, numeración global, cifras, denominadores, enlaces y límites. Comprobá que cada elemento independiente tenga una mención explícita en los párrafos y que su identificador y destino coincidan; no dejes figuras o tablas huérfanas. Revisá también las remisiones cuando un elemento se renumere o se traslade a un anexo.
5. Revisá naturalidad, concisión, orden y facilidad de verificación. Eliminá las advertencias sobre la aceptación del autor y los condicionales de rutina; comprobá que las conclusiones sean explícitas y que las cautelas restantes respondan a límites reales de la evidencia. Comprobá el uso de comillas dobles rectas y paréntesis comunes por defecto, y que cualquier excepción tenga una causa justificada, sin alterar referencias bibliográficas ni símbolos técnicos; corregí antes de entregar.

## Output Contract

Entregá texto o archivo y comprobaciones breves de claridad, evidencia, citas, tablas y figuras cuando correspondan. Explicitá los límites metodológicos y pendientes sustantivos, no advertencias sobre la revisión o aceptación del autor. Distinguí figuras propuestas de archivos efectivamente generados; no declares revisión visual si no se realizó. Dejá comentarios de revisión fuera del cuerpo, salvo pendientes de borrador. No declares cumplimiento IEEE completo por verificar solo citas numéricas.
