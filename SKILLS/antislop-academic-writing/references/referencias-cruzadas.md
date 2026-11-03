# Referencias cruzadas internas

Usá este recurso para dar continuidad al argumento y evitar explicar otra vez contenido ya desarrollado. Toda tabla, figura, gráfico, diagrama, algoritmo, ecuación numerada o elemento independiente debe tener al menos una referencia explícita en un párrafo del cuerpo. La mención introduce el elemento o explica su aporte; no se satisface solo con un pie, título, índice o inserción. Esta cobertura obligatoria no exige repetir remisiones por cuota ni obligar al lector a saltar entre secciones para comprender cada párrafo.

## Destino y redacción

- Comprobá que el destino exista en el documento de trabajo y contenga lo que le atribuís. No inventes numeración, etiquetas ni figuras futuras.
- Usá el mecanismo del documento: en Markdown, identificadores coincidentes entre las menciones y los pies, y enlaces o anclas comprobables cuando corresponda; en LaTeX, etiquetas existentes y referencias automáticas. No reemplaces etiquetas por números escritos a mano.
- Distinguí referencias internas de citas bibliográficas: "sección {referencia}" o "figura {referencia}" no son fuentes `[1]`. Una referencia interna no respalda por sí sola una afirmación nueva que requiere evidencia externa.
- Usá "siguiente" o "a continuación" solo si la posición y proximidad son comprobables. Si la disposición puede cambiar, preferí "en la figura {referencia}". No asegures la ubicación final de un flotante LaTeX sin comprobarla.
- Si el destino está previsto pero todavía no existe o no puede resolverse, marcá `PENDIENTE: referencia cruzada` en el borrador. No lo presentes como contenido ya desarrollado.

## Cobertura de elementos

- Identificá cada elemento independiente y localizá el párrafo que lo introduce o interpreta. Expresiones como "los siguientes gráficos" no bastan si no permiten identificar cada figura.
- Mencioná el identificador concreto, por ejemplo "la figura 8a". Una remisión conjunta es válida si identifica sin ambigüedad cada elemento y explica su relación con el texto.
- Si falta un identificador, establecé uno coherente con la serie editorial y comprobá que no colisione con otros. No inventes numeración de una versión LaTeX todavía no preparada.
- Si un elemento pasa a un anexo, actualizá las remisiones desde el cuerpo al nuevo destino. No lo dejes sin conexión ni conserves referencias a ubicaciones o números anteriores.
- La cobertura de referencias no exige volver a incluir citas bibliográficas en los pies cuando la procedencia ya está atribuida de forma inequívoca en la prosa.

## Patrones esquemáticos

Sustituí los campos entre llaves solo con destinos y contenidos comprobados; variá la fórmula según el contexto:

- "En la sección {referencia previa} se detalló {aspecto}; aquí se analiza {aporte nuevo}".
- "Como se muestra en la figura {referencia cercana}, {interpretación pertinente}".
- "El siguiente diagrama, identificado como figura {referencia inmediata}, representa {relación documentada}".

Resumí únicamente lo necesario para enlazar ambos pasajes. No repitas el párrafo al que remitís ni uses la referencia como adorno. Al revisar el documento, comprobá nuevamente los destinos y las expresiones de posición después de reordenar secciones o figuras.
