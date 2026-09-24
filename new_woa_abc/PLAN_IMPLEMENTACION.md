# Plan de implementación de `new_woa_abc`

## Objetivo y aislamiento

Crear desde cero una WOA--ABC continua para CEC2022/HRES2 y binaria para MKP,
usando
`DTW_optimization-main` solo como referencia conceptual. Todo componente nuevo
vive en esta carpeta y el sistema seguirá funcionando cuando se elimine la
carpeta de referencia.

Quedan fuera del alcance actual MCDP, trayectorias, rotación de múltiples MH,
inyección externa de soluciones y el análisis estadístico de corridas emparejadas.

## Arquitectura implementada

```text
new_woa_abc/
├── config.py                 configuración y perfiles
├── core/
│   ├── profiles.py           parámetro base + factor + valor efectivo
│   ├── woa_abc.py            motor continuo CEC/HRES2
│   └── binary_woa_abc.py     motor binario MKP + LB2
├── dtw/
│   ├── monitor.py            DTW, DDTW, D1, D2, delta y tres condiciones
│   └── strategies.py         A3, A4, B3, B1, histéresis y cuatro estados
├── domains/
│   ├── cec.py                exclusivamente CEC2022
│   ├── hres2.py              exclusivamente HRES2-H2/WPEB
│   └── mkp.py                parser, mkcbres, reparación y pulido MKP
├── runners/
│   ├── run_cec.py            F1--F12; D=10 o D=20 por separado
│   ├── run_hres2.py          corrida y métricas HRES2
│   └── run_mkp.py            familias/instancias MKP seleccionables
├── reporting/                CSV, JSON, TXT, resúmenes y gráficos
├── tests/                    pruebas unitarias del monitor y del motor
├── run_cec.sh
├── run_hres2.sh
└── run_mkp.sh
```

## Decisiones técnicas

- Una única población pasa por WOA y luego ABC en cada iteración.
- Las propuestas de ambas fases usan aceptación greedy.
- ABC conserva la vecindad estándar por coordenada y añade una perturbación
  vectorial escalada al rango, guía suave al mejor y un paso decreciente.
- El momentum usa únicamente el desplazamiento del mejor global entre
  iteraciones; no introduce una MH adicional.
- El mejor global nunca empeora.
- El monitor recibe `-best_cost`, porque el monitor original está definido como
  maximización y CEC/HRES2 son minimización.
- La señal detectada al final de la iteración `t` modifica exclusivamente los
  parámetros de `t+1`; esto evita retroactividad.
- M0 no instancia el monitor.
- Los límites son vectores en el núcleo, por lo que CEC y HRES2 no requieren
  mezclar código de evaluación.
- Las seeds están emparejadas entre variantes: para una misma corrida todas
  parten de la misma seed.
- HRES2 se evalúa contra el mejor LCOE factible encontrado; su `0.30` original
  se trata como referencia histórica, no como óptimo certificado.
- MKP maximiza directamente: DTW recibe `best_profit` y toda solución se repara
  antes de evaluarse.
- LB2 conserva L1/L2 y su calendario G1/G2/G3. DTW solo modifica `a`, `phi`,
  guía, vecindad y límite scout de WOA--ABC.
- La tabla mkcbres de las 270 instancias está dentro del dominio nuevo.
- El pulido del élite es un operador acotado de mejora, no otra MH.

## Correspondencia con `DTW_optimization-main`

| Referencia | Implementación nueva | Estado |
|---|---|---|
| `mkp_common/monitor.py` | `dtw/monitor.py` | implementado localmente |
| Binary-Simple A3 | `M1_fire_d2` | implementado |
| Binary-Complex A4 | `M2_fire_3cond` | implementado |
| Continuous-Simple B3 | `M4_d2_continuous` | implementado |
| Continuous-Complex B1 | `M5_sigmoid_delta` | implementado |
| extensión con histéresis | `M6_hysteresis_woa_abc` | implementado |
| extensión de cuatro estados | `M8_four_state_woa_abc` | implementado |

La referencia resuelve MKP binario con PSO/GA/GWO/DE. Esas MH no se incorporan:
la implementación nueva usa exclusivamente WOA--ABC, LB2 y el mismo sensor.

## Fases

- [x] Monitor DTW/DDTW autocontenido.
- [x] WOA--ABC vanilla y reproducible.
- [x] Siete variantes.
- [x] Adaptadores separados CEC, HRES2 y MKP.
- [x] Ejecutores separados y carpetas únicas por experimento.
- [x] Historiales completos por corrida individual; en campañas estadísticas,
  tabla de todas las corridas y detalle/gráficos solo de la mejor por variante.
- [x] Guía ABC, selección por ranking, paso/momentum y vecindad vectorial.
- [x] Reparación explícita de variables discretas de HRES2.
- [x] Pruebas de humo CEC y HRES2 con una seed.
- [x] Parser de las 270 instancias Chu--Beasley y tabla mkcbres independientes.
- [x] WOA--ABC binaria con LB2, reparación, pulido y parada por óptimo.
- [x] Runner MKP paralelo por instancia, `.sh` HPC de nueve CPU, resultados,
  gráficos y pruebas.
- [ ] Validación experimental de hiperparámetros con las doce funciones D=10.
- [ ] Experimento separado D=20.
- [x] Análisis de seeds emparejadas: Shapiro, Wilcoxon, Holm, Mann--Whitney,
  Friedman, ranking medio y boxplots.
- [ ] Ejecutar la campaña definitiva de 31 seeds.
- [x] Análisis descriptivo e inferencial.
- [x] Guardar gráficos detallados solo de la mejor corrida en la fase
  estadística.

## Criterio para pasar a estadística

Antes de activar 31 corridas se revisarán para las doce funciones D=10:

1. convergencia monótona del mejor global;
2. frecuencia razonable de transiciones/fires;
3. cambios visibles en los parámetros efectivos;
4. igualdad de seed entre variantes;
5. coherencia entre CSV, JSON, TXT y gráficos;
6. factibilidad y métricas físicas correctas en HRES2;
7. factibilidad, gap y tasa de óptimos correctos en MKP.
