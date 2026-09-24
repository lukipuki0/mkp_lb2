# Explicación detallada de HRES2--H₂ y DTW/DDTW

Este documento sirve para estudiar la presentación. La meta es entender la lógica del sistema antes de memorizar un discurso.

## 1. La idea completa en una sola cadena

La implementación funciona como un ciclo:

```text
metaheurística propone x
        ↓
decodificador transforma x en capacidades válidas
        ↓
modelo HRES simula 8.760 horas
        ↓
se calculan LCOE, LCOH, H₂, AGSR y factibilidad
        ↓
la metaheurística recibe el valor objetivo
        ↓
se guarda la trayectoria del mejor valor
        ↓
DTW/DDTW decide si la búsqueda sigue progresando
        ↓
si hay estancamiento, se cambia de solucionador y se transfiere el incumbente
```

Hay dos partes diferentes que no conviene mezclar:

1. **HRES2--H₂** es el modelo físico-económico. Dice qué significa una solución y cuánto cuesta operarla durante un año.
2. **DTW/DDTW** es el mecanismo de coordinación. No calcula la física del HRES; observa cómo evoluciona el valor objetivo y decide cuándo conviene cambiar de metaheurística.

Una frase útil es:

> HRES evalúa las soluciones; DTW/DDTW evalúa el comportamiento de la búsqueda.

---

## 2. ¿Qué significa HRES2--H₂?

HRES significa *Hybrid Renewable Energy System*. En este caso el sistema combina:

- generación eólica;
- generación fotovoltaica;
- un electrolizador PEM para producir hidrógeno;
- una batería BESS (*Battery Energy Storage System*);
- conexión a la red para vender el excedente.

El nombre WPEB viene de *Wind--Photovoltaic--Electrolyzer--Battery*. El modelo actual es un sistema conectado a la red: cuando sobra energía renovable, se vende; no se compra energía de la red para alimentar el electrolizador en la regla de despacho implementada.

Es importante conocer también el alcance. El HRES actual produce hidrógeno y vende excedentes, pero no incluye explícitamente un tanque de almacenamiento de hidrógeno ni una pila de combustible que vuelva a convertir H₂ en electricidad. Es un modelo de dimensionamiento y despacho simplificado, no una representación completa de toda una planta de hidrógeno.

### 2.1. Qué decide el optimizador

El diseño físico tiene cuatro componentes. Para explicarlo, se puede escribir:

**z = [P_WT, N_el, P_bat, tau_bat]**

Aquí `z` es el diseño ya decodificado: `P_WT` es la potencia eólica instalada en MW, `N_el` el número de módulos PEM, `P_bat` la potencia máxima de la batería en MW y `tau_bat` su duración nominal en horas.

El optimizador, en cambio, recibe un vector numérico codificado **x = [x_0, x_1, x_2, x_3]**. El decodificador convierte `x_0` en potencia eólica; redondea `x_1` al número entero de módulos; redondea `x_2` al paso de 5 MW; y usa `x_3` redondeado como índice de la lista `[1, 2, 4]` horas. Por eso, `x_3=2` significa índice 2 y se convierte en `tau_bat=4 h`; no significa una batería de 2 horas.

| Componente | Significado | Dominio implementado |
|---|---|---|
| P_WT | potencia eólica instalada | continua, [0, 200] MW |
| N_el | número de módulos PEM | entero, 10 a 20 |
| P_bat | potencia máxima de la batería | pasos de 5 MW, 0 a 50 MW |
| tau_bat | duración nominal de la batería | 1, 2 o 4 horas |

La potencia fotovoltaica instalada no es una quinta variable. Se calcula como:

**P_PV = 200 - P_WT**

`P_PV` y `P_WT` son capacidades instaladas en MW; el número 200 MW es la capacidad renovable total fijada en el modelo.

La capacidad renovable total queda fija en 200 MW. Si se eligen 174.47 MW de viento, automáticamente quedan 25.53 MW de PV.

Cada módulo de electrolizador aporta 5 MW:

**P_el = 5 × N_el**

`P_el` es la potencia nominal del electrolizador en MW; `N_el` es el número entero de módulos y cada módulo aporta 5 MW.

Por ejemplo, N_el = 14 significa un electrolizador de 70 MW. La energía nominal de la batería es:

**E_bat = P_bat × tau_bat**

`E_bat` es la capacidad de energía de la batería en MWh; `P_bat` se mide en MW y `tau_bat` en horas. La multiplicación MW × h produce MWh.

Una batería de 50 MW y 4 horas tiene una capacidad energética nominal de 200 MWh.

Además, el electrolizador no puede superar el 50 % de la capacidad renovable total:

**P_el <= 0.50 × (P_WT + P_PV) = 100 MW**

`0.50` es el límite `electrolyzer_ratio_max`; `P_WT + P_PV = 200 MW` es la capacidad renovable fija. Como cada módulo entrega 5 MW, esto también implica `N_el <= 20`.

Este es un problema **mixto** porque P_WT es continuo, pero N_el, P_bat y tau_bat tienen valores enteros o discretos. Una metaheurística no puede proponer cualquier número y asumir que representa un equipo real; por eso existe una etapa de decodificación.

### 2.2. Qué se hereda del WPEB publicado

El trabajo de referencia de Li et al. mantiene una capacidad eólica más fotovoltaica de 200 MW, simula un año de 8.760 horas, minimiza LCOE y restringe el excedente vendido a la red mediante AGSR menor o igual a 20%.

El caso publicado W190--P10--E95--B30 representa aproximadamente:

- 190 MW eólicos;
- 10 MW fotovoltaicos;
- 95 MW de electrólisis;
- batería de 30 MW y 1 hora;
- LCOE reportado de 0.2692 CNY/kWh.

La implementación de HRES2--H₂ conserva la capacidad renovable total, la simulación anual y la restricción AGSR, pero cambia el espacio de decisión a módulos enteros de electrólisis y opciones discretas de batería. Además, el método de búsqueda deja de ser solamente grid search más descenso de gradiente y pasa a ser un portafolio cooperativo con DTW/DDTW.

---

## 3. Cómo se calcula la generación hora a hora

Una solución no se evalúa con una generación promedio, sino con una serie de 8.760 horas. El código multiplica cada capacidad instalada por un perfil horario normalizado:

**P_gen(t) = P_WT × f_WT(t) + P_PV × f_PV(t), para t = 1, ..., 8760**

`t` es la hora del año; `P_WT` y `P_PV` son capacidades instaladas en MW; `f_WT(t)` y `f_PV(t)` son factores horarios entre 0 y 1; y `P_gen(t)` es la potencia renovable disponible en MW. El paso temporal es `Δt = 1 h`, por lo que una potencia constante durante una fila horaria representa numéricamente esa misma cantidad de MWh en esa hora.

El modelo usa un perfil meteorológico anual sintético fijo. Eso significa que cada solución se prueba sobre las mismas horas de viento, irradiancia y temperatura. Es una condición útil para comparar algoritmos, pero también una limitación: todavía no se estudian varios años meteorológicos ni incertidumbre climática.

### 3.1. Generación eólica

La velocidad del viento `v(t)` determina el factor de potencia eólica `f_WT(t)`, que el código calcula así:

**f_WT(t) = 0**, si `v(t) < v_i` o `v(t) >= v_o`  
**f_WT(t) = [v(t)^3 - v_i^3] / [v_r^3 - v_i^3]**, si `v_i <= v(t) < v_r`  
**f_WT(t) = 1**, si `v_r <= v(t) < v_o`

`v_i = 2.5 m/s` es la velocidad de arranque; `v_r = 10.5 m/s`, la velocidad a la que se alcanza potencia nominal; y `v_o = 25 m/s`, la velocidad de corte por seguridad. `v(t)` es la velocidad medida en la hora `t`. El factor `f_WT(t)` no tiene unidades y queda entre 0 y 1. La potencia eólica de esa hora es `P_WT × f_WT(t)`. El parámetro de turbina nominal del caso es 5 MW, pero el simulador aplica el perfil normalizado directamente a la capacidad eólica agregada `P_WT`.

### 3.2. Generación fotovoltaica

La implementación usa `g(t)`, el valor horario del perfil `ghi_kwh_m2` (numéricamente equivalente a kW/m² para intervalos de una hora). Primero estima temperatura de celda:

**T_c(t) = T_a(t) + [(NOCT - 20) / 0.8] × g(t)**

`T_c(t)` es la temperatura de celda en °C; `T_a(t)` es la temperatura ambiente en °C; `NOCT = 47 °C` es la temperatura nominal de operación; `20 °C` es la temperatura ambiente de referencia; `0.8 kW/m²` es la irradiancia de referencia usada por NOCT; y `g(t)` es la irradiancia horaria. La fracción `g(t)/0.8` representa cuántas veces la irradiancia está respecto de 0.8 kW/m².

Luego calcula el perfil solar normalizado y la potencia:

**f_PV(t) = clip(g(t) × f_der × max(1 + gamma_T × [T_c(t) - 25], 0), 0, 1)**  
**P_PV,gen(t) = P_PV × f_PV(t)**

`f_PV(t)` es el factor horario de PV entre 0 y 1; `clip(q,0,1)` limita `q` a ese intervalo; `max(...,0)` impide que la corrección térmica sea negativa; `f_der = 0.90` modela pérdidas; `gamma_T = -0.005 / °C` equivale a `-0.5 %/°C`; `T_c(t)-25 °C` es el cambio respecto de la temperatura estándar de celda; `P_PV` es la capacidad instalada en MW; y `P_PV,gen(t)` es la potencia fotovoltaica generada en MW. La constante 25 °C corresponde a la referencia de temperatura de celda.

Así, más irradiancia aumenta la producción, mientras que una celda más caliente la reduce. Esta forma reproduce el cálculo de `precalculate_profiles()` en el código, incluida la normalización y el recorte del factor a `[0,1]`.

**Nota de consistencia:** el código `HRES2-H2/wpeb_model.py`, el `paper_final.tex` y la diapositiva se dejaron alineados con gamma_T = -0.5 % por grado Celsius.

---

## 4. Cómo funciona el despacho HRES en una hora

Para cada hora el simulador sigue una regla de prioridad. Supongamos que el electrolizador tiene 70 MW. Su carga mínima es:

**P_el,min = 0.30 × P_el = 0.30 × 70 = 21 MW**

`P_el,min` es la potencia mínima de operación en MW; `P_el` es la potencia nominal del electrolizador en MW; y `0.30` es `electrolyzer_min_load_ratio`, es decir, el PEM debe recibir al menos el 30 % de su potencia nominal para operar.

### Paso 1: calcular la generación renovable

Se suman viento y PV. La generación puede cambiar mucho entre una hora soleada y una hora con poco viento.

### Paso 2: alimentar el electrolizador

Si `P_gen(t) >= P_el,min`, el electrolizador opera. Consume la generación disponible hasta su máximo:

**P_el,usada(t) = min(P_gen(t), P_el)**

`P_el,usada(t)` es la potencia que recibe el electrolizador en MW; `P_gen(t)` es la generación renovable disponible en MW; y `P_el` es el límite nominal en MW. `min(a,b)` devuelve el menor de los dos valores. Esta expresión describe la rama en que la generación ya alcanza la carga mínima; si no la alcanza, el código intenta completarla con batería.

Ejemplo: si hay 50 MW disponibles y el electrolizador es de 70 MW, consume 50 MW; no puede producir a 70 MW porque no hay suficiente generación.

Si hay 100 MW disponibles, consume 70 MW y quedan 30 MW de excedente.

### Paso 3: cubrir una caída con la batería

Si `P_gen(t) < P_el,min`, el modelo intenta descargar la batería. Define la potencia de descarga posible como:

**P_dis,max(t) = min(P_bat, SOC(t) × eta_dis / Δt)**

`P_dis,max(t)` es la máxima potencia descargable en MW; `P_bat` limita la potencia del convertidor en MW; `SOC(t)` es la energía almacenada al comienzo de la hora en MWh; `eta_dis` es la eficiencia de descarga; y `Δt` es la duración del intervalo en horas. Como `Δt=1 h`, el código calcula numéricamente `min(P_bat, SOC × eta_dis)`. La condición `P_gen(t)+P_dis,max(t) >= P_el,min` solo decide si la batería alcanza para encender el electrolizador. Si alcanza, la descarga se calcula como `P_dis(t)=min(P_dis,max(t), P_el-P_gen(t))`: se utiliza toda la potencia disponible hasta llegar a la potencia nominal `P_el`, no solo hasta el mínimo. Si no alcanza `P_el,min`, el electrolizador queda detenido esa hora.

Ejemplo: si hay 15 MW renovables, el electrolizador de 70 MW tiene un mínimo de 21 MW y necesita al menos 6 MW de batería para poder operar. Pero esos 6 MW son solo el umbral de encendido: si la batería puede entregar 8 MW, el código los entrega y el electrolizador recibe 23 MW; si puede entregar 55 MW o más, llega hasta su máximo de 70 MW. Si solo puede entregar 4 MW, no alcanza el mínimo y el electrolizador se detiene.

La batería no se usa para superar la capacidad nominal del electrolizador. Cuando logra cruzar el umbral mínimo, el código aprovecha la descarga posible para elevar la carga del PEM por encima de ese mínimo, hasta donde permita la energía de la batería, su límite de potencia y la potencia nominal del electrolizador.

### Paso 4: almacenar excedentes

Después de alimentar el electrolizador, el excedente se usa para cargar la batería hasta:

**SOC_max = P_bat × tau_bat**

En estas ecuaciones, `SOC(t)` significa energía almacenada en la batería, en MWh (no una fracción entre 0 y 1); `SOC_max` es su máximo en MWh; `P_bat` limita potencia de carga o descarga, en MW; y `tau_bat` fija la duración nominal, en horas. El estado inicia vacío (`SOC(1)=0`) y el código limita la carga por la potencia del convertidor y por el espacio disponible en `SOC_max`.

La regla de carga puede escribirse así:

**S(t) = max(P_gen(t) + P_dis(t) - P_el,usada(t), 0)**  
**P_ch(t) = min(S(t), P_bat, [SOC_max - SOC(t)] / [eta_ch × Δt])**  
**P_grid_sales(t) = S(t) - P_ch(t)**, si el excedente queda positivo.

S(t) es el excedente en MW antes de cargar; P_dis(t) es la potencia que aporta la batería en MW (cero cuando no se descarga); P_ch(t) es la potencia absorbida por la batería en MW; eta_ch es la eficiencia de carga; y Δt=1 h es el paso temporal. min selecciona el menor límite: el excedente disponible, la potencia máxima de batería o la potencia que cabe sin sobrepasar SOC_max. Lo que no se puede almacenar se exporta como P_grid_sales(t) en MW.

La batería no puede cargar más potencia que P_bat en una hora. Si todavía queda energía después de cargarla, esa energía se registra como venta a la red.

### Paso 5: calcular indicadores anuales

El ciclo se repite para las 8.760 horas. Al terminar se suman energía del electrolizador, energía vendida, generación renovable y ciclos de batería.

La eficiencia de carga y descarga se obtiene desde una eficiencia round-trip de 0.90:

**eta_ch = eta_dis = sqrt(0.90) ≈ 0.9487**

`eta_ch` y `eta_dis` son eficiencias sin unidades de carga y descarga. `eta_rt=0.90` es la eficiencia round-trip, igual a `eta_ch × eta_dis`; al asignar la misma eficiencia a ambos sentidos, cada una vale su raíz cuadrada. Una forma de resumir la actualización del estado es `SOC(t+1)=SOC(t)+eta_ch×P_ch(t)×Δt-[P_dis(t)×Δt]/eta_dis`, donde `P_ch(t)` y `P_dis(t)` son potencias de carga y descarga en MW y `Δt=1 h`. El código aplica los límites de potencia y capacidad en cada paso horario.

Para convertir potencia en energía anual, el simulador suma los valores horarios (MW) sobre las 8.760 horas. En forma explícita, `E_el,annual = sum_t P_el,usada(t) × Δt` y `E_grid,sales,annual = sum_t P_grid_sales(t) × Δt`; ambos resultados quedan en MWh/año porque `Δt=1 h`. El mismo criterio se usa para la generación renovable total.

---

## 5. Cómo se calcula el hidrógeno

El electrolizador recibe una cantidad anual de energía, expresada en MWh. Primero se convierte a kWh multiplicando por 1.000. Luego se aplica la eficiencia del electrolizador y se divide por el poder calorífico superior del hidrógeno:

**m_H2 = (E_el,annual × 1000 × eta_el) / HHV_H2**

`m_H2` es la masa producida durante el año en kg/año; `E_el,annual` es la energía anual recibida por el electrolizador en MWh/año; `1000` convierte MWh a kWh; `eta_el=0.75` es la eficiencia de conversión (adimensional); y `HHV_H2=39.4 kWh/kg` es el poder calorífico superior usado para convertir energía química en masa de hidrógeno.

En la implementación:

- eta_el = 0.75;
- HHV_H2 = 39.4 kWh/kg.

Por tanto, más energía realmente entregada al electrolizador implica más kilogramos de H₂, pero aumentar la capacidad del electrolizador también aumenta CAPEX, reemplazos y operación. Esa tensión económica es la razón por la que no conviene simplemente hacer el electrolizador lo más grande posible.

En la versión actual LCOH se reporta como indicador secundario:

**LCOH = costo anualizado / m_H2**

El numerador es el costo anual equivalente en CNY/año y `m_H2` es la producción anual en kg/año; por eso el resultado se expresa en CNY/kg. `LCOH` significa *Levelized Cost of Hydrogen*. En este trabajo se informa como métrica, no se minimiza como objetivo.

No se está optimizando LCOH simultáneamente con LCOE todavía.

---

## 6. Objetivo económico y restricciones

### 6.1. NPC y costo anualizado

El modelo calcula el NPC (*Net Present Cost*) sumando, para viento, PV, electrolizador y batería:

- inversión inicial o CAPEX;
- reemplazos durante la vida útil;
- operación y mantenimiento anual;
- descuento financiero de los flujos.

La estructura que implementa el código se puede escribir de forma compacta como:

**NPC = sum_j [1000 P_j C_cap,j + sum_(y en R_j) (1000 P_j C_rep,j)/(1+r)^y + sum_(y=1..N) (1000 P_j C_OM,j)/(1+r)^y]**

`j` identifica cada tecnología (`WT`, `PV`, `EL` o `BAT`); `P_j` es su capacidad en MW (para batería se usa potencia `P_bat`, no energía `E_bat`); `1000 P_j` convierte MW a kW; `C_cap,j` es CAPEX en CNY/kW; `C_rep,j` es el costo de reemplazo en CNY/kW; `C_OM,j` es operación y mantenimiento en CNY/(kW·año); `R_j` es el conjunto de años de reemplazo definidos por la vida útil de ese componente, excluyendo el año final del proyecto; `y` es el año del flujo; `r` es la tasa de descuento; y `N` es el horizonte del proyecto. El primer término es la inversión al inicio (no descontada); los dos sumatorios son costos futuros traídos a valor presente.

Los valores unitarios predeterminados del código son:

| Tecnología j | CAPEX (CNY/kW) | Reemplazo (CNY/kW) | O&M (CNY/(kW·año)) | Vida útil L_j (años) |
|---|---:|---:|---:|---:|
| Eólica (WT) | 5917.00 | 0.00 | 40.20 | 25 |
| Fotovoltaica (PV) | 4633.00 | 0.00 | 17.60 | 25 |
| Electrolizador (EL) | 6964.00 | 5969.14 | 208.92 | 15 |
| Batería (BAT) | 2549.00 | 500.00 | 10.00 | 10 |

L_j es la vida útil que determina los años de reemplazo. Con horizonte N=25, el electrolizador se reemplaza en el año 15 y la batería en los años 10 y 20; eólica y PV no se reemplazan dentro del horizonte según esta configuración. El costo de batería se calcula por kW de potencia instalada, no por kWh de capacidad energética.

Después se anualiza el NPC con el factor de recuperación de capital:

**costo anualizado = NPC × CRF(r, N)**

`NPC` es el costo presente neto en CNY; `CRF` es el factor que lo convierte a una cuota anual uniforme; `r` es la tasa real anual de descuento, 0.0435 (4.35 %); y `N` es la vida del proyecto, 25 años. El código calcula:

**CRF(r,N) = [r(1+r)^N] / [(1+r)^N - 1]**

`r` debe ingresarse como fracción decimal, no como el número 4.35. Cada costo futuro en el NPC se descuenta dividiéndolo por `(1+r)^y`, donde `y` es el año en que ocurre el reemplazo o gasto de operación.

### 6.2. LCOE

El objetivo principal es:

**LCOE = (NPC × CRF) / E_entregada,annual**

`LCOE` (*Levelized Cost of Electricity*) es el costo nivelado en CNY/kWh; `NPC × CRF` es el costo anualizado en CNY/año; y `E_entregada,annual` es la energía anual contabilizada en kWh/año.

En el código, la energía entregada anual corresponde a la energía que llega al electrolizador más la energía vendida a la red:

**E_entregada,annual = 1000 × (E_el,annual + E_grid,sales,annual)**

`E_el,annual` es la energía anual absorbida por el electrolizador y `E_grid,sales,annual` es la energía anual exportada a la red; ambas están en MWh/año. El factor `1000` las convierte a kWh/año antes de dividir el costo anual por la energía anual entregada. En el sumatorio que las produce, `P(t)` es potencia en MW y `Δt=1 h` es la duración de cada intervalo.

El factor de utilización del electrolizador que también reporta el código es `CF_el = E_el,annual / (P_el × 8760 h)`. `CF_el` es adimensional; `P_el` es la potencia nominal en MW; y `8760 h` es la duración del año simulado. Indica qué fracción de la energía que consumiría operando a plena potencia se utilizó realmente.

Por eso el modelo busca un equilibrio: demasiada infraestructura aumenta el numerador; muy poca infraestructura puede reducir la energía entregada o producir diseños inviables.

### 6.3. AGSR

AGSR es el *Annual Grid Surplus Ratio*:

**AGSR = (energía total vendida a la red) / (generación renovable total)**

En unidades de energía, la expresión es **AGSR = E_grid,sales,annual / E_ren,annual**, donde `E_grid,sales,annual = sum_t P_grid_sales(t) × Δt` y `E_ren,annual = sum_t P_gen(t) × Δt`.

`P_grid_sales(t)` es la potencia excedente exportada en la hora `t`; `P_gen(t)` es la generación renovable total en esa hora; `Δt=1 h`; y `E_grid,sales,annual` y `E_ren,annual` resultan en MWh/año. `AGSR` es una razón sin unidades; 0.20 equivale a 20 %. Como todas las filas duran exactamente una hora, en el código las sumas numéricas de MW son iguales a las energías anuales expresadas en MWh.

La restricción es:

**AGSR <= 0.20** (como máximo 20 %)

Interpretación: como máximo 20% de la generación renovable anual puede terminar como excedente vendido a la red. No es un objetivo que queramos minimizar sin límite; es una condición de diseño.

Si AGSR supera 20%, el diseño se marca como no factible y la función devuelve una penalización:

**f_penalized = 100 + 10 × AGSR**

`f_penalized` es el valor artificial que recibe el optimizador cuando una solución no cumple la restricción; `AGSR` es la razón calculada. Las constantes 100 y 10 son parámetros de penalización para que una solución inviable resulte mucho peor que un LCOE factible cercano a 0.27. El código activa esta penalización si `AGSR > 0.20`.

Como un LCOE factible es cercano a 0.27, una penalización de alrededor de 100 es enormemente peor. Así, la metaheurística aprende que primero debe buscar la región factible y luego mejorar el costo.

### 6.4. La solución encontrada

La configuración común de DTW y DDTW fue aproximadamente:

**P_WT = 174.47 MW; P_PV = 25.53 MW**

**N_el = 14; P_el = 70 MW; P_bat = 50 MW; tau_bat = 4 h**

Su batería equivale a 200 MWh nominales. En las 31 corridas se obtuvo LCOE medio de 0.267160 CNY/kWh, mejor LCOE de 0.267159, LCOH alrededor de 17.631 CNY/kg, AGSR de 20% y 100% de factibilidad.

---

## 7. ¿Qué es DTW?

DTW significa *Dynamic Time Warping*. Es un algoritmo para medir la similitud entre dos series permitiendo que una avance más rápido o más lento que la otra.

### 7.1. Por qué sirve en una metaheurística

Durante una optimización se registra el mejor valor encontrado hasta cada iteración:

**f_1, f_2, f_3, ..., f_t**

Esa secuencia es una trayectoria. Una búsqueda activa suele mostrar mejoras; una búsqueda estancada muestra una meseta.

Un contador simple podría decir “llevamos 15 iteraciones sin mejora”. El problema es que ese contador no conoce la forma anterior. Una pausa de 15 iteraciones puede ser normal después de una mejora grande, o puede indicar que el algoritmo quedó atrapado. DTW agrega contexto temporal.

### 7.2. Dos trayectorias y una plantilla

El monitor toma una ventana reciente **X = [x_1, ..., x_W]** y la compara con una trayectoria de referencia **Y = [y_1, ..., y_M]**.

En la implementación se usan dos referencias:

- una rampa que representa progreso sostenido;
- una constante que representa estancamiento.

El monitor actual no aplica una normalización min--max explícita: ancla las referencias al primer valor de la ventana y calcula una pendiente adaptativa; DDTW reduce además la influencia del nivel absoluto.

### 7.3. Qué significa “deformar el tiempo”

Imagina dos personas subiendo una escalera. Una sube un peldaño por segundo y otra se detiene unos segundos entre peldaños. Ambas siguen la misma tendencia, pero sus cambios no ocurren en los mismos instantes. Comparar punto a punto las declararía diferentes. DTW puede alinear varios puntos de una trayectoria con un punto de la otra y reconocer que la forma general es parecida.

En optimización, esto permite que una metaheurística que mejora lentamente todavía se parezca a la plantilla de progreso. Una meseta persistente, en cambio, no se parece a una rampa descendente en un problema de minimización.

### 7.4. La matriz de programación dinámica

Para cada par de puntos se calcula un costo local:

**d(x_i, y_j) = |x_i - y_j|**

Luego se construye una matriz acumulada. Cada celda se calcula así:

**D(i, j) = d(x_i, y_j) + min{ D(i-1, j), D(i, j-1), D(i-1, j-1) }**

El significado de los tres términos es:

- D(i-1, j): avanzar en la primera serie;
- D(i, j-1): avanzar en la segunda serie;
- D(i-1, j-1): avanzar en ambas.

El valor final D(n, m) es el costo del mejor camino desde el primer par de puntos hasta el último. La banda de Sakoe--Chiba limita cuánto puede deformarse el camino. En el código, si la banda se deja en cero, se calcula automáticamente como aproximadamente 10% de la ventana; para W = 40 eso equivale a una banda cercana a 4.

### 7.5. Ejemplo numérico del delta

Para que se vean los tres movimientos de DTW, usamos seis puntos y una banda `b = 2`:

- ventana observada: **X = [0.50, 0.50, 0.50, 0.56, 0.60, 0.65]**;
- progreso: **r = [0.50, 0.55, 0.60, 0.64, 0.65, 0.65]**;
- meseta: **c = [0.50, 0.50, 0.50, 0.50, 0.50, 0.50]**.

Para `DTW(X,r)`, el camino óptimo calculado con la misma recurrencia del código es:

`(1,1) → (2,1) → (3,1) → (4,2) → (5,3) → (6,4) → (6,5) → (6,6)`.

Este camino contiene:

- dos movimientos verticales;
- tres movimientos diagonales;
- dos movimientos horizontales.

Los costos locales recorridos son:

`0 + 0 + 0 + 0.01 + 0 + 0.01 + 0 + 0 = 0.02`.

Por tanto, **D_prog = DTW(X,r) = 0.02**.

Al comparar la misma ventana con la meseta, el código entrega:

**D_mes = DTW(X,c) = 0.31**.

Finalmente:

**delta = D_prog - D_mes = 0.02 - 0.31 = -0.29**.

El signo negativo indica que la ventana se parece más al progreso. La ventaja de DTW es precisamente que reconoce esa semejanza aunque las dos series avancen a ritmos diferentes. En el benchmark HRES2-H2 se usan 40 valores y una banda automática cercana a 4.

### 7.6. Un detalle crucial: la minimización se transforma

El monitor genérico `StagnationMonitor` está formulado para interpretar que un valor mayor significa progreso. En HRES, en cambio, se minimiza LCOE: un LCOE menor es mejor.

Por eso los solucionadores llaman al monitor con:

```python
monitor.update(-mejor_lcoe)
```

Si el LCOE baja de 0.28 a 0.27, el valor observado por el monitor sube de -0.28 a -0.27. Para el monitor, eso representa una mejora. Al final, el orquestador sigue comparando los LCOE originales con `<`, porque la solución global continúa siendo la de menor costo.

Esta inversión de signo es solo una adaptación interna; no significa que el objetivo del HRES pase a maximizarse.

---

## 8. ¿Qué es DDTW?

DDTW es *Derivative Dynamic Time Warping*. En vez de comparar directamente los niveles de la serie, compara una aproximación de sus cambios:

**x'_t ≈ x_t - x_(t-1)**

Si el fitness mejora continuamente, la derivada tiene una tendencia asociada a la mejora. Si la trayectoria queda plana, la derivada se acerca a cero.

La ventaja conceptual es que dos series con niveles distintos pueden tener la misma forma de cambio. DDTW reduce la dependencia del valor absoluto y enfatiza la pendiente.

La desventaja es que derivar puede amplificar ruido. Por eso DDTW no es automáticamente mejor que DTW. En los resultados HRES2--H₂ ambas variantes llegaron prácticamente al mismo LCOE y la comparación directa no mostró una diferencia significativa después de Holm.

---

## 9. Cómo funciona exactamente el monitor implementado

En el benchmark HRES2 se usan, como referencia, estos parámetros:

| Parámetro | Valor | Significado |
|---|---:|---|
| `window` | 40 | puntos recientes usados por DTW/DDTW |
| `band` | 0 | se convierte automáticamente en una banda cercana a 10% de la ventana |
| `plateau_max` | 15 | iteraciones sin mejora antes de considerar meseta |
| `patience` | 3 | épocas consecutivas que deben confirmar la condición |
| `use_ddtw` | True en la variante DDTW | usar derivadas en lugar de niveles |
| `adapt_thresholds` | True | calcular umbrales desde el historial |
| `p_low` | 30 | percentil bajo para la distancia a la constante |
| `p_high` | 70 | percentil alto para rampa y delta |

Cada actualización hace lo siguiente:

1. recibe el mejor valor de la iteración;
2. aplica el signo negativo para que la mejora de una minimización sea un aumento interno;
3. actualiza el contador de iteraciones sin mejora;
4. espera hasta tener al menos 40 valores;
5. extrae los últimos 40 y construye una rampa y una constante;
6. calcula D1, distancia a la rampa, y D2, distancia a la constante;
7. calcula Delta = D1 - D2;
8. estima umbrales adaptativos a partir de los historiales;
9. verifica tres condiciones;
10. incrementa o reinicia la racha de disparo.

Las tres condiciones son:

1. no_improve_len >= 15;
2. D2 <= theta_c;
3. D1 >= theta_r **o** Delta >= theta_Delta.

Interpretación:

- el contador confirma que no hubo mejora suficiente;
- la ventana se parece a una constante;
- y además se aleja de la rampa de progreso.

Si las tres condiciones se cumplen, aumenta `trigger_streak`. Si dejan de cumplirse, la racha vuelve a cero. El cambio solo se ejecuta cuando la racha llega a `patience=3`. Esto evita cambiar de algoritmo por una única ventana anómala.

Cuando todavía no hay diez valores de distancia para construir percentiles, el código usa umbrales iniciales proporcionales al tamaño de la ventana. Después utiliza el percentil 30 de D2 y el percentil 70 de D1 y Delta. Así el criterio se adapta a la escala de la trayectoria observada.

---

## 10. Cómo DTW cambia de metaheurística

El pool HRES2 tiene dos grupos:

### Poblacionales

PSO, GWO, WOA, EHO, ACO y ABC mantienen poblaciones de candidatos. Su fortaleza principal es explorar regiones diferentes del espacio.

### De trayectoria o búsqueda local

ILS, SA, TS y VNS trabajan con vecinos, perturbaciones, memoria o vecindades. Su fortaleza principal es intensificar una región prometedora.

Una época es un bloque de iteraciones de un solucionador. Si el solucionador termina su época sin alerta, se ejecuta otra. Si el monitor confirma estancamiento, el epoch se aborta y el orquestador selecciona otro solucionador.

La transferencia funciona así:

1. se toma la mejor solución factible del solucionador saliente;
2. se compara con el incumbente global;
3. se conserva la mejor de las dos;
4. el receptor recibe ese vector como warm start;
5. si necesita una población, se crea una nueva población aleatoria y se inyecta el incumbente; los modos opcionales `mutated` y `mixed` agregan perturbaciones alrededor de él;
6. empieza una nueva época.

El incumbente no se pierde al cambiar de metaheurística. La memoria compartida es únicamente esa mejor solución global; no se conserva el estado interno del algoritmo saliente.

### ¿Qué significa exactamente *warm start* aquí?

*Warm start* significa **inicio con información útil**. El nuevo solucionador no comienza con una solución completamente desconocida: recibe el vector de diseño que hasta ese momento tenía el mejor LCOE factible. En HRES2--H₂ ese vector tiene cuatro coordenadas:

**x = [viento, módulos de electrolizador, potencia BESS, índice de duración]**

Por ejemplo, si el incumbente es aproximadamente **[174.47, 14, 50, 2]**, el receptor hereda 174.47 MW eólicos, 14 módulos, 50 MW de batería y el índice 2, que el decodificador convierte en una duración de 4 horas. El receptor puede modificar ese punto y buscar una mejora, pero ya conoce una región prometedora del espacio.

La forma concreta depende del tipo de metaheurística:

- **SA, ILS, TS y VNS:** el incumbente se usa como solución actual desde la primera iteración y se generan vecinos a su alrededor.
- **PSO, GWO, WOA, EHO, ACO y ABC:** se crea una población nueva, se recorta el vector a los límites y se inyecta el incumbente. Con el modo predeterminado (`random`) reemplaza al individuo de peor calidad; con `mutated` o `mixed` también se crean candidatos perturbados alrededor de él.

Por eso *warm start* no es lo mismo que copiar toda la población anterior, conservar todas las velocidades o continuar exactamente el estado interno del algoritmo. Se conserva la mejor solución global y se entrega una inicialización informada al receptor. Lo contrario es un **cold start**: comenzar completamente desde puntos aleatorios y descartar lo aprendido por el solucionador anterior.

---

## 11. Ejemplo completo con la solución encontrada

Considera el vector de diseño **x = [174.47, 14, 50, 2]**.

### Decodificación

- viento: 174.47 MW;
- PV: 200 - 174.47 = 25.53 MW;
- electrolizador: 14 × 5 = 70 MW;
- batería: 50 MW;
- energía de batería: 50 × 4 = 200 MWh.

### Despacho de tres horas hipotéticas

**Hora A: P_gen = 100 MW**

- el electrolizador usa 70 MW y quedan 30 MW de excedente;
- suponiendo que la batería está vacía, puede cargar los 30 MW porque su límite es 50 MW;
- con eta_ch = 0.9487 y Δt = 1 h, el SOC sube en 30 × 0.9487 × 1 = 28.46 MWh;
- no queda excedente para vender a la red en esta hora.

**Hora B: P_gen = 15 MW**

- para P_el = 70 MW, el mínimo es P_el,min = 0.30 × 70 = 21 MW;
- desde el SOC de la hora anterior, P_dis,max = min(50, 28.46 × 0.9487 / 1) ≈ 27 MW;
- como 15 + 27 supera 21 MW, la batería puede sostener operación y el código descarga hasta 27 MW;
- el electrolizador recibe 15 + 27 = 42 MW, no solo los 6 MW mínimos que bastaban para cruzar el umbral.

**Hora C: P_gen = 5 MW y batería vacía**

- no queda energía almacenada, por lo que P_dis,max = 0 MW y no se alcanza el mínimo de 21 MW;
- el electrolizador se detiene esa hora;
- no se produce hidrógeno durante esa hora.

Al repetir esto durante 8.760 horas se obtiene energía de electrólisis, hidrógeno anual, energía vendida y AGSR. Luego se calcula el costo. Ese único número de LCOE es el valor que recibe la metaheurística.

En paralelo, la metaheurística registra la secuencia de mejores LCOE. DTW/DDTW no vuelve a simular el HRES: solo analiza esa secuencia y decide si la siguiente época debe usar otro algoritmo.

---

## 12. Cómo leer los resultados de la presentación

### Tabla de métricas

- **LCOE medio:** rendimiento promedio de las 31 corridas.
- **Mejor LCOE:** mínimo observado entre las corridas.
- **sigma:** dispersión; valores del orden de 10^(-6) indican alta estabilidad.
- **LCOH:** costo por kilogramo de hidrógeno, reportado como indicador secundario.
- **AGSR:** excedente vendido a red; debe ser menor o igual a 20%.
- **Factibilidad:** porcentaje de corridas que respetan las restricciones.

DTW y DDTW obtienen la misma media y el mismo mejor LCOE, con factibilidad completa. La conclusión no es que DDTW sea siempre superior: en este caso ambos métodos son equivalentes en el resultado final.

### Tabla de solucionadores base

Los métodos independientes pueden obtener ocasionalmente un buen mínimo, pero muestran mayor dispersión y porcentajes menores de factibilidad. El framework busca combinar:

1. menor costo;
2. baja variabilidad;
3. cumplimiento de AGSR;
4. transferencia adaptativa durante la ejecución.

### Gráficos de convergencia

La curva muestra el mejor LCOE acumulado. Como el problema minimiza, la curva baja cuando aparece una mejora. Las líneas verticales indican cambios de solucionador. Una línea horizontal larga es un plateau: el incumbente no está mejorando.

La interpretación correcta es “el monitor detectó una fase poco productiva y permitió probar otro mecanismo”, no “cada línea vertical significa que se encontró un nuevo óptimo”. Algunas rotaciones pueden no mejorar inmediatamente; su valor está en evitar que toda la búsqueda quede atrapada en el mismo comportamiento.

---

## 13. Respuestas esenciales para entender y defender el trabajo

### ¿Qué optimiza exactamente el algoritmo?

Optimiza el vector mixto de capacidades para minimizar LCOE, sujeto a la capacidad renovable total, los dominios de módulos y batería, el límite de electrólisis y AGSR menor o igual a 20%. LCOH, hidrógeno y emisiones se reportan como métricas adicionales.

### ¿DTW optimiza el HRES?

No directamente. DTW no cambia capacidades ni calcula costos. Observa la trayectoria del objetivo generada por los solucionadores y decide cuándo cambiar de solucionador.

### ¿Por qué no usar solamente un contador de iteraciones?

Porque el contador no distingue una pausa corta de un estancamiento persistente ni se adapta a la escala de cada objetivo. DTW compara la forma de una ventana con patrones de progreso y meseta.

### ¿Cuál es la diferencia entre DTW y DDTW?

DTW compara niveles de la trayectoria; DDTW compara sus cambios o pendientes. DDTW reduce el efecto del nivel absoluto, pero puede ser más sensible al ruido.

### ¿Por qué se usa una plantilla constante?

Porque una meseta es el patrón que queremos reconocer. La distancia a la rampa indica cuánto se aleja la ventana del progreso; la distancia a la constante indica cuánto se parece a una fase plana.

### ¿Por qué AGSR aparece cerca de 20% en la solución óptima?

Porque el límite está activo. Vender más excedente puede aumentar energía entregada y favorecer el LCOE, pero no está permitido superar 20%. El optimizador termina buscando el mejor costo en la frontera de factibilidad.

### ¿Qué significa que DTW y DDTW no sean significativamente diferentes?

Significa que con este perfil meteorológico, este presupuesto y esta configuración no hay evidencia estadística suficiente para afirmar que una variante sea mejor que la otra. No significa que sean idénticas en todas las ejecuciones o que no puedan diferenciarse en otros escenarios.

### ¿Cuál es la principal limitación actual?

El caso usa un único perfil meteorológico anual sintético y un modelo HRES simplificado sin tanque de hidrógeno ni fuel cell. La siguiente etapa debe evaluar escenarios meteorológicos múltiples, degradación de equipos, tarifas variables y una formulación multiobjetivo real.

## 14. Resumen que debes recordar

1. Una solución HRES es un diseño de capacidades, no una sola potencia horaria.
2. El simulador convierte ese diseño en un despacho de 8.760 horas.
3. El despacho produce LCOE, LCOH, hidrógeno, excedentes y factibilidad.
4. La metaheurística intenta minimizar LCOE respetando AGSR.
5. DTW compara la forma de la trayectoria reciente con progreso y meseta.
6. DDTW hace esa comparación sobre las pendientes.
7. Una alerta persistente cambia el solucionador, pero conserva el incumbente.
8. En los resultados actuales, DTW y DDTW alcanzan la misma configuración HRES2--H₂ y resultados prácticamente iguales.
