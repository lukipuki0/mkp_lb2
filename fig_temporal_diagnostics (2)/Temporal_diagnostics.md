# Diagnóstico temporal de una sola corrida

Copia del procedimiento usado en DTW_optimization para responder un pedido
de figura sin repetir la campaña completa. El script que lo hace es
`temporal_diagnostics.py`, en esta misma carpeta.

## Qué se pidió

El manuscrito ya decía que la figura de conteos (cuántas veces el
controlador entra a exploración) no reemplaza un diagnóstico temporal.
Había que generar esa figura antes de contestar, con una sola ejecución:

- una instancia, un algoritmo y una semilla fija;
- guardar \(f_t^*\), \(R_t\), \(C_t\), \(D_R(t)\), \(D_C(t)\), \(\Delta_t\),
  \(\theta_R(t)\), \(\theta_C(t)\), \(\theta_\Delta(t)\) y \(s_t\);
- una figura de 3 o 4 paneles.

No se rehicieron las miles de corridas. No se tocó la configuración global
del proyecto (`mkp_common/config.py` acá tenía otros valores). Los
parámetros del manuscrito se fijaron dentro del script.

## Qué se eligió en este proyecto

| Decisión | Valor | Por qué |
|---|---|---|
| Estrategia | Binary-Complex | Es el controlador que usa todas las señales |
| Algoritmo | BDE | Es el caso en el que el manuscrito discute más cambios de modo |
| Instancia | `mknapcb1`, índice 0 | Primera instancia del protocolo, \(n=100\), corre en cerca de un minuto |
| Semilla | \(q=1\) | Primera réplica independiente. En la campaña esa semilla ya tenía 3 entradas a exploración y fitness 24273 |
| Parámetros | Los de la tabla del paper: \(W=200\), banda 2, \(s_{\min}=2\), percentiles 40/60, \(\pi_{\max}=5\), \(\tau_{\mathrm{pat}}=3\), DDTW activo | Tienen que coincidir con lo que lee el revisor, no con el config actual del código |

Comprobar el fitness contra el JSON de la campaña. Si no coincide, la
corrida no es la del manuscrito.

## Pasos

1. Elegir la corrida en los resultados que ya existen. Mirar el conteo de
   cambios de modo de la semilla 1. Si esa semilla no se mueve, no
   cazar otra semilla a ojo: documentar la regla (por ejemplo, la primera
   semilla del protocolo).
2. Escribir un script aparte. No cambiar el config compartido.
3. Repetir el mismo lazo del runner del paper: misma semilla, mismo modo
   inicial, misma decisión. En cada iteración guardar las series. Las
   referencias \(R_t\) y \(C_t\) son ventanas de largo \(W\); antes de que
   la ventana se llene van en NaN.
4. Verificar que al recomputar la distancia con las mismas funciones del
   monitor se obtiene el mismo \(D_R\) y \(D_C\). Si no, la figura no
   corresponde al controlador.
5. Guardar un `.npz` con las ventanas y un `.csv` con las series escalares.
6. Dibujar cuatro paneles a ancho completo, no en una grilla 2×2 apretada.
   Los números de iteración de cuatro cifras no entran en media columna.
7. Poner la explicación en el párrafo del manuscrito. La leyenda de la
   figura queda en una frase.
8. Encerrar el bloque nuevo entre `\hl{inicio}` y `\hl{fin}` para ver
   qué copiar. `soul` ya carga `\hl` en la clase MDPI.

## Los cuatro paneles

- (a) \(f_t^*\) con el estado \(s_t\) de fondo. Un recuadro marca la
  ventana de subida y otro la ventana plana.
- (b) Esas dos ventanas, cada una restada de su propio inicio, contra la
  rampa \(R_t\) y la constante \(C_t\). Así las dos quedan en la misma
  escala. La ventana de subida se elige en explotación y se corre unos
  pasos hacia adelante si el salto cae en el último punto: si no, el
  escalón final no se dibuja. La ventana plana es el medio del tramo
  más largo en el que la ventana entera no se mueve y el controlador
  está en exploración.
- (c) \(D_R(t)\), \(D_C(t)\) y sus umbrales, en un solo eje.
- (d) \(\Delta_t\) en su propia escala (acá varía poco, entre 370 y 398)
  y, debajo, \(s_t\) con los parámetros que cambia. Los tramos cortos de
  explotación se marcan con triángulos porque en 2000 iteraciones una
  ráfaga de 7 pasos no se ve.

No uses un eje gemelo para \(\Delta_t\) junto a las distancias: estira una
variación chica a toda la altura y se pisa con las etiquetas de \(s_t\).

## Texto que va al LaTeX

El detalle va en el párrafo. La leyenda solo identifica la corrida.
El bloque queda entre las dos marcas:

```latex
\hl{inicio}
Figure~\ref{fig:temporal_diagnostics} supplies one such synchronized trace.
It is a single Binary-Complex run of BDE on the first instance of
\texttt{mknapcb1} ($n=100$, $m=5$), with seed $q=1$ and the
parameterization in Table~\ref{tab:dtw_params}. The run reaches a
best-so-far fitness of $24\,273$ (gap $0.44\%$ from the known optimum
$24\,381$) and enters the exploration-oriented configuration three times.
Panel (a) shows $f_t^*$ with the controller state $s_t$ in the background.
... el resto de los paneles, en prosa ...
Figure~\ref{fig:dtw_activations}.

\begin{figure}[htbp]
  \centering
  \includegraphics[width=\textwidth]{fig_temporal_diagnostics.pdf}
  \caption{Temporal diagnostics for one Binary-Complex run
  (BDE, \texttt{mknapcb1}, seed $q=1$).}
  \label{fig:temporal_diagnostics}
\end{figure}
\hl{fin}
```

En este manuscrito el bloque está en `contexto/latex/template.tex`,
subsección Adaptive Controller Behavior, después de la frase que dice que
los conteos no reemplazan un diagnóstico temporal.

El PDF de la figura tiene que estar en la misma carpeta que el `.tex`,
porque el `\includegraphics` no lleva ruta.

## Qué cambiar en otro proyecto

En `temporal_diagnostics.py`, todo lo que hay que tocar está marcado
con `ADAPTAR`:

1. `ROOT`, para que apunte a la raíz del otro proyecto.
2. Los imports del controlador, el problema y el monitor.
3. La tabla de parámetros, la instancia, la semilla y el presupuesto.
   Usar los valores publicados, no los del config de desarrollo.
4. El cuerpo de `run_representative`: el lazo tiene que ser el mismo que
   el runner del paper (modo inicial, momento en que se decide \(s_t\),
   cuándo la ventana está lista).

No hace falta tocar `save_series` ni la figura si las series salen con
estas claves: `t`, `f_star`, `R`, `C`, `D_R`, `D_C`, `delta`, `theta_R`,
`theta_C`, `theta_delta`, `s`, `adaptive_threshold`, `meta`.

`R` y `C` tienen forma `(T, W)`.

## Cómo correrlo acá

Desde la raíz de DTW_optimization:

```bash
python "lucas copiar/temporal_diagnostics.py"
python "lucas copiar/temporal_diagnostics.py" --plot-only
```

La primera línea repite la corrida (cerca de un minuto en `mknapcb1`).
La segunda solo redibuja a partir del `.npz`.

Salidas, que este script sigue escribiendo en el proyecto original:

- `analisis/temporal_diagnostics/series_binary_complex_bde_mknapcb1_seed1.npz`
- `analisis/temporal_diagnostics/series_binary_complex_bde_mknapcb1_seed1.csv`
- `analisis/temporal_diagnostics/fig_temporal_diagnostics.pdf`
- `contexto/latex/fig_temporal_diagnostics.pdf`
