# Cambios pendientes del manuscrito GWO–DE–DTW

Revisión de solo lectura de `overleaf/gwo_de_dtw_paper/` (2026-11-23), hecha por cinco revisores por sección con las skills `scientific-writing`, `antislop-academic-writing`, `peer-review` y `scientific-critical-thinking`, más una revisión de experimentos contra los CSV de campaña.

**Resultado general:** las cifras del manuscrito coinciden con los datos (Friedman, gaps, conteos, HRES2, evaluaciones). Los cambios son de redacción, incongruencias entre secciones y afirmaciones que exceden la evidencia.

Las rutas son relativas a `overleaf/gwo_de_dtw_paper/`. Los textos de reemplazo están en inglés, el idioma del manuscrito. Los números de línea corresponden a la versión revisada y pueden desplazarse.

Leyenda: 🔴 alta · 🟡 media · 🟢 baja/criterio · ☐ pendiente

---

## 🔴 Alta prioridad

### ☐ 1. Abstract con conclusión vacía
- **Dónde:** `mdpi_gwo_de_dtw.tex:77-79`
- **Actual:** "These results show that trajectory-driven parameter control applies to both problem domains, but…"
- **Problema:** "applies" no es un hallazgo; el método corre en ambos dominios por diseño. Omite el único resultado positivo respaldado y que DTW/DDTW fueron idénticos en HRES2.
- **Reemplazo:**
  > In each domain, Binary-Complex was better than the weaker fixed profile, which differed between domains, but it was not separated from the stronger one; DTW and DDTW produced identical final objectives on HRES2. The results therefore do not establish an advantage over choosing the better fixed profile.

### ☐ 2. $G_3$ descrito como vínculo con el control, pero es fijo
- **Dónde:** `sections/03_background/background.tex:22`; `sections/04_proposed_approach/proposed_approach.tex:27`
- **Actual:** "The offset $G_3$ is the base flip probability contributed by the active parameter profile, which is what links binarization to the control layer."
- **Problema:** $G_3=0.1$ en ambos perfiles; el control no actúa sobre la binarización. El valor solo aparece en la sección de sensibilidad (`experimental_setup.tex:114`), no en las tablas de la campaña principal.
- **Reemplazo (background):**
  > The offset $G_3$ is the base flip probability. It belongs to the parameter profile, but both profiles use the same value (Table~\ref{tab:profiles}), so in the reported experiments the control layer acts on $a$, $F$ and $CR$ rather than on the binarization.
- **Reemplazo (proposed_approach:27):** "it sets the GWO and DE parameters; the LB2 offset $G_3$ is fixed at 0.1 in both profiles."
- **Además:** añadir la fila `$G_3$ (LB2, MKP only) & 0.1 & 0.1` a `tab:profiles`.

### ☐ 3. "Ajustes por dominio" cuando son idénticos
- **Dónde:** `sections/09_conclusions/conclusions.tex:4` y `:12`; `background.tex:295`; `proposed_approach.tex:55`
- **Problema:** `experimental_setup.tex:62` dice "the same window, band, reference slope, percentile levels and confirmation settings are used for MKP and HRES2", y `tab:dtw_params` tiene valores iguales en ambas columnas.
- **Reemplazos:**
  - conclusions:4 → "applied with the same monitor settings (Table~\ref{tab:dtw_params}) to a binary and to a mixed continuous–discrete problem"
  - conclusions:12 → "evaluated in two domains under one monitor configuration"
  - background:295 → "The percentile levels are configuration settings, identical for MKP and HRES2 (Section~\ref{sec:experimental_setup})."
  - proposed_approach:55 → "The percentile levels are those listed in Table~\ref{tab:dtw_params}; the same pair is used for MKP and HRES2."

### ☐ 4. Umbrales "relativos, no absolutos": falso en el calentamiento
- **Dónde:** `proposed_approach.tex:55` (y la justificación de la línea 49)
- **Actual:** "Each threshold is thus relative to the distances this run has already produced, not to an absolute constant."
- **Problema:** antes de 10 observaciones el código usa constantes absolutas, 0.1W/0.5W/0.3W (`dtw_stagnation.py:249-256`). Las políticas binarias reemplazan el monitor tras cada cambio (`gwo_de_dtw/controller.py:82-83`), así que ese periodo se repite.
- **Reemplazo:**
  > Once ten distances have accumulated, each threshold is relative to the distances this run has already produced. Before that point, and again after every monitor replacement by a binary strategy, the fallback values are absolute constants expressed in objective units, so they do not follow the scale of the problem.

### ☐ 5. Falta la limitación del efecto techo en HRES2
- **Dónde:** `conclusions.tex:21`
- **Problema:** 5 de 6 políticas alcanzan el mismo diseño en las 31 semillas (desviación estándar 0), en el límite AGSR = 0.200. Ausencia de diferencias no es igualdad de capacidad.
- **Añadir:**
  > Because five of the six policies reached the same design in every seed, the scenario offers little power to discriminate among policies; a ceiling effect, rather than equal capability, may explain these ties.

### ☐ 6. Tamaños de efecto e intervalos de confianza no reportados
- **Dónde:** `sections/07_statistical_analysis/statistical_analysis.tex:10`
- **Actual:** "Mean and median paired profit differences and the number of zero differences are also recorded for every contrast."
- **Problema:** se dice que se registran, pero no se muestran. Sin ellos no se ve la precisión de "no difference established against Vanilla-Exploitation".
- **Acción:** añadir a `tab:mkp_wilcoxon` la mediana de $r_{rb}$ y la mediana de la diferencia pareada por comparador; añadir el IC 95 % de la diferencia de gap medio Bin-C vs Van-Exploit sobre los 27 casos.
- **Texto sugerido:** "Across the 27 cases, the instance-level mean gap difference between Binary-Complex and Vanilla-Exploitation is X pp (95% CI [a, b])." *(requiere cálculo)*

---

## 🟡 Media prioridad

### ☐ 7. Statistical Analysis está como subsección
- **Dónde:** `statistical_analysis.tex:1` (`\subsection{Statistical Analysis}`; la línea 4 dice "This subsection applies…")
- **Problema:** queda como 6.x dentro de Results, pero `introduction.tex:25` y `results.tex:14` la tratan como sección aparte.
- **Acción:** `\section{Statistical Analysis}`, "This section applies…", y subir un nivel las `\subsubsection`.

### ☐ 8. Notación triplicada y choques de símbolos
- **Dónde:** `background.tex:291-302`, `proposed_approach.tex:51-55, 81, 107-109, 137, 142`, `algorithm_gwo_de.tex:11`, `experimental_setup.tex:62`
- **Problema:**
  - Sección 3 usa $D_2,\theta_c,K_{max},N_{no\_improve},P$; sección 4 usa $D_C,\theta_C,\pi_{\max},\ell_{noimp},\tau_{pat}$; sección 5 mezcla ambas. Una frase de alias lo parchea y `eq:thresholds` repite `eq:theta_c`/`eq:theta_r`.
  - $\alpha,\beta$ son líderes GWO y también escala y centro de la sigmoide; $\pi$ es política y aparece en $\pi_{\max}$; $\boldsymbol{\theta}$ (configuración) convive con los umbrales $\theta$.
- **Acción:**
  - Una sola notación en las secciones 3–6: $D_R,D_C,\theta_R,\theta_C,\ell_{\mathrm{noimp}},\pi_{\max},\tau_{\mathrm{pat}}$. Eliminar la frase de alias y referenciar `eq:theta_c` en vez de redefinirla.
  - Escala $\sigma=2.0$ y centro $c_0=0.5$ (en el código, `SCALE` y `CENTER`).
  - Estrategia $\mathcal{S}$ en lugar de $\pi$.
  - Configuración $\boldsymbol{\phi}_t=(a_t,F_t,CR_t)$.
  - Renombrar $C_{\mathrm{plateau}}$ a $C_{\mathrm{const}}$: en el código, `cond_plateau` es el contador y la condición sobre $D_C$ es `cond_constant`.

### ☐ 9. Terminología alternada
- **Dónde:** `proposed_approach.tex:4, 27, 60, 63, 79, 144`; caption y cabecera de `tab:policies`; `algorithm_gwo_de.tex:11`
- **Problema:** "profile / configuration / parameter configurations / search mode" y "strategy / policy" se usan como sinónimos.
- **Acción:** "profile" para $(a,F,CR)$; "policy" para la regla de control, que es el término que usan las secciones 5–9; "mode" solo para $s_t$. Caption: "Control policies evaluated in this study"; cabecera "Policy".

### ☐ 10. AGSR con dos expansiones y siglas no definidas
- **Dónde:** `backmatter/abbreviations.tex:9` ("Annual Grid Surplus Ratio") y `background.tex:144` ("annual grid sales rate")
- **Acción:**
  - Unificar en "Annual Grid Sales Rate", el término de `li2024capacity`, y definirlo en el primer uso: "…the annual grid sales rate (AGSR) limit…".
  - Añadir a Abbreviations y definir al primer uso: LB2, BESS, NOCT, HHV, OAT, CMA-ES.
  - Unificar HRES2 / HRES2–H2.

### ☐ 11. Falta la limitación de baselines externos en MKP
- **Dónde:** `conclusions.tex:17`; contextualizar también en `sections/06_results/results.tex:21`
- **Problema:** gaps medios de 3.3–6.8 % en n=500, ninguna ejecución alcanza f\*, y todas las comparaciones son internas al híbrido.
- **Añadir (conclusions):**
  > Absolute MKP gaps remain far from the best-known values and no specialized MKP solver or established parameter-control scheme (e.g., success-history adaptation or a time-scheduled or random profile switch with the same number of changes) was included, so the comparison is internal to the hybrid.
- **Añadir (results):**
  > All comparators share the GWO–DE core; no standalone GWO, DE or state-of-the-art MKP solver is included, so the reported gaps characterize this hybrid under the stated budget rather than its competitiveness on the benchmark.

### ☐ 12. Parámetro $a$ de GWO constante, a diferencia del original
- **Dónde:** `proposed_approach.tex:20`
- **Añadir:**
  > Unlike the original GWO, $a$ is not decreased linearly over the run; it is held at the value of the active profile (Table~\ref{tab:profiles}) and changes only through the control policy.

### ☐ 13. Atribución causal excesiva a la regla de decisión
- **Dónde:** `proposed_approach.tex:144`
- **Actual:** "Because the six strategies share every other component, a difference in their results can be attributed to the decision rule."
- **Reemplazo:**
  > The six policies share every other component, so the design contrasts control rules, including their starting profile and monitor-reset behaviour, under a common budget, subject to the seed-pairing assumption of Section~\ref{subsec:reproduction}.

### ☐ 14. Data availability redactado en negativo
- **Dónde:** `backmatter/dataavailability.tex:1`; también `experimental_setup.tex:7` ("software versions … were not re-measured")
- **Actual:** "This repository link does not imply that all campaign records have been publicly deposited."
- **Acción:** depositar `runs.csv`, `statistics.csv` y `campaign_config.json` (MKP, sensibilidad, HRES2) y el generador del clima sintético, por ejemplo en Zenodo, y citar el DOI. Si algo no se deposita, dar el motivo y explicar cómo solicitarlo. Si `run.log` registra versiones de Python y NumPy, usarlas en lugar de "not re-measured".

### ☐ 15. Continuous-Complex: un único cruce sin explicación
- **Dónde:** `sections/08_discussion/discussion.tex:8`
- **Problema:** cruza 0.5 una sola vez y queda cerca de la exploración fija, pero sus medias se parecen a las de las otras adaptativas y no a Vanilla-Exploration. Además, "as in all its runs" en Results solo está respaldado para el conteo de cruces.
- **Añadir:**
  > Although Continuous-Complex remains above the 0.5 level after its single crossing, its means are closer to the other adaptive policies than to Vanilla-Exploration; the saved traces cannot tell whether the initial exploitation phase or the intermediate intensity explains this, because only one trace per case was retained.

### ☐ 16. Rechazos aislados compatibles con azar
- **Dónde:** `statistical_analysis.tex:12`
- **Añadir:**
  > Excluding Vanilla-Exploration, 9 of 108 unadjusted tests fall below 0.05, against about 5.4 expected under a global null, so these isolated rejections are compatible with chance.

### ☐ 17. Trabajo futuro desconectado de las limitaciones
- **Dónde:** `conclusions.tex:25-29`
- **Acción:** anteponer el párrafo siguiente y reducir o mover la línea multiobjetivo.
  > The most direct next step is an ablation that replaces the elastic distances with the window gain, the no-improvement counter and a Euclidean distance, together with standalone GWO and DE runs and a factorial sensitivity design, under equal budgets.

### ☐ 18. Contribución 4 sobredimensionada
- **Dónde:** `sections/01_introduction/introduction.tex:22`
- **Reemplazo:**
  > …comparing the adaptive policies with fixed-profile baselines through non-parametric statistical tests, complemented by a descriptive one-at-a-time sensitivity analysis of the monitor settings for the two binary policies on three MKP cases.

---

## 🟢 Baja prioridad / criterio del autor

- ☐ **19. Muletillas en Related Work** (`sections/02_related_work/related_work.tex:4, 6, 8, 10, 12, 14, 20`): eliminar los cierres "Collectively…", "Taken together…" e "In these…" de las líneas 8 y 10. En la línea 6, sustituir por "None of these mechanisms changes a parameter profile shared by the operators." Dejar un solo "Taken together" en la línea 20.
- ☐ **20. Vacío de investigación ausente en la Introducción** (`introduction.tex:8`):
  > These representations describe sampled locations or scalar improvement; none of the cited controllers compares the ordered best-so-far history with explicit progress and stagnation references. This work asks whether that comparison can drive the joint parameter profile of a hybrid metaheuristic.

  *(Confirmar que "none of the cited" es exacto.)*
- ☐ **21. Contribuciones 1 y 2 solapadas** (`introduction.tex:10-20`). Contribución 2:
  > A GWO–DE hybrid in which separately evaluated GWO and DE candidates compete for each individual of a shared population, so that a profile change alters both generators without discarding the population or the incumbent solution.
- ☐ **22. Baotou sugiere datos medidos** (`background.tex:34`):
  > …following the Baotou case of \cite{li2024capacity}. The hourly weather series used here is a synthetic typical year (Section~\ref{sec:protocol}), not measured site data.

  En `introduction.tex:10`: "built on the techno-economic model of \cite{li2024capacity} and evaluated on one synthetic typical-year scenario".
- ☐ **23. Banda en MKP** (`sections/05_experimental_setup/experimental_setup.tex:82`): `campaign_config.json` registra `"band": 4` explícito.
  > In both domains the recorded configuration specifies a half-width of $w=4$, which coincides with the ten-percent rule $\max(1,\lfloor 0.1W\rfloor)=4$.
- ☐ **24. Resultado por construcción presentado como hallazgo** (`results.tex:339`):
  > As expected by construction, the patience and $P_{\mathrm{high}}$ perturbations leave Binary-Simple unchanged, because that policy reads neither setting; for Binary-Complex, …
- ☐ **25. "Therefore" sin justificación** (`conclusions.tex:12`): "The contribution is a trajectory-based controller of the joint parameter profile… The evidence shows that it avoided the weaker fixed profile in both domains, but not that temporal alignment provides a uniform benefit…"
- ☐ **26. Cautelas repetidas:** "non-rejection is not equivalence" aparece en `statistical_analysis.tex:12`, en el párrafo de Bonferroni, en `discussion.tex:4` y en `conclusions.tex:6`; "training-free is not parameter-free" aparece en `introduction.tex:12` y en `experimental_setup.tex:82`. Dejar cada una una sola vez, en el protocolo estadístico y en las limitaciones.
- ☐ **27. Título** (`mdpi_gwo_de_dtw.tex:33-34`): "via Dynamic Time Warping", pero MKP usa solo DDTW y en HRES2 DTW = DDTW. Opción: "…via Elastic Distances to Progress and Plateau References", o mantener el título y declarar en el abstract que no se aisló la contribución del alineamiento.
- ☐ **28. Otros detalles:**
  - `background.tex:7`: "NP-hard" sin cita.
  - `related_work.tex:10`: generalización apoyada en un único trabajo de 2025.
  - Ecuaciones numeradas no referenciadas en el texto (`eq:gwo_leader_distance`, `eq:theta_c`).
- ☐ **29. Restos de plantilla y archivos:**
  - `mdpi_gwo_de_dtw.tex:7`: `\documentclass[journal,...]` sin revista definida.
  - Macro `\pending` definida y sin uso (línea 97).
  - Label `fig:mkp_gap` sin uso.
  - Figuras sin uso en `figura/`: `mkp-gap-by-case.*` y `mkp-temporal-mknapcb1-0-*`.
  - Entradas del `.bib` no citadas: `modu2023systematic`, `thirunavukkarasu2023comprehensive`.

---

## Fuera del manuscrito (código y datos que lo respaldan)

- ☐ **A. Script de estadísticas no versionado:** no se encontró en el repo el script que produce Wilcoxon exacto, Friedman/Bonferroni y la permutación $6/6^{12}$ (`tab:hres_paired`, `tab:bonferroni_global`). Versionarlo con `SHA256SUMS`.
- ☐ **B. Verificar el emparejamiento por semilla** (`experimental_setup.tex:120, 126`): regenerar las poblaciones iniciales de las semillas 1–31 y compararlas con la generación 0 de `best_run/temporal_series.csv`. Si coinciden, reemplazar la cautela por "Regenerated initial populations for seeds 1–31 match the stored generation-0 records."
- ☐ **C. Holm en `gwo_de_hres2/campaign.py:157-161`** cuenta pares idénticos (p=1.0), con m=15 frente a m=5 documentado en `campaign.py:36-37` y `README.md:40`. Además, compara políticas vanilla entre señales (DTW vs DDTW) aunque ignoran el monitor. Marcar `tests.csv` (Mann–Whitney/Holm) como histórico u obsoleto.
- ☐ **D. Efecto techo en HRES2:** `campaign.py:61-62` (`SERIES_KEYS`) descarta `best_series`. Para comprobar el techo hay que guardar `best_series` y medir evaluaciones hasta el objetivo, ERT o checkpoints (T=25/50/100/250), y comprobar si `battery_mw` sigue en la cota de 50 al ampliarla.
- ☐ **E. Procedencia de campañas:** `campaign_config.json` (HRES2 y MKP) no guarda git SHA, estado *dirty*, versiones de Python/NumPy/SciPy ni hash del clima.
- ☐ **F. Baselines faltantes:** GWO solo, DE solo y `gwo_de_sequential` con vanilla, todos con el mismo presupuesto de evaluaciones.
- **Nota:** `gwo_de_hres2/results/campaign_31_local` no existe en disco; solo está `campaign_31_mkp_monitor`.

---

## Orden sugerido

1. Correcciones de texto sin decisión científica: 1–4, 7–10, 12, 13, 18, 19, 23–26.
2. Limitaciones a declarar: 5, 11, 15, 16, 17.
3. Requieren cálculo o decisión del autor: 6, 14, 20–22, 27 y A–F.
