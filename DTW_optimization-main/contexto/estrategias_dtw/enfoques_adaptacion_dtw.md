# Enfoques de Auto-Adaptación con DTW para MKP

## Contexto

El DTW (Dynamic Time Warping) se usa como **sensor** para medir qué tan estancada está la metaheurística. Compara la curva de convergencia real contra dos patrones de referencia:
- **D1**: distancia a la rampa ideal (progreso constante)
- **D2**: distancia a la meseta (estancamiento total)
- **delta = D1 - D2**: positivo = más cerca de meseta que de rampa

Las métricas clave disponibles son: D1, D2, delta, theta_c, theta_r, theta_delta, no_improve_len, trigger_streak.

---

## Enfoque A: Fire Binario (implementación actual del notebook)

### Cómo funciona
- 3 condiciones deben cumplirse **simultáneamente**:
  1. `cond_plateau`: `no_improve_len >= plateau_max`
  2. `cond_constant`: `D2 <= theta_c`
  3. `cond_ramp`: `D1 >= theta_r OR delta >= theta_delta`
- Si las 3 se cumplen `patience` veces consecutivas → `fire = True`
- fire=True → switch a params EXPLORE (fijos)
- fire=False → volver a EXPLOIT (fijos)

### Características
- Decisión **binaria**: estancado o no
- **Reactivo**: detecta estancamiento DESPUÉS de que ocurrió (window + patience iteraciones)
- 2 estados con parámetros **fijos** por estado
- Múltiples hiperparámetros: window, band, plateau_max, patience, min_slope, adapt_thresholds, p_low, p_high

### Limitaciones
- Desperdicia la información continua del DTW (reduce D1, D2, delta a booleanos)
- Respuesta idéntica para estancamiento leve y severo
- Puede oscilar entre EXPLOIT/EXPLORE en el borde

---

## Enfoque B: Multi-Estado con Delta/Theta (enfoque anterior de José)

### Cómo funciona
- Usa **delta y theta_delta directamente** como medidores
- Define múltiples rangos/estados basados en el valor de delta:
  - `delta < 0` → EXPLOIT fuerte (la curva progresa)
  - `0 < delta < θ₁` → EXPLOIT suave (progreso lento pero existe)
  - `θ₁ < delta < θ₂` → EXPLORE suave (estancamiento leve)
  - `delta > θ₂` → EXPLORE agresivo (estancamiento severo)
- Cada estado tiene su propio set de parámetros para la MH

### Diferencia clave con Fire Binario
| Aspecto | Fire Binario (A) | Multi-Estado (B) |
|---------|-----------------|-------------------|
| Decisión | Binaria (sí/no) | Discreta (4+ niveles) |
| Info de DTW usada | Reducida a bool | Rangos de delta |
| Granularidad de respuesta | 2 sets de params | 4+ sets de params |
| Sensibilidad | Requiere 3 condiciones + patience | Responde directo al valor de delta |
| Reactividad | Lenta (acumula evidencia) | Más rápida (lee delta cada iteración) |
| Complejidad | Más condiciones lógicas | Más rangos/umbrales que definir |
| Riesgo de oscilación | Bajo (patience amortigua) | Medio (puede saltar entre estados) |

### Ventajas sobre Fire Binario
- **Más granular**: distingue grados de estancamiento
- **Más reactivo**: no necesita patience para actuar
- **Más informativo**: usa el valor numérico de delta, no solo "es mayor que theta"

### Limitaciones
- Sigue siendo **discreto** (categorías, no continuo)
- Los umbrales θ₁, θ₂ son hiperparámetros que hay que tunear
- No escala suavemente — hay saltos al cruzar un umbral

---

## Enfoque C: Adaptación Proporcional/Continua (propuesta para investigación)

### Cómo funciona
- Normalizar delta a un score continuo [0, 1]:
  ```
  score = sigmoid(delta / scale_factor)
  # o: score = (delta - min_hist) / (max_hist - min_hist)
  # o: score basado en percentil del historial
  ```
- Interpolar parámetros proporcionalmente:
  ```
  w  = w_exploit  + score * (w_explore  - w_exploit)
  c1 = c1_exploit + score * (c1_explore - c1_exploit)
  c2 = c2_exploit - score * (c2_exploit - c2_explore)
  ```

### Diferencia con los anteriores
- **No hay estados discretos**: los parámetros se ajustan suavemente
- **Usa TODA la información del DTW**: el valor exacto de delta modula la intensidad
- **Proactivo**: empieza a ajustar ANTES del estancamiento total
- Elimina hiperparámetros: no necesita plateau_max, patience, ni umbrales de estado

### Variante: Ratio D1/D2
- En vez de delta, usar ratio = D1 / (D2 + ε)
- Se auto-normaliza (independiente de la escala del fitness)
- ratio > 1 → estancamiento, ratio < 1 → progreso

---

## Comparación de los 3 enfoques

```
                    Discreto                          Continuo
                    ────────────────────────────────  ──────────
Fire Binario (A) → |████████|                     | → 2 niveles
Multi-Estado (B) → |██|████|██████|██████████████| → 4+ niveles
Proporcional (C) → |▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓| → ∞ niveles (gradiente)
```

## Recomendación para paper

Implementar y comparar los 3:
- A (Fire) como **baseline** (ya existe en el notebook)
- B (Multi-Estado) como **evolución incremental**
- C (Proporcional) como **contribución principal**

Métricas de comparación: fitness final, gap al óptimo, convergencia speed, estabilidad entre seeds.
