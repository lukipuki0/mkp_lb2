# Instancias MKP: OR-Library (Chu-Beasley)

## Descripción

Las instancias de Chu-Beasley son el **benchmark estándar** para el Multidimensional Knapsack Problem (MKP). Fueron creadas por P. C. Chu y J. E. Beasley en 1998 y están disponibles en OR-Library.

## Instancias Disponibles

| Archivo | n (ítems) | m (restricciones) | Tightness (α) | Instancias | Dificultad |
|---------|-----------|-------------------|----------------|------------|------------|
| `mknapcb1` | 100 | 5 | 0.25 | 30 | Media |
| `mknapcb2` | 100 | 5 | 0.50 | 30 | Media-baja |
| `mknapcb3` | 100 | 5 | 0.75 | 30 | Baja |
| `mknapcb4` | 100 | 10 | 0.25 | 30 | Alta |
| `mknapcb5` | 100 | 10 | 0.50 | 30 | Media |
| `mknapcb6` | 100 | 10 | 0.75 | 30 | Media-baja |
| `mknapcb7` | 250 | 5 | 0.25 | 30 | Alta |
| `mknapcb8` | 250 | 5 | 0.50 | 30 | Media |
| `mknapcb9` | 250 | 5 | 0.75 | 30 | Media-baja |

**Total: 270 instancias** (9 archivos × 30 instancias cada uno)

## ¿Qué es el Tightness (α)?

El tightness define qué tan **restrictivas** son las capacidades:

```
capacity[i] = α × sum(weights[i])
```

- **α = 0.25** → Capacidad = 25% del total → MUY restrictivo → pocas soluciones factibles → MÁS DIFÍCIL
- **α = 0.50** → Capacidad = 50% del total → Moderado
- **α = 0.75** → Capacidad = 75% del total → Relajado → muchas soluciones factibles → MENOS DIFÍCIL

## Formato del Archivo

Cada archivo contiene 30 instancias con el siguiente formato:

```
30                          ← número de instancias en el archivo

100 5 24381                 ← n ítems, m restricciones, óptimo conocido
 92  81  98  54 ... 42      ← profits (pueden abarcar varias líneas)
 67  23  ...                ← weights restricción 1
 45  12  ...                ← weights restricción 2
 ...                        ← weights restricción m
 2137  1546  ...            ← capacities (m valores)

100 5 24274                 ← siguiente instancia
...
```

## Descarga

Fuente oficial: [OR-Library](http://people.brunel.ac.uk/~mastjjb/jeb/orlib/mknapinfo.html)

## Referencia

Chu, P. C., & Beasley, J. E. (1998). "A Genetic Algorithm for the Multidimensional Knapsack Problem." *Journal of Heuristics*, 4(1), 63-86.

## Métricas de Evaluación

Para comparar resultados contra el óptimo conocido:

```
RPD = 100 × (óptimo - best_fitness) / óptimo
```

- **RPD = 0%** → Encontró el óptimo exacto
- **RPD < 1%** → Solución excelente
- **RPD < 5%** → Solución buena
- **RPD > 10%** → Solución deficiente

Se reporta el **RPD promedio** sobre las 30 instancias de cada archivo y las 30 corridas de cada instancia.
