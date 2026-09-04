# Implementación CEC2022

`funciones_cec2022.py` utiliza los archivos oficiales de desplazamiento,
rotación y permutación ubicados en `input_data/`. Estos archivos provienen del
repositorio oficial [2022-SO-BO](https://github.com/P-N-Suganthan/2022-SO-BO).

La implementación admite la interfaz existente del proyecto:

- `cec2022_func(x, func_num, n_dim)` evalúa una función para un vector.
- `get_test_functions(10)` o `get_test_functions(20)` devuelve los 12
  descriptores usados por los experimentos.
- `get_cec2022_optimum_point(func_num, n_dim)` devuelve el primer vector de
  desplazamiento oficial.

No se generan matrices ni desplazamientos sustitutos. Si falta un archivo
oficial, la evaluación falla explícitamente para evitar resultados no
comparables. La traducción elimina el estado global mutable del script de
referencia y conserva sus ecuaciones, escalados, particiones, pesos y sesgos.

Verificación ejecutada:

```text
python continuous_benchmark/test_cec2022.py
```

Los 12 óptimos conocidos (`300, 400, ..., 2700`) se verifican en `D=10` y
`D=20`.
