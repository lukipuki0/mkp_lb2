# MH Pendientes de Implementar

Estas metaheurísticas están planificadas para completar los pools a 4 algoritmos de cada tipo.

## Poblacionales (falta 1)
- [ ] `de_mkp/` — **Differential Evolution**: mutación diferencial + cruce binomial.

## Trayectoria (faltan 2)
- [ ] `hc_mkp/` — **Hill Climbing con reinicios**: explotación pura 1-flip.
- [ ] `ils_mkp/` — **Iterated Local Search**: HC + perturbación aleatoria.

## Nota
Para agregar una nueva MH al pipeline, una vez implementada:
1. Agregarle `sol_inicial` o `sol_inyectada` y `stag_strategy="abort"`.
2. Incluirla en `POOL_POBLACIONAL` o `POOL_TRAYECTORIA` en `hybrid_mkp/orchestrator.py`.
3. Agregar su dispatcher en la función `_ejecutar_mh` del mismo archivo.
