# Futuro MCDP

Esta carpeta contiene todo el material MCDP retirado del flujo activo:

- `IMPLEMENTACION_VARIANTES_DTW_MCDP.md`: plan histórico para MCDP;
- `codigo/mcdp_core/`: modelo, datos y utilidades MCDP;
- `codigo/woa_abc/`: solver cooperativo WOA--ABC, runners y óptimos;
- `resultados/`: corridas y tablas generadas previamente;
- `README_WOA_ABC_HISTORICO.md`: documentación histórica del híbrido.

El paquete activo `woa_abc` ya no importa este código. Para retomarlo en el
futuro, trabajar desde `planes/futuro_mcdp/codigo/` y crear una integración
explícita, sin mezclar sus resultados con CEC2022 o HRES2.
