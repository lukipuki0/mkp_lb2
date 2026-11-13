# ASOC DDTW scalar recovery

Original campaign not completed; scalar DDTW recovery only. No optimization, inference or telemetry reconstruction. Statistics include all 31 physical LCOEs; best is the minimum feasible LCOE, ties by run ID.

- DDTW: 31 records; excluded baselines: 71.
- Feasible: 31/31.
- Physical LCOE (CNY/kWh): mean 0.26715945105709271; sample std (ddof=1) 1.9724747946685063e-14; min 0.26715945105708805; max 0.2671594510571968.
- Unavailable telemetry: convergence, DDTW, Gantt, switches.
- Source paths and SHA-256 hashes: see summary.json.

## Best feasible saved solution: run 1, seed 43

Physical LCOE: 0.26715945105708805; LCOH: 17.631389137532363 CNY/kg; AGSR: 0.19999999999999982.

- wind_mw: 174.47315399420154
- pv_mw: 25.52684600579846
- n_el_units: 14
- electrolyzer_mw: 70.0
- battery_mw: 50.0
- battery_duration_h: 4.0
- battery_mwh: 200.0
- total_h2_kg: 8862879.446117368
- electrolyzer_cf: 0.7592899547228187
