"""
HRES2-H2/wpeb_model.py
----------------------
Modelo de simulación física y económica vectorizado y ultra-rápido para HRES2-H2.
Optimizaciones: Perfiles normalizados precalculados y bucle de despacho optimizado.
"""

from __future__ import annotations

import math
import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from continuous_benchmark.funciones_cec2022 import ContinuousFunction


# ── Configuración Predeterminada ───────────────────────────────────────────────

CONFIG_DEFAULT = {
    "lat": 41.70,
    "lon": 110.43,
    "total_generation_capacity_mw": 200.0,
    "agsr_max": 0.20,
    "electrolyzer_ratio_max": 0.50,

    "electrolyzer_unit_mw": 5.0,
    "min_electrolyzer_units": 10,  # 50 MW
    "max_electrolyzer_units": 20,  # 100 MW

    "battery_power_min_mw": 0.0,
    "battery_power_max_mw": 50.0,
    "battery_duration_candidates_h": [1.0, 2.0, 4.0],

    "h2_hhv_kwh_per_kg": 39.4,
    "electrolyzer_efficiency": 0.75,
    "electrolyzer_min_load_ratio": 0.30,

    "project_lifetime_years": 25,
    "real_discount_rate": 0.0435,

    "wind_turbine_rated_mw": 5.0,
    "wind_cut_in_ms": 2.5,
    "wind_rated_ms": 10.5,
    "wind_cut_out_ms": 25.0,

    "pv_dc_ac_ratio": 1.2,
    "pv_temp_coeff_pct_per_c": -0.5,
    "pv_noct_c": 47.0,
    "pv_stc_efficiency_pct": 13.0,
    "pv_derating_factor": 0.90,

    "converter_efficiency": 0.95,
    "battery_roundtrip_efficiency": 0.90,
}

COSTS_DEFAULT = {
    "wind_capex_cny_per_kw": 5917.0,
    "wind_replacement_cny_per_kw": 0.0,
    "wind_om_cny_per_kw_year": 40.2,
    "wind_life_years": 25,

    "pv_capex_cny_per_kw": 4633.0,
    "pv_replacement_cny_per_kw": 0.0,
    "pv_om_cny_per_kw_year": 17.6,
    "pv_life_years": 25,

    "electrolyzer_capex_cny_per_kw": 6964.0,
    "electrolyzer_replacement_cny_per_kw": 5969.14,
    "electrolyzer_om_cny_per_kw_year": 208.92,
    "electrolyzer_life_years": 15,

    "battery_capex_cny_per_kw": 2549.0,
    "battery_replacement_cny_per_kw": 500.0,
    "battery_om_cny_per_kw_year": 10.0,
    "battery_life_years": 10,
}


# ── Estructura de Resultado ───────────────────────────────────────────────────

@dataclass
class ExtendedSimulationResult:
    wind_mw: float
    pv_mw: float
    electrolyzer_mw: float
    n_el_units: int
    battery_mw: float
    battery_mwh: float
    battery_duration_h: float
    feasible: bool
    agsr: float
    total_wind_mwh: float
    total_pv_mwh: float
    total_renewable_mwh: float
    total_grid_sales_mwh: float
    total_electrolyzer_load_mwh: float
    electrolyzer_cf: float
    total_h2_kg: float
    npc_cny: float
    annualized_cost_cny: float
    lcoe_cny_per_kwh: float
    lcoh_cny_per_kg: float
    curtailment_mwh: float
    battery_throughput_mwh: float


# ── Modelos Financieros y Perfiles Precalculados ──────────────────────────────

def crf(i: float, n: int) -> float:
    return (i * (1 + i) ** n) / (((1 + i) ** n) - 1)


def present_value_of_replacements(capacity_kw: float, replacement_cost_per_kw: float,
                                  life_years: int, project_years: int, discount_rate: float) -> float:
    years = list(range(life_years, project_years, life_years))
    return sum((capacity_kw * replacement_cost_per_kw) / ((1 + discount_rate) ** y) for y in years)


def present_value_of_om(capacity_kw: float, om_per_kw_year: float,
                        project_years: int, discount_rate: float) -> float:
    return sum((capacity_kw * om_per_kw_year) / ((1 + discount_rate) ** y) for y in range(1, project_years + 1))


def npc_from_capacities(wind_mw: float, pv_mw: float, electrolyzer_mw: float,
                        battery_mw: float, config: Dict, costs: Dict) -> float:
    i = config["real_discount_rate"]
    n = config["project_lifetime_years"]

    def comp(cap_mw, capex, repl, om, life):
        cap_kw = cap_mw * 1000.0
        return (
            cap_kw * capex
            + present_value_of_replacements(cap_kw, repl, life, n, i)
            + present_value_of_om(cap_kw, om, n, i)
        )

    npc = 0.0
    npc += comp(wind_mw, costs["wind_capex_cny_per_kw"], costs["wind_replacement_cny_per_kw"],
                costs["wind_om_cny_per_kw_year"], costs["wind_life_years"])
    npc += comp(pv_mw, costs["pv_capex_cny_per_kw"], costs["pv_replacement_cny_per_kw"],
                costs["pv_om_cny_per_kw_year"], costs["pv_life_years"])
    npc += comp(electrolyzer_mw, costs["electrolyzer_capex_cny_per_kw"], costs["electrolyzer_replacement_cny_per_kw"],
                costs["electrolyzer_om_cny_per_kw_year"], costs["electrolyzer_life_years"])
    npc += comp(battery_mw, costs["battery_capex_cny_per_kw"], costs["battery_replacement_cny_per_kw"],
                costs["battery_om_cny_per_kw_year"], costs["battery_life_years"])
    return npc


def precalculate_profiles(df: pd.DataFrame, config: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """Precalcula los perfiles unitarios (por MW instalado) de generación eólica y solar."""
    # Solar
    ghi = df["ghi_kwh_m2"].values
    t_amb = df["t2m_c"].values
    t_cell = t_amb + (config["pv_noct_c"] - 20.0) / 0.8 * ghi
    gamma = config["pv_temp_coeff_pct_per_c"] / 100.0
    temp_factor = 1.0 + gamma * (t_cell - 25.0)
    rel_pv = ghi * config["pv_derating_factor"] * np.maximum(temp_factor, 0.0)
    pv_unit = np.clip(rel_pv, 0.0, 1.0)

    # Eólica
    ws = df["ws50m"].values
    cut_in = config["wind_cut_in_ms"]
    rated = config["wind_rated_ms"]
    cut_out = config["wind_cut_out_ms"]
    wind_unit = np.zeros_like(ws, dtype=float)
    m1 = (ws >= cut_in) & (ws < rated)
    m2 = (ws >= rated) & (ws < cut_out)
    wind_unit[m1] = (ws[m1]**3 - cut_in**3) / (rated**3 - cut_in**3)
    wind_unit[m2] = 1.0
    wind_unit = np.clip(wind_unit, 0.0, 1.0)

    return wind_unit, pv_unit


try:
    from numba import njit
except ImportError:
    def njit(func):
        return func


# ── Simulación Rápida Numba/Fast C-style ───────────────────────────────────────

@njit
def _fast_dispatch_simulation(
    p_gen: np.ndarray,
    electrolyzer_mw: float,
    battery_mw: float,
    battery_mwh: float,
    min_elz: float,
    eta_ch: float,
    eta_dis: float,
) -> Tuple[float, float, float, float]:

    """Bucle de despacho horario optimizado de alta velocidad."""
    soc = 0.0
    soc_max = battery_mwh
    p_batt_max = battery_mw

    total_grid = 0.0
    total_elz = 0.0
    curtailment = 0.0
    batt_tp = 0.0

    for gen in p_gen:
        p_elz = 0.0
        p_charge = 0.0
        p_discharge = 0.0
        p_grid = 0.0

        if gen >= min_elz and electrolyzer_mw > 0:
            p_elz = gen if gen <= electrolyzer_mw else electrolyzer_mw
        elif electrolyzer_mw > 0 and p_batt_max > 0 and soc > 0:
            max_dis = p_batt_max if p_batt_max < soc * eta_dis else soc * eta_dis
            if gen + max_dis >= min_elz:
                p_discharge = max_dis if max_dis < (electrolyzer_mw - gen) else (electrolyzer_mw - gen)
                p_elz = gen + p_discharge if (gen + p_discharge) < electrolyzer_mw else electrolyzer_mw

        surplus = gen + p_discharge - p_elz

        if surplus > 1e-12 and p_batt_max > 0 and soc_max > 0:
            max_ch_soc = (soc_max - soc) / eta_ch if eta_ch > 0 else 0.0
            p_charge = surplus
            if p_batt_max < p_charge:
                p_charge = p_batt_max
            if max_ch_soc < p_charge:
                p_charge = max_ch_soc

            soc += p_charge * eta_ch
            surplus -= p_charge
            batt_tp += p_charge

        if p_discharge > 0:
            soc -= p_discharge / eta_dis
            if soc < 0.0:
                soc = 0.0
            batt_tp += p_discharge

        if surplus > 1e-12:
            p_grid = surplus

        total_grid += p_grid
        total_elz += p_elz

    return total_grid, total_elz, curtailment, batt_tp


# ── Simulación Principal ──────────────────────────────────────────────────────

def simulate_wpeb_fast(
    wind_unit: np.ndarray,
    pv_unit: np.ndarray,
    wind_mw: float,
    n_el_units: int,
    battery_mw: float,
    battery_duration_h: float,
    config: Dict = CONFIG_DEFAULT,
    costs: Dict = COSTS_DEFAULT,
) -> ExtendedSimulationResult:
    total_capacity = config["total_generation_capacity_mw"]
    pv_mw = total_capacity - wind_mw
    electrolyzer_mw = n_el_units * config["electrolyzer_unit_mw"]
    battery_mwh = battery_mw * battery_duration_h

    if wind_mw < 0 or pv_mw < 0 or abs(wind_mw + pv_mw - total_capacity) > 1e-6:
        return ExtendedSimulationResult(wind_mw, pv_mw, electrolyzer_mw, n_el_units, battery_mw, battery_mwh,
                                        battery_duration_h, False, np.nan, 0,0,0,0,0,0,0,0,0,np.inf,np.inf,0,0)

    if n_el_units < config["min_electrolyzer_units"] or n_el_units > config["max_electrolyzer_units"]:
        return ExtendedSimulationResult(wind_mw, pv_mw, electrolyzer_mw, n_el_units, battery_mw, battery_mwh,
                                        battery_duration_h, False, np.nan, 0,0,0,0,0,0,0,0,0,np.inf,np.inf,0,0)

    if battery_mw < config["battery_power_min_mw"] or battery_mw > config["battery_power_max_mw"]:
        return ExtendedSimulationResult(wind_mw, pv_mw, electrolyzer_mw, n_el_units, battery_mw, battery_mwh,
                                        battery_duration_h, False, np.nan, 0,0,0,0,0,0,0,0,0,np.inf,np.inf,0,0)

    if battery_duration_h not in config["battery_duration_candidates_h"]:
        return ExtendedSimulationResult(wind_mw, pv_mw, electrolyzer_mw, n_el_units, battery_mw, battery_mwh,
                                        battery_duration_h, False, np.nan, 0,0,0,0,0,0,0,0,0,np.inf,np.inf,0,0)

    if electrolyzer_mw > config["electrolyzer_ratio_max"] * total_capacity:
        return ExtendedSimulationResult(wind_mw, pv_mw, electrolyzer_mw, n_el_units, battery_mw, battery_mwh,
                                        battery_duration_h, False, np.nan, 0,0,0,0,0,0,0,0,0,np.inf,np.inf,0,0)

    p_gen = wind_mw * wind_unit + pv_mw * pv_unit
    min_elz = config["electrolyzer_min_load_ratio"] * electrolyzer_mw

    eta_rt = config["battery_roundtrip_efficiency"]
    eta_ch = math.sqrt(eta_rt)
    eta_dis = math.sqrt(eta_rt)

    total_grid_sales_mwh, total_elz_mwh, curtailment_mwh, batt_tp = _fast_dispatch_simulation(
        p_gen=p_gen,
        electrolyzer_mw=electrolyzer_mw,
        battery_mw=battery_mw,
        battery_mwh=battery_mwh,
        min_elz=min_elz,
        eta_ch=eta_ch,
        eta_dis=eta_dis,
    )

    total_wind_mwh = float(np.sum(wind_mw * wind_unit))
    total_pv_mwh = float(np.sum(pv_mw * pv_unit))
    total_renewable_mwh = total_wind_mwh + total_pv_mwh
    electrolyzer_cf = total_elz_mwh / (electrolyzer_mw * len(p_gen)) if electrolyzer_mw > 0 else 0.0

    agsr = (total_grid_sales_mwh / total_renewable_mwh) if total_renewable_mwh > 0 else np.inf
    feasible = agsr <= config["agsr_max"]

    total_h2_kg = total_elz_mwh * 1000.0 * config["electrolyzer_efficiency"] / config["h2_hhv_kwh_per_kg"]

    npc_cny = npc_from_capacities(wind_mw, pv_mw, electrolyzer_mw, battery_mw, config, costs)
    annualized_cost_cny = npc_cny * crf(config["real_discount_rate"], config["project_lifetime_years"])

    delivered_kwh = (total_elz_mwh + total_grid_sales_mwh) * 1000.0
    lcoe = annualized_cost_cny / delivered_kwh if feasible and delivered_kwh > 0 else np.inf
    lcoh = annualized_cost_cny / total_h2_kg if feasible and total_h2_kg > 0 else np.inf

    return ExtendedSimulationResult(
        wind_mw=wind_mw,
        pv_mw=pv_mw,
        electrolyzer_mw=electrolyzer_mw,
        n_el_units=int(n_el_units),
        battery_mw=battery_mw,
        battery_mwh=battery_mwh,
        battery_duration_h=battery_duration_h,
        feasible=bool(feasible),
        agsr=float(agsr),
        total_wind_mwh=total_wind_mwh,
        total_pv_mwh=total_pv_mwh,
        total_renewable_mwh=total_renewable_mwh,
        total_grid_sales_mwh=total_grid_sales_mwh,
        total_electrolyzer_load_mwh=total_elz_mwh,
        electrolyzer_cf=float(electrolyzer_cf),
        total_h2_kg=float(total_h2_kg),
        npc_cny=float(npc_cny),
        annualized_cost_cny=float(annualized_cost_cny),
        lcoe_cny_per_kwh=float(lcoe),
        lcoh_cny_per_kg=float(lcoh),
        curtailment_mwh=float(curtailment_mwh),
        battery_throughput_mwh=float(batt_tp),
    )


def decode_solution(x: np.ndarray, config: Dict = CONFIG_DEFAULT) -> Dict:
    x_clip = np.copy(x)
    wind_mw = float(np.clip(x_clip[0], 0.0, config["total_generation_capacity_mw"]))

    n_el_raw = float(np.clip(x_clip[1], config["min_electrolyzer_units"], config["max_electrolyzer_units"]))
    n_el_units = int(round(n_el_raw))

    batt_raw = float(np.clip(x_clip[2], config["battery_power_min_mw"], config["battery_power_max_mw"]))
    battery_mw = float(round(batt_raw / config.get("battery_power_step_mw", 5.0)) * config.get("battery_power_step_mw", 5.0))

    dur_candidates = config["battery_duration_candidates_h"]
    idx_raw = float(np.clip(x_clip[3], 0.0, len(dur_candidates) - 1))
    dur_idx = int(round(idx_raw))
    battery_duration_h = float(dur_candidates[dur_idx])

    return {
        "wind_mw": wind_mw,
        "pv_mw": config["total_generation_capacity_mw"] - wind_mw,
        "n_el_units": n_el_units,
        "electrolyzer_mw": n_el_units * config["electrolyzer_unit_mw"],
        "battery_mw": battery_mw,
        "battery_duration_h": battery_duration_h,
        "battery_mwh": battery_mw * battery_duration_h,
    }


def generate_synthetic_tmy_8760() -> pd.DataFrame:
    np.random.seed(2008)
    hours = 8760
    timestamps = pd.date_range("2008-01-01 00:00", periods=hours, freq="h")
    t_hour = np.arange(hours)

    day_of_year = t_hour / 24.0
    solar_decl = 0.4 * np.sin(2 * np.pi * (day_of_year - 80) / 365.0)
    hour_of_day = t_hour % 24
    solar_elevation = np.maximum(0.0, np.sin(np.pi * (hour_of_day - 6) / 12.0)) * (0.6 + solar_decl)
    ghi = np.maximum(0.0, solar_elevation * 1.0 + np.random.normal(0, 0.05, hours))

    ws_base = np.random.weibull(2.1, hours) * 7.8
    ws_diurnal = 1.0 + 0.15 * np.cos(2 * np.pi * (hour_of_day - 2) / 24.0)
    ws50m = np.clip(ws_base * ws_diurnal, 0.0, 25.0)

    t2m_c = 10.0 + 15.0 * np.sin(2 * np.pi * (day_of_year - 100) / 365.0) + 5.0 * np.sin(2 * np.pi * (hour_of_day - 14) / 24.0)

    return pd.DataFrame({
        "timestamp": timestamps,
        "ws50m": ws50m,
        "ghi_kwh_m2": ghi,
        "t2m_c": t2m_c,
    })


# ── Clase Envolvente HRES2Function ────────────────────────────────────────────

class HRES2Function(ContinuousFunction):
    def __init__(self, df_year: pd.DataFrame | None = None, config: Dict = CONFIG_DEFAULT, costs: Dict = COSTS_DEFAULT):
        if df_year is None:
            df_year = generate_synthetic_tmy_8760()
        self.df_year = df_year
        self.config = config
        self.costs = costs

        self.wind_unit, self.pv_unit = precalculate_profiles(self.df_year, self.config)

        lb = np.array([0.0, config["min_electrolyzer_units"], config["battery_power_min_mw"], 0.0], dtype=float)
        ub = np.array([config["total_generation_capacity_mw"], config["max_electrolyzer_units"], config["battery_power_max_mw"], 2.0], dtype=float)

        super().__init__(
            name="HRES2_H2_WPEB",
            func=self._evaluate,
            n_dim=4,
            lb=lb[0],
            ub=ub[0],
            optimum=0.30,
        )
        self.lb_vector = lb
        self.ub_vector = ub

    def _evaluate(self, x: np.ndarray) -> float:
        d = decode_solution(x, self.config)
        res = simulate_wpeb_fast(
            wind_unit=self.wind_unit,
            pv_unit=self.pv_unit,
            wind_mw=d["wind_mw"],
            n_el_units=d["n_el_units"],
            battery_mw=d["battery_mw"],
            battery_duration_h=d["battery_duration_h"],
            config=self.config,
            costs=self.costs,
        )
        if res.feasible:
            return res.lcoe_cny_per_kwh
        else:
            agsr_val = 1.0 if np.isnan(res.agsr) else res.agsr
            return 100.0 + 10.0 * agsr_val

    def get_info(self, x: np.ndarray) -> Dict:
        d = decode_solution(x, self.config)
        res = simulate_wpeb_fast(
            wind_unit=self.wind_unit,
            pv_unit=self.pv_unit,
            wind_mw=d["wind_mw"],
            n_el_units=d["n_el_units"],
            battery_mw=d["battery_mw"],
            battery_duration_h=d["battery_duration_h"],
            config=self.config,
            costs=self.costs,
        )
        return asdict(res)
