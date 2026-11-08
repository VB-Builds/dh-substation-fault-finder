"""Simulator of district-heating customer substations with injected, labelled faults.

Each substation has a space-heating (SH) heat exchanger, a domestic hot water (DHW) heat exchanger
and a DHW circulation loop behind one heat meter. The model is quasi-steady, hourly:

* building heat demand follows outdoor temperature (12 h lag), internal gains and a night setback;
* radiator emitters set the secondary supply/return temperatures (exponent n = 1.3);
* the primary return temperature of the SH exchanger follows from UA and the log-mean temperature
  difference (counter-flow), so a smaller UA or a hotter secondary side gives a hotter return;
* DHW and circulation loads add low-temperature and high-temperature return streams;
* the heat meter sees the flow-weighted mix, plus sensor noise.

Everything here is SIMULATED. Weather is real (see ``weather.py``); the substations are not.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from .weather import load_hourly_temperature

C_WATER = 1.163  # kWh / (m3 K)

FAULT_TYPES = (
    "fouled_hex",
    "bypass",
    "stuck_valve",
    "wrong_curve",
    "dhw_circulation",
    "sensor_swap",
    "sensor_drift",
    "no_setback",
)
HEALTHY = "healthy"
# Faults whose cost is physical (hot water returned to the network). Sensor faults are billing/metering
# errors and have no physical return-temperature penalty.
PHYSICAL_FAULTS = ("fouled_hex", "bypass", "stuck_valve", "wrong_curve", "dhw_circulation", "no_setback")

# earliest / latest onset day per fault, counted from the start of the test year (0-based).
ONSET_RANGE = {
    "stuck_valve": (15, 120),
    "default": (30, 280),
}


@dataclass
class SimConfig:
    n_substations: int = 240
    faults_per_type: int = 5
    seed: int = 1
    tps_design: float = 84.0


def _lmtd_cold_end(q_over_ua: np.ndarray, d1: np.ndarray) -> np.ndarray:
    """Solve LMTD(d1, d2) = L for the cold-end temperature difference d2 (vectorised bisection)."""
    lo = np.full_like(d1, 1e-3)
    hi = d1.copy()
    for _ in range(45):
        mid = 0.5 * (lo + hi)
        ratio = np.clip(d1 / mid, 1.0 + 1e-9, None)
        lm = np.where(np.abs(d1 - mid) < 1e-6, d1, (d1 - mid) / np.log(ratio))
        too_low = lm < q_over_ua
        lo = np.where(too_low, mid, lo)
        hi = np.where(too_low, hi, mid)
    return 0.5 * (lo + hi)


def lmtd(d1: np.ndarray, d2: np.ndarray) -> np.ndarray:
    d1 = np.asarray(d1, dtype=float)
    d2 = np.asarray(d2, dtype=float)
    same = np.abs(d1 - d2) < 1e-9
    safe = np.where(same, d1 + 1.0, d2)
    return np.where(same, d1, (d1 - safe) / np.log(d1 / safe))


def effectiveness_ntu(ua: np.ndarray, c_p: np.ndarray, c_s: np.ndarray) -> np.ndarray:
    """Counter-flow heat-exchanger effectiveness."""
    cmin = np.minimum(c_p, c_s)
    cmax = np.maximum(c_p, c_s)
    cr = cmin / cmax
    ntu = ua / cmin
    e = np.exp(-ntu * (1 - cr))
    with np.errstate(invalid="ignore", divide="ignore"):
        general = (1 - e) / (1 - cr * e)
    return np.where(np.abs(1 - cr) < 1e-6, ntu / (1 + ntu), general)


def _dhw_profile(index: pd.DatetimeIndex) -> np.ndarray:
    h = index.hour.to_numpy()
    wk = index.day_of_week.to_numpy() >= 5
    base = np.full(len(index), 0.25)
    base += 2.0 * np.exp(-0.5 * ((h - 7.0) / 1.2) ** 2) * np.where(wk, 0.7, 1.0)
    base += 1.6 * np.exp(-0.5 * ((h - 19.0) / 1.8) ** 2)
    base += np.where(wk, 0.9 * np.exp(-0.5 * ((h - 10.5) / 2.0) ** 2), 0.0)
    return base / base.mean()


def _ema(a: np.ndarray, alpha: float) -> np.ndarray:
    """Exponential moving average along the last axis, started at the first sample."""
    a = np.asarray(a, dtype=float)
    zi = (1.0 - alpha) * a[..., :1]
    return lfilter([alpha], [1.0, -(1.0 - alpha)], a, axis=-1, zi=zi)[0]


def make_population(cfg: SimConfig) -> pd.DataFrame:
    """Draw substation parameters and assign faults (``faults_per_type`` of each type)."""
    rng = np.random.default_rng(cfg.seed)
    n = cfg.n_substations
    n_fault = cfg.faults_per_type * len(FAULT_TYPES)
    if n_fault > n:
        raise ValueError("more faulty substations requested than substations")
    q_d = np.exp(rng.uniform(np.log(40), np.log(700), n))
    ts_d = rng.choice([55.0, 60.0, 65.0, 70.0], n, p=[0.25, 0.3, 0.3, 0.15])
    spread_d = rng.uniform(16, 25, n)
    approach = rng.uniform(3.5, 9.0, n)
    pop = pd.DataFrame(
        {
            "substation": [f"S{i:03d}" for i in range(n)],
            "q_design_kw": q_d,
            "ts_design": ts_d,
            "tr_design_sec": ts_d - spread_d,
            "approach_design": approach,
            "gains_k": rng.uniform(4.5, 7.5, n),
            "dhw_share": rng.uniform(0.05, 0.14, n),
            "circ_share": rng.uniform(0.008, 0.02, n),
            "dhw_approach": rng.uniform(11, 17, n),
            "setback_k": rng.uniform(2.5, 4.0, n),
        }
    )
    order = rng.permutation(n)
    fault = np.array([HEALTHY] * n, dtype=object)
    onset = np.full(n, -1)
    severity = np.zeros(n)
    k = 0
    for ftype in FAULT_TYPES:
        lo, hi = ONSET_RANGE.get(ftype, ONSET_RANGE["default"])
        for _ in range(cfg.faults_per_type):
            i = order[k]
            k += 1
            fault[i] = ftype
            onset[i] = int(rng.integers(lo, hi + 1))
            severity[i] = rng.uniform(0.0, 1.0)
    pop["fault"] = fault
    pop["onset_day"] = onset
    pop["severity"] = severity
    return pop


def _fault_arrays(pop: pd.DataFrame, n_hours: int, healthy: bool, ref_days: int) -> dict[str, np.ndarray]:
    """Time-varying fault multipliers, shape (N, T)."""
    n = len(pop)
    t = np.arange(n_hours)[None, :]
    onset_h = ((pop["onset_day"].to_numpy() + ref_days) * 24)[:, None]
    sev = pop["severity"].to_numpy()[:, None]
    ftype = pop["fault"].to_numpy()[:, None]
    after = (t >= onset_h) & (onset_h >= 0)
    if healthy:
        after = np.zeros_like(after)

    def ramp(days: float) -> np.ndarray:
        return np.clip((t - onset_h) / (days * 24.0), 0.0, 1.0) * after

    ua_scale = 1.0 - ramp(60) * (ftype == "fouled_hex") * (0.40 + 0.25 * sev)
    bypass = after * (ftype == "bypass") * (0.10 + 0.15 * sev)
    stuck = after * (ftype == "stuck_valve")
    stuck_phi = 0.25 + 0.25 * sev
    curve_shift = after * (ftype == "wrong_curve") * (5.0 + 7.0 * sev)
    circ_mult = 1.0 + after * (ftype == "dhw_circulation") * (2.5 + 2.5 * sev)
    swap = after * (ftype == "sensor_swap")
    drift = ramp(90) * (ftype == "sensor_drift") * (3.0 + 4.0 * sev)
    setback_off = after * (ftype == "no_setback")
    return {
        "ua_scale": ua_scale,
        "bypass": bypass,
        "stuck": stuck.astype(bool),
        "stuck_phi": np.broadcast_to(stuck_phi, (n, n_hours)),
        "curve_shift": curve_shift,
        "circ_mult": circ_mult,
        "swap": swap.astype(bool),
        "drift": drift,
        "setback_off": setback_off.astype(bool),
    }


def _physics(
    pop: pd.DataFrame,
    t_out: np.ndarray,
    index: pd.DatetimeIndex,
    fa: dict[str, np.ndarray],
    noise: dict[str, np.ndarray],
    tps_design: float,
) -> dict[str, np.ndarray]:
    n, n_hours = len(pop), len(index)
    col = lambda name: pop[name].to_numpy()[:, None]  # noqa: E731
    q_d = col("q_design_kw")
    ts_d, tr_d = col("ts_design"), col("tr_design_sec")
    spread_d = ts_d - tr_d
    ex_d = 0.5 * (ts_d + tr_d) - 20.0

    t_eff = np.broadcast_to(_ema(t_out[None, :], 1.0 / 12.0), (n, n_hours))
    hour = index.hour.to_numpy()[None, :]
    night = (hour >= 22) | (hour < 5)
    setback = np.where(night & ~fa["setback_off"], col("setback_k"), 0.0)
    t_in = 21.0 - setback
    gains = col("gains_k")
    t_des_out = -8.0
    x = np.clip((t_in - gains - t_eff) / (21.0 - gains - t_des_out), 0.0, 1.2)
    x = _ema(x, 0.5)
    x = np.where(x < 0.02, 0.0, x)
    x_noise = np.exp(noise["load"])
    shift = fa["curve_shift"]
    x_q = x * x_noise * (1.0 + 0.03 * shift * (x > 0))

    heating = x > 0
    xs = np.where(heating, x, 1.0)
    mean_ex = ex_d * xs ** (1 / 1.3) + shift
    spread = spread_d * xs**0.6 * np.where(shift > 0, 0.8, 1.0)
    ts_out = np.where(heating, 20.0 + mean_ex + 0.5 * spread, 20.0)
    ts_in = np.where(heating, 20.0 + mean_ex - 0.5 * spread, 20.0)

    tps = np.clip(68.0 + 0.9 * (10.0 - t_eff), 68.0, 88.0) + noise["tps"]
    tps_d = tps_design
    ua_nom = q_d / lmtd(tps_d - ts_d, np.broadcast_to(col("approach_design"), ts_d.shape))

    q_x = x_q * q_d
    d1 = np.clip(tps - ts_out, 1.0, None)
    l_target = np.where(heating, q_x / np.where(heating, ua_nom * fa["ua_scale"], 1.0), 0.0)
    l_max = 0.985 * d1
    l_eff = np.minimum(l_target, l_max)
    q_sh = np.where(l_target > l_max, l_eff * ua_nom * fa["ua_scale"], q_x)
    d2 = _lmtd_cold_end(np.maximum(l_eff, 1e-6), d1)
    tpr_sh = np.where(heating, ts_in + d2, tps)

    # stuck valve: primary flow frozen at a fraction of the design flow while the building heats
    v_design = q_d / (C_WATER * (tps_d - (tr_d + col("approach_design"))))
    stuck = fa["stuck"] & heating
    if stuck.any():
        c_p = C_WATER * fa["stuck_phi"] * v_design
        c_s = np.where(heating, q_x / np.maximum(ts_out - ts_in, 1.0), 1.0)
        eps = effectiveness_ntu(ua_nom * fa["ua_scale"], c_p, np.maximum(c_s, 1e-3))
        q_stk = eps * np.minimum(c_p, c_s) * (tps - ts_in)
        tpr_stk = tps - q_stk / c_p
        q_sh = np.where(stuck, q_stk, q_sh)
        tpr_sh = np.where(stuck, tpr_stk, tpr_sh)

    dhw_profile = _dhw_profile(index)[None, :]
    p_dhw = q_d * col("dhw_share") * dhw_profile * np.exp(noise["dhw"])
    tpr_dhw = 10.0 + col("dhw_approach") + 0.06 * (tps - 70.0)
    p_circ = q_d * col("circ_share") * fa["circ_mult"]
    tpr_circ = np.where(fa["circ_mult"] > 1.0, 55.0, 50.0) + 0.0 * tps

    v_sh = q_sh / (C_WATER * np.maximum(tps - tpr_sh, 1.0))
    v_dhw = p_dhw / (C_WATER * np.maximum(tps - tpr_dhw, 1.0))
    v_circ = p_circ / (C_WATER * np.maximum(tps - tpr_circ, 1.0))
    v_hex = v_sh + v_dhw + v_circ
    tr_hex = (v_sh * tpr_sh + v_dhw * tpr_dhw + v_circ * tpr_circ) / v_hex
    beta = fa["bypass"]
    v_byp = v_hex * beta / (1.0 - beta)
    v = v_hex + v_byp
    tr = (v_hex * tr_hex + v_byp * tps) / v
    q = q_sh + p_dhw + p_circ
    return {"t_supply": tps, "t_return": tr, "flow_m3h": v, "power_kw": q, "heating": heating}


def _draw_noise(n: int, n_hours: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    load = rng.normal(0, 0.05, (n, n_hours))
    load = _ema(load, 0.3) * 1.6
    return {
        "load": load,
        "dhw": rng.normal(0, 0.35, (n, n_hours)),
        "tps": rng.normal(0, 0.3, (n, n_hours)),
        "m_ts": rng.normal(0, 0.15, (n, n_hours)),
        "m_tr": rng.normal(0, 0.15, (n, n_hours)),
        "m_v": rng.normal(0, 0.01, (n, n_hours)),
    }


@dataclass
class Network:
    """A simulated network: long-format meter data, substation table and ground truth."""

    meters: pd.DataFrame  # index: (substation, time); t_out, t_supply, t_return, flow_lph, power_kw, energy_kwh
    pop: pd.DataFrame  # per substation parameters and fault labels
    truth: pd.DataFrame  # per substation true excess return temperature x flow over the year
    km3_daily: np.ndarray  # (N, days) true excess K*m3 per day (counterfactual healthy minus faulty)
    config: SimConfig
    ref_days: int = 365


def simulate_network(
    cfg: SimConfig | None = None, n_days: int = 365, ref_days: int = 365
) -> Network:
    """Simulate a network over real hourly Swiss outdoor temperature (see ``weather.py``).

    The first ``ref_days`` (2022) are fault-free history used to learn each substation's baseline.
    Faults are injected during the following ``n_days`` (2023); ``onset_day`` counts from its start.
    """
    cfg = cfg or SimConfig()
    wx = load_hourly_temperature()
    wx = wx.iloc[: (ref_days + n_days) * 24]
    index = pd.DatetimeIndex(wx.index)
    t_out = wx["temperature_2m"].to_numpy(dtype=float)
    pop = make_population(cfg)
    n, n_hours = len(pop), len(index)
    rng = np.random.default_rng(cfg.seed + 1000)
    noise = _draw_noise(n, n_hours, rng)

    fa_f = _fault_arrays(pop, n_hours, healthy=False, ref_days=ref_days)
    fa_h = _fault_arrays(pop, n_hours, healthy=True, ref_days=ref_days)
    faulty = _physics(pop, t_out, index, fa_f, noise, cfg.tps_design)
    # counterfactual (same noise, no fault) is only needed for substations that have a fault
    fidx = np.flatnonzero(pop["fault"].to_numpy() != HEALTHY)
    sub = lambda d: {k: v[fidx] for k, v in d.items()}  # noqa: E731
    healthy = _physics(pop.iloc[fidx], t_out, index, sub(fa_h), sub(noise), cfg.tps_design)

    ts_m = faulty["t_supply"] + noise["m_ts"]
    tr_m = faulty["t_return"] + noise["m_tr"] + fa_f["drift"]
    swap = fa_f["swap"]
    ts_rep = np.where(swap, tr_m, ts_m)
    tr_rep = np.where(swap, ts_m, tr_m)
    v_m = faulty["flow_m3h"] * (1.0 + noise["m_v"])
    p_m = C_WATER * v_m * (ts_rep - tr_rep)
    p_m = np.maximum(p_m, 0.0)
    energy = np.cumsum(p_m, axis=1)

    mi = pd.MultiIndex.from_product([pop["substation"].tolist(), index], names=["substation", "time"])
    meters = pd.DataFrame(
        {
            "t_out": np.tile(t_out, n),
            "t_supply": ts_rep.ravel(),
            "t_return": tr_rep.ravel(),
            "flow_lph": (v_m * 1000.0).ravel(),
            "power_kw": p_m.ravel(),
            "energy_kwh": energy.ravel(),
        },
        index=mi,
    ).astype(np.float32)

    excess = np.zeros_like(faulty["t_return"])
    excess[fidx] = np.clip(faulty["t_return"][fidx] - healthy["t_return"], 0.0, None)
    v_true = faulty["flow_m3h"]
    km3 = excess * v_true  # K*m3 per hour
    truth = pd.DataFrame(
        {
            "substation": pop["substation"],
            "true_excess_km3_year": km3[:, ref_days * 24 :].sum(axis=1),
            "true_excess_mean_k": km3[:, ref_days * 24 :].sum(axis=1)
            / v_true[:, ref_days * 24 :].sum(axis=1),
        }
    )
    km3_daily = km3.reshape(n, ref_days + n_days, 24).sum(axis=2)[:, ref_days:]
    return Network(meters=meters, pop=pop, truth=truth, km3_daily=km3_daily, config=cfg, ref_days=ref_days)
