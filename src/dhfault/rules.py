"""Rule library: physics-based fault rules with plain-language explanations.

Rules operate on the rolling-window features of ``features.py``. ``z`` features are divided by the
substation's own spread on fault-free history, so thresholds are in "sigmas of this substation".
A rule fires when all its conditions hold; if several fire, the one with the highest score wins
(score = the smallest margin of its conditions, so a rule needs *all* of its evidence).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .sim import FAULT_TYPES, HEALTHY

LABELS = (HEALTHY, *FAULT_TYPES)
LABEL_CODE = {name: i for i, name in enumerate(LABELS)}


@dataclass
class RuleConfig:
    """Thresholds. Defaults were set on a separate tuning network (seed 11), not the evaluation one."""

    min_rt_k: float = 1.2  # minimum excess return temperature (K) for a return-temperature flag
    z_rt: float = 4.0  # ... and in sigmas of the substation's own fault-free spread
    z_flow_up: float = 3.0
    z_energy_flat: float = 4.0  # |z| below this counts as "energy unchanged"
    z_energy_up: float = 3.5
    z_energy_down: float = -1.5
    bypass_min_rt_k: float = 4.5  # all-hours excess above this, energy unchanged: bypass
    neg_dt_frac: float = 0.3
    z_night: float = 5.5
    z_base_load: float = 6.0  # low-load energy/flow deviation for runaway base load (circulation, stuck valve)
    z_circ_night: float = 7.0
    extra: dict = field(default_factory=dict)


EXPLANATIONS = {
    "fouled_hex": (
        "Return temperature is {rT:.1f} K above this substation's own signature while energy is "
        "unchanged ({rP:+.0%}); excess grows with load. Likely fouled/scaled heat exchanger."
    ),
    "bypass": (
        "Return temperature is {rT:.1f} K above signature at all loads, flow is {rV:+.0%} while energy is "
        "unchanged ({rP:+.0%}). Likely open bypass / short-circuit between supply and return."
    ),
    "stuck_valve": (
        "Flow stays flat while demand changes: flow at high load {rVh:+.0%}, night energy {rPn:+.0%}. "
        "Likely control valve stuck (open or partly open)."
    ),
    "wrong_curve": (
        "Energy {rP:+.0%} and return temperature {rT:+.1f} K versus signature: building is overheated. "
        "Likely heating curve too high / parallel shift."
    ),
    "dhw_circulation": (
        "Constant extra heat draw (low-load energy {rPl:+.0%}) with return temperature {rT:+.1f} K. "
        "Likely DHW circulation loop short-circuit or runaway pump."
    ),
    "sensor_swap": (
        "Supply minus return temperature is below 1 K in {neg:.0%} of hours (negative energy). "
        "Likely supply and return temperature sensors swapped."
    ),
    "sensor_drift": (
        "Return temperature reads {rT:.1f} K above signature while flow is normal ({rV:+.0%}) and "
        "metered energy is {rP:+.0%}. Likely return temperature sensor drift (billing error)."
    ),
    "no_setback": (
        "Night energy is {rPn:+.0%} versus signature while daytime is {rPd:+.0%}. "
        "Likely night setback not active."
    ),
}


def _g(feats: dict[str, np.ndarray], key: str) -> np.ndarray:
    return np.nan_to_num(feats[key], nan=0.0)


def apply_rules(
    raw: dict[str, np.ndarray],
    z: dict[str, np.ndarray],
    cfg: RuleConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (label_code, score), each shape (N, D). Code 0 is healthy.

    Evidence is read separately for all hours, low-load hours (summer base load) and high-load hours
    (winter), because different faults show up in different regimes.
    """
    cfg = cfg or RuleConfig()

    def zv(k):
        return np.nan_to_num(z[k], nan=0.0)

    def hot(tag):
        r = np.nan_to_num(raw[f"rT_{tag}"], nan=0.0)
        return (r > cfg.min_rt_k) & (zv(f"rT_{tag}") > cfg.z_rt)

    a_all, a_lo, a_hi = hot("all"), hot("lo"), hot("hi")
    any_hot = a_all | a_lo | a_hi
    rt_all = np.nan_to_num(raw["rT_all"], nan=0.0)
    flow_up = (zv("rV_all") > cfg.z_flow_up) | (zv("rV_lo") > cfg.z_flow_up) | (zv("rV_hi") > cfg.z_flow_up)
    energy_flat = (
        (np.abs(zv("rP_all")) < cfg.z_energy_flat)
        & (np.abs(zv("rP_hi")) < cfg.z_energy_flat)
        & (zv("rP_lo") < cfg.z_base_load)
    )
    neg = np.nan_to_num(raw["neg_dT"], nan=0.0)
    n = len(LABELS)
    sc = np.full((n,) + rt_all.shape, -1.0)

    def put(name, ok, base, boost=0.0):
        sc[LABEL_CODE[name]] = np.where(ok, base + np.clip(boost, 0.0, 0.5), -1.0)

    put("sensor_swap", neg > cfg.neg_dt_frac, 5.0)
    drift = any_hot & ~flow_up & ((zv("rP_all") < cfg.z_energy_down) | (zv("rP_hi") < -3.0))
    put("sensor_drift", drift, 4.0, zv("rT_all") / 100)
    circ = (
        (a_lo | a_all)
        & (zv("rP_night") > cfg.z_circ_night)
        & (zv("rP_day") > 3.0)
        & ((zv("rP_lo") > cfg.z_base_load) | ~np.isfinite(raw["rP_lo"]))
    )
    put("dhw_circulation", circ, 3.0, zv("rP_night") / 100)
    # stuck valve: flow no longer follows demand, so energy and flow run high all day while the
    # return temperature does NOT rise (unlike a wrong heating curve or a bypass)
    stuck = ~any_hot & (
        ((zv("rP_day") > 5.0) & (zv("rP_all") > 5.0) & (zv("rV_all") > 5.0) & (zv("rT_all") < 1.0))
        | ((zv("rP_lo") > 8.0) & (zv("rV_lo") > 8.0) & (zv("rP_night") < cfg.z_circ_night))
    )
    put("stuck_valve", stuck, 2.5, zv("rP_lo") / 100)
    curve = (
        ((a_hi | a_all) & (zv("rP_hi") > cfg.z_energy_up) & (zv("rV_hi") > cfg.z_flow_up) & ~a_lo)
        | (a_all & (zv("rP_all") > cfg.z_energy_up) & (zv("rV_all") > cfg.z_flow_up) & ~a_lo)
    )
    put("wrong_curve", curve, 2.0, zv("rP_hi") / 100)
    bypass = (
        any_hot
        & flow_up
        & energy_flat
        & ((a_lo & (a_hi | a_all)) | ((rt_all > cfg.bypass_min_rt_k) & a_all))
    )
    put("bypass", bypass, 1.5, zv("rT_all") / 100)
    fouled = a_hi & ~a_lo & energy_flat & (zv("rV_hi") > 1.0)
    put("fouled_hex", fouled, 1.0, zv("rT_hi") / 100)
    setback = (zv("rP_night") > cfg.z_night) & (zv("rP_day") < 3.0) & ~any_hot
    put("no_setback", setback, 0.8, zv("rP_night") / 100)

    best = sc.argmax(axis=0)
    best_score = sc.max(axis=0)
    code = np.where(best_score > 0, best, 0)
    return code, np.where(code > 0, best_score, 0.0)


def explain(label: str, raw: dict[str, float]) -> str:
    """Plain-language explanation for a rule label from raw (unscaled) window feature values."""
    if label == HEALTHY:
        return "No rule fired: behaviour is within this substation's own signature."
    tpl = EXPLANATIONS[label]
    vals = {
        "rT": raw.get("rT_all", float("nan")),
        "rV": float(np.expm1(raw.get("rV_all", 0.0))),
        "rP": float(np.expm1(raw.get("rP_all", 0.0))),
        "rVh": float(np.expm1(raw.get("rV_hi", 0.0))),
        "rPn": float(np.expm1(raw.get("rP_night", 0.0))),
        "rPd": float(np.expm1(raw.get("rP_day", 0.0))),
        "rPl": float(np.expm1(raw.get("rP_lo", 0.0))),
        "neg": raw.get("neg_dT", 0.0),
    }
    return tpl.format(**{k: (0.0 if v != v else v) for k, v in vals.items()})
