"""End-to-end analysis: signature, features, rules, anomaly score, supervised baseline, cost, ranking."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier

from .cost import TariffConfig
from .features import (
    FEATURE_NAMES,
    MeterArrays,
    Signature,
    feature_sigma,
    stack_features,
    to_arrays,
    window_features,
    zscore_features,
)
from .rules import LABEL_CODE, LABELS, RuleConfig, apply_rules
from .sim import HEALTHY, PHYSICAL_FAULTS, Network

PERSIST_DAYS = 5  # a label must appear on this many of the last 7 days to raise an alarm
PERSIST_WINDOW = 7
DIAGNOSIS_DAYS = 30


@dataclass
class Analysis:
    """Everything derived from one meter data set."""

    arrays: MeterArrays
    ref_days: int
    signature: Signature
    raw: dict[str, np.ndarray]  # (N, D) rolling-window features
    sigma: dict[str, np.ndarray]
    z: dict[str, np.ndarray]
    excess_km3_daily: np.ndarray  # (N, D) sum of flow(m3) x (return - signature) per day, signed
    missing_kwh_daily: np.ndarray  # (N, D) sum of (signature power - metered power) per day, signed
    flow_m3_daily: np.ndarray  # (N, D) metered flow volume per day
    expected_kwh_daily: np.ndarray  # (N, D) energy per day expected from the signature
    annual_m3: np.ndarray  # (N,) yearly flow volume in the fault-free history
    annual_kwh: np.ndarray  # (N,) yearly energy in the fault-free history

    @property
    def n_days(self) -> int:
        return self.raw["rT_all"].shape[1]


def analyse(meters: pd.DataFrame, ref_days: int = 365) -> Analysis:
    """Learn signatures on the first ``ref_days`` days (fault-free) and compute residual features."""
    arrays = to_arrays(meters)
    if ref_days < 60 or ref_days * 24 >= len(arrays.index):
        raise ValueError("ref_days must be >= 60 and shorter than the data")
    sig = Signature().fit(arrays, slice(0, ref_days * 24))
    raw = window_features(arrays, sig)
    sigma = feature_sigma({k: v[:, :ref_days] for k, v in raw.items()})
    z = zscore_features(raw, sigma)
    pred = sig.predict(arrays)
    n_days = len(arrays.index) // 24
    v_m3 = arrays.flow_lph / 1000.0
    r_t = arrays.t_return - pred["tr"]
    exc = (v_m3 * r_t).reshape(len(arrays.ids), n_days, 24).sum(axis=2)
    miss = (np.exp(pred["lp"]) - np.maximum(arrays.power_kw, 0.0)).reshape(
        len(arrays.ids), n_days, 24
    ).sum(axis=2)
    n = len(arrays.ids)
    flow_m3_daily = v_m3.reshape(n, n_days, 24).sum(axis=2)
    expected = np.exp(pred["lp"]).reshape(n, n_days, 24).sum(axis=2)
    metered = np.maximum(arrays.power_kw, 0.0).reshape(n, n_days, 24).sum(axis=2)
    scale = 365.0 / ref_days
    return Analysis(
        arrays,
        ref_days,
        sig,
        raw,
        sigma,
        z,
        exc,
        miss,
        flow_m3_daily,
        expected,
        flow_m3_daily[:, :ref_days].sum(axis=1) * scale,
        metered[:, :ref_days].sum(axis=1) * scale,
    )


# ----------------------------------------------------------------------------- alarms


def persistent_labels(codes: np.ndarray, need: int = PERSIST_DAYS, window: int = PERSIST_WINDOW) -> np.ndarray:
    """Daily label that has appeared on >= ``need`` of the last ``window`` days (0 if none). Shape (N, D)."""
    n, d = codes.shape
    out = np.zeros_like(codes)
    n_lab = len(LABELS)
    counts = np.zeros((n_lab, n, d), dtype=int)
    for k in range(1, n_lab):
        hit = (codes == k).astype(int)
        c = np.cumsum(hit, axis=1)
        w = c.copy()
        w[:, window:] = c[:, window:] - c[:, :-window]
        counts[k] = w
    best = counts.argmax(axis=0)
    ok = counts.max(axis=0) >= need
    out[ok] = best[ok]
    return out


@dataclass
class Alarms:
    """First alarm per substation (index into the day axis) and the diagnosis after one month."""

    day: np.ndarray  # (N,) first persistent alarm day at or after ``start_day``; -1 if none
    pre_day: np.ndarray  # (N,) first persistent alarm day before ``onset`` (premature); -1 if none
    label: np.ndarray  # (N,) diagnosed label code (mode over DIAGNOSIS_DAYS after the alarm)


def extract_alarms(
    codes: np.ndarray,
    start_day: int,
    onset_day: np.ndarray | None = None,
) -> Alarms:
    """First alarm per substation from daily label codes.

    ``start_day`` is the first day of the test period. Alarms are only searched from there. If
    ``onset_day`` (absolute day index, -1 for healthy) is given, an alarm raised before the onset of a
    fault is reported as ``pre_day`` (a false alarm) and the search continues after the onset.
    """
    n, _ = codes.shape
    per = persistent_labels(codes)
    alarm = np.full(n, -1)
    pre = np.full(n, -1)
    label = np.zeros(n, dtype=int)
    for i in range(n):
        hit = np.flatnonzero(per[i, start_day:] > 0)
        if len(hit) == 0:
            continue
        hit = hit + start_day
        if onset_day is not None and onset_day[i] >= 0:
            early = hit[hit < onset_day[i]]
            if len(early):
                pre[i] = early[0]
            hit = hit[hit >= onset_day[i]]
            if len(hit) == 0:
                continue
        alarm[i] = hit[0]
        seg = per[i, hit[0] : hit[0] + DIAGNOSIS_DAYS]
        seg = seg[seg > 0]
        label[i] = np.bincount(seg).argmax() if len(seg) else 0
    return Alarms(alarm, pre, label)


# ----------------------------------------------------------------------------- models


def rule_codes(an: Analysis, cfg: RuleConfig | None = None) -> np.ndarray:
    code, _ = apply_rules(an.raw, an.z, cfg)
    return code


def feature_matrix(an: Analysis) -> np.ndarray:
    """(N, D, F) matrix of z-scored features plus raw excess return temperature, NaN -> 0."""
    z = stack_features(an.z)
    extra = np.stack([an.raw["rT_all"], an.raw["neg_dT"]], axis=2)
    x = np.concatenate([z, extra], axis=2)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(x, -200.0, 200.0)


class AnomalyModel:
    """Unsupervised anomaly score: isolation forest trained on fault-free history only."""

    def __init__(self, seed: int = 0, quantile: float = 0.999) -> None:
        self.seed = seed
        self.quantile = quantile
        self.model: IsolationForest | None = None
        self.threshold = float("inf")

    def fit(self, an: Analysis) -> AnomalyModel:
        x = feature_matrix(an)[:, 14 : an.ref_days, :].reshape(-1, len(FEATURE_NAMES) + 2)
        rng = np.random.default_rng(self.seed)
        if len(x) > 120_000:
            x = x[rng.choice(len(x), 120_000, replace=False)]
        self.model = IsolationForest(n_estimators=150, random_state=self.seed, n_jobs=-1).fit(x)
        self.threshold = float(np.quantile(-self.model.score_samples(x), self.quantile))
        return self

    def score(self, an: Analysis) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("fit first")
        x = feature_matrix(an)
        n, d, f = x.shape
        return (-self.model.score_samples(x.reshape(-1, f))).reshape(n, d)

    def codes(self, an: Analysis) -> np.ndarray:
        """1 where anomalous (any fault; the type is unknown to this model)."""
        return (self.score(an) > self.threshold).astype(int)


def true_state(pop: pd.DataFrame, ref_days: int, n_total_days: int, settle: int = 14) -> np.ndarray:
    """Per substation-day label codes for supervised training; -1 = ambiguous (window straddles onset)."""
    n = len(pop)
    state = np.zeros((n, n_total_days), dtype=int)
    for i, (ft, on) in enumerate(zip(pop["fault"], pop["onset_day"], strict=True)):
        if ft == HEALTHY:
            continue
        a = ref_days + int(on)
        state[i, a : a + settle - 1] = -1
        state[i, a + settle - 1 :] = LABEL_CODE[ft]
    return state


class SupervisedBaseline:
    """Random forest on the same window features, trained on a *different* simulated network."""

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self.model = RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=4,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        )

    def fit(self, an: Analysis, pop: pd.DataFrame) -> SupervisedBaseline:
        x = feature_matrix(an)
        y = true_state(pop, an.ref_days, x.shape[1])
        keep = np.zeros_like(y, dtype=bool)
        keep[:, 14:] = True
        keep &= y >= 0
        xs, ys = x[keep], y[keep]
        rng = np.random.default_rng(self.seed)
        # healthy rows dominate; cap them so fault classes are visible
        healthy = np.flatnonzero(ys == 0)
        faulty = np.flatnonzero(ys > 0)
        if len(healthy) > 4 * max(len(faulty), 1):
            healthy = rng.choice(healthy, 4 * max(len(faulty), 1), replace=False)
        idx = np.concatenate([healthy, faulty])
        self.model.fit(xs[idx], ys[idx])
        return self

    def codes(self, an: Analysis) -> np.ndarray:
        x = feature_matrix(an)
        n, d, f = x.shape
        return self.model.predict(x.reshape(-1, f)).reshape(n, d)


# ----------------------------------------------------------------------------- cost and ranking


def estimate_costs(
    an: Analysis,
    labels: np.ndarray,
    alarm_day: np.ndarray,
    tariff: TariffConfig | None = None,
    recent_days: int = 30,
) -> pd.DataFrame:
    """Cost per substation: accumulated since the alarm, and the current annual rate.

    Physical labels: return-temperature penalty = flow x excess return temperature x tariff. The annual
    rate assumes the excess seen over the last ``recent_days`` (flow-weighted, K) persists for a whole
    year of the substation's normal flow volume (taken from its fault-free history).
    Sensor labels (swap, drift): the return-temperature excess is a measurement artefact, so the
    exposure is the share of expected heat that was not metered, applied to the yearly energy, x price.
    """
    tariff = tariff or TariffConfig()
    n = an.excess_km3_daily.shape[0]
    sensor = np.isin(labels, [LABEL_CODE["sensor_swap"], LABEL_CODE["sensor_drift"]])
    after = np.zeros(n)
    rate = np.zeros(n)
    k = recent_days
    for i in range(n):
        a = alarm_day[i]
        unit = tariff.energy_chf_per_kwh * an.missing_kwh_daily[i] if sensor[i] else (
            tariff.chf_per_km3 * an.excess_km3_daily[i]
        )
        if a >= 0:
            after[i] = max(unit[a:].sum(), 0.0)
        if sensor[i]:
            frac = an.missing_kwh_daily[i, -k:].sum() / max(an.expected_kwh_daily[i, -k:].sum(), 1e-9)
            rate[i] = max(frac, 0.0) * an.annual_kwh[i] * tariff.energy_chf_per_kwh
        else:
            r_bar = an.excess_km3_daily[i, -k:].sum() / max(an.flow_m3_daily[i, -k:].sum(), 1e-9)
            rate[i] = max(r_bar, 0.0) * an.annual_m3[i] * tariff.chf_per_km3
    return pd.DataFrame({"cost_after_alarm_chf": after, "rate_chf_per_year": rate, "sensor_label": sensor})


def rank_substations(rate_chf_per_year: np.ndarray) -> np.ndarray:
    """Indices sorted by descending cost rate."""
    return np.argsort(-rate_chf_per_year, kind="stable")


def pipeline_summary(net: Network, an: Analysis, codes: np.ndarray) -> dict:
    """Convenience wrapper: alarms for rule codes on a simulated network."""
    onset = np.where(net.pop["onset_day"].to_numpy() >= 0, net.pop["onset_day"].to_numpy() + an.ref_days, -1)
    alarms = extract_alarms(codes, an.ref_days, onset)
    return {"alarms": alarms, "onset": onset, "physical": net.pop["fault"].isin(PHYSICAL_FAULTS).to_numpy()}
