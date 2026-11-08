"""Physics-informed features from heat-meter data.

Per substation we learn a *signature* from fault-free history: return temperature, flow and thermal
power as functions of outdoor temperature (12 h lag) and time-of-day class. New data is compared with
that signature; the residuals are aggregated over a rolling 14-day window into a few interpretable
features (K of excess return temperature, % flow and energy deviation, night versus day power, ...).
An energy-signature regression with a change-point (balance temperature) is provided as well.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import lfilter

WINDOW_DAYS = 14
N_BINS = 24
BIN_W = 2.0
BIN_LO = -12.0
MIN_CELL = 8

FEATURE_NAMES = (
    "rT_all",
    "rT_lo",
    "rT_hi",
    "rV_all",
    "rV_lo",
    "rV_hi",
    "rP_all",
    "rP_lo",
    "rP_hi",
    "rP_night",
    "rP_day",
    "neg_dT",
)
# absolute floors for the reference spread of each feature (units: K for rT, log-ratio otherwise)
SIGMA_FLOOR = {
    "rT": 0.35,
    "rV": 0.025,
    "rP": 0.02,
    "neg": 0.01,
}


@dataclass
class MeterArrays:
    """Meter data as (N, T) arrays on a shared hourly index."""

    ids: list[str]
    index: pd.DatetimeIndex
    t_out: np.ndarray  # (T,)
    t_supply: np.ndarray
    t_return: np.ndarray
    flow_lph: np.ndarray
    power_kw: np.ndarray


def to_arrays(meters: pd.DataFrame) -> MeterArrays:
    """Convert a long-format meter frame indexed by (substation, time) into ``MeterArrays``."""
    required = {"t_out", "t_supply", "t_return", "flow_lph", "power_kw"}
    missing = required - set(meters.columns)
    if missing:
        raise ValueError(f"meter frame is missing columns: {sorted(missing)}")
    if not isinstance(meters.index, pd.MultiIndex) or meters.index.nlevels != 2:
        raise ValueError("meter frame must be indexed by (substation, time)")
    m = meters.sort_index()
    ids = list(m.index.get_level_values(0).unique())
    index = pd.DatetimeIndex(m.loc[ids[0]].index)
    n, t = len(ids), len(index)
    if len(m) != n * t:
        raise ValueError("all substations must share the same hourly timestamps")

    def arr(col: str) -> np.ndarray:
        return m[col].to_numpy(dtype=float).reshape(n, t)

    return MeterArrays(
        ids=ids,
        index=index,
        t_out=arr("t_out")[0],
        t_supply=arr("t_supply"),
        t_return=arr("t_return"),
        flow_lph=arr("flow_lph"),
        power_kw=arr("power_kw"),
    )


def ema(a: np.ndarray, hours: float) -> np.ndarray:
    """Exponential moving average along the last axis (building thermal lag), started at sample 0."""
    alpha = 1.0 / hours
    a = np.asarray(a, dtype=float)
    zi = (1.0 - alpha) * a[..., :1]
    return lfilter([alpha], [1.0, -(1.0 - alpha)], a, axis=-1, zi=zi)[0]


def hour_class(index: pd.DatetimeIndex) -> np.ndarray:
    """0 night (22-05), 1 morning (05-10), 2 midday (10-17), 3 evening (17-22)."""
    h = index.hour.to_numpy()
    cls = np.zeros(len(h), dtype=int)
    cls[(h >= 5) & (h < 10)] = 1
    cls[(h >= 10) & (h < 17)] = 2
    cls[(h >= 17) & (h < 22)] = 3
    return cls


def outdoor_bin(t_out: np.ndarray) -> np.ndarray:
    t_eff = ema(np.asarray(t_out, dtype=float), 12.0)
    return np.clip(((t_eff - BIN_LO) // BIN_W).astype(int), 0, N_BINS - 1)


def _fill_nan_rows(a: np.ndarray) -> np.ndarray:
    """Fill NaN along axis 1 by nearest valid neighbour (forward then backward fill)."""
    df = pd.DataFrame(a.T)
    df = df.ffill().bfill()
    return df.to_numpy().T


class Signature:
    """Per-substation binned-median signature of return temperature, flow and power."""

    def __init__(self) -> None:
        self.tr: np.ndarray | None = None  # (N, bins, 4)
        self.lv: np.ndarray | None = None
        self.lp: np.ndarray | None = None
        self.p_max: np.ndarray | None = None

    def fit(self, arrays: MeterArrays, hours: slice) -> Signature:
        """Fit on ``arrays[:, hours]`` (fault-free history)."""
        n = len(arrays.ids)
        b = outdoor_bin(arrays.t_out)[hours]
        c = hour_class(arrays.index)[hours]
        tr = arrays.t_return[:, hours]
        lv = np.log(np.maximum(arrays.flow_lph[:, hours], 1.0))
        lp = np.log(np.maximum(arrays.power_kw[:, hours], 0.1))
        out = []
        for q in (tr, lv, lp):
            tab = np.full((n, N_BINS, 4), np.nan)
            for bi in range(N_BINS):
                for ci in range(4):
                    sel = (b == bi) & (c == ci)
                    if sel.sum() >= MIN_CELL:
                        tab[:, bi, ci] = np.median(q[:, sel], axis=1)
            for ci in range(4):
                tab[:, :, ci] = _fill_nan_rows(tab[:, :, ci])
            out.append(tab)
        self.tr, self.lv, self.lp = out
        self.p_max = np.exp(np.percentile(self.predict(arrays)["lp"][:, hours], 95, axis=1))
        return self

    def predict(self, arrays: MeterArrays) -> dict[str, np.ndarray]:
        if self.tr is None:
            raise RuntimeError("Signature.fit must be called first")
        b = outdoor_bin(arrays.t_out)
        c = hour_class(arrays.index)
        return {
            "tr": self.tr[:, b, c],
            "lv": self.lv[:, b, c],
            "lp": self.lp[:, b, c],
        }


def _day_sums(x: np.ndarray, n_days: int) -> np.ndarray:
    return x[:, : n_days * 24].reshape(x.shape[0], n_days, 24).sum(axis=2)


def _rolling(x: np.ndarray, window: int) -> np.ndarray:
    c = np.cumsum(x, axis=1)
    out = c.copy()
    out[:, window:] = c[:, window:] - c[:, :-window]
    return out


def window_features(arrays: MeterArrays, sig: Signature, window: int = WINDOW_DAYS) -> dict[str, np.ndarray]:
    """Rolling-window residual features, one value per substation and day: dict of (N, D) arrays.

    The value at day d summarises days d-window+1..d. Days before ``window`` are NaN.
    """
    t = len(arrays.index)
    n_days = t // 24
    pred = sig.predict(arrays)
    v = arrays.flow_lph
    p = np.maximum(arrays.power_kw, 0.0)
    v_hat = np.exp(pred["lv"])
    p_hat = np.exp(pred["lp"])
    r_t = arrays.t_return - pred["tr"]

    rel = p_hat / sig.p_max[:, None]
    hi = rel > 0.45
    lo = rel < 0.2
    night = np.broadcast_to(hour_class(arrays.index) == 0, v.shape)
    allm = np.ones_like(hi)

    feats: dict[str, np.ndarray] = {}

    def sums(x):
        return _rolling(_day_sums(x, n_days), window)

    for tag, mask in (("all", allm), ("lo", lo), ("hi", hi)):
        m = mask.astype(float)
        num = sums(v * r_t * m)
        den = sums(v * m)
        with np.errstate(invalid="ignore", divide="ignore"):
            feats[f"rT_{tag}"] = np.where(den > 0, num / den, np.nan)
            feats[f"rV_{tag}"] = np.log(np.maximum(sums(v * m), 1e-9) / np.maximum(sums(v_hat * m), 1e-9))
            feats[f"rP_{tag}"] = np.log(np.maximum(sums(p * m), 1e-9) / np.maximum(sums(p_hat * m), 1e-9))
        thin = sums(m) < window * 2  # fewer than ~2 h per day in that regime: not informative
        for k in ("rT", "rV", "rP"):
            if tag != "all":
                feats[f"{k}_{tag}"] = np.where(thin, np.nan, feats[f"{k}_{tag}"])
    for tag, mask in (("night", night), ("day", ~night)):
        m = mask.astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            feats[f"rP_{tag}"] = np.log(np.maximum(sums(p * m), 1e-9) / np.maximum(sums(p_hat * m), 1e-9))
    neg = (arrays.t_supply - arrays.t_return < 1.0).astype(float)
    feats["neg_dT"] = sums(neg) / (window * 24)
    for k, arr in feats.items():
        arr[:, : window - 1] = np.nan
        feats[k] = arr
    return feats


def feature_sigma(feats_ref: dict[str, np.ndarray], inflate: float = 1.3) -> dict[str, np.ndarray]:
    """Per-substation spread of each feature on fault-free history, with absolute floors, shape (N, 1)."""
    sig = {}
    for k, a in feats_ref.items():
        base = k.split("_")[0]
        floor = SIGMA_FLOOR[base]
        s = np.nanstd(a, axis=1, keepdims=True) * inflate
        sig[k] = np.maximum(np.nan_to_num(s, nan=floor), floor)
    return sig


def zscore_features(feats: dict[str, np.ndarray], sigma: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {k: feats[k] / sigma[k] for k in FEATURE_NAMES}


def stack_features(feats: dict[str, np.ndarray]) -> np.ndarray:
    """Stack feature dict into (N, D, F)."""
    return np.stack([feats[k] for k in FEATURE_NAMES], axis=2)


@dataclass
class EnergySignature:
    """Daily energy versus outdoor temperature with a change-point (balance temperature).

    E_d = base + slope * max(t_balance - t_out_d, 0). The change-point is found by grid search.
    """

    t_balance: float = 15.0
    base: float = 0.0
    slope: float = 0.0
    rmse: float = float("nan")

    def fit(self, t_out_daily: np.ndarray, energy_daily: np.ndarray) -> EnergySignature:
        t = np.asarray(t_out_daily, dtype=float)
        e = np.asarray(energy_daily, dtype=float)
        ok = np.isfinite(t) & np.isfinite(e)
        t, e = t[ok], e[ok]
        if len(t) < 20:
            raise ValueError("need at least 20 days to fit an energy signature")
        best = None
        for tb in np.arange(8.0, 21.01, 0.5):
            x = np.maximum(tb - t, 0.0)
            a = np.column_stack([np.ones_like(x), x])
            coef, *_ = np.linalg.lstsq(a, e, rcond=None)
            rss = float(np.sum((a @ coef - e) ** 2))
            if best is None or rss < best[0]:
                best = (rss, tb, coef)
        rss, tb, coef = best
        self.t_balance, self.base, self.slope = float(tb), float(coef[0]), float(coef[1])
        self.rmse = float(np.sqrt(rss / len(t)))
        return self

    def predict(self, t_out_daily: np.ndarray) -> np.ndarray:
        x = np.maximum(self.t_balance - np.asarray(t_out_daily, dtype=float), 0.0)
        return self.base + self.slope * x
