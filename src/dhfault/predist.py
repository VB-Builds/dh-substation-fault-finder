"""Evaluation on the public labelled PreDist benchmark of district-heating substations.

Dataset: Roelofs et al., "PreDist dataset - Operational data of district heating substations labelled
with faults and maintenance information", Zenodo, doi:10.5281/zenodo.17522255, licence CC BY 4.0
(use and redistribution of derived results allowed with attribution). Paper: arXiv:2511.14791.
The data is NOT stored in this repository; ``fetch`` downloads it from Zenodo and verifies the checksum.

Protocol (our own, simpler than the paper's, so numbers are NOT comparable to the paper's):

* Only heat-meter signals are used: primary supply/return temperature, flow, thermal power, plus
  outdoor temperature. Hourly means.
* For each labelled fault report (``efd_possible``) the model is trained on the dataset's recommended
  training window and tested on the window from its end up to the customer's report date.
* For each predefined normal event it is trained on the recommended window and tested on the event.
* An event "alarms" when a day with a high share of anomalous hours is seen. The threshold is set
  from the last 30 percent of the training window only (never from test data).
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

ZENODO_API = "https://zenodo.org/api/records/17522255"
ZIP_NAME = "predist_dataset.zip"
METER_COLS = {
    "outdoor_temperature": "t_out",
    "p_net_supply_temperature": "t_supply",
    "p_net_return_temperature": "t_return",
    "p_net_meter_flow": "flow",
    "p_net_meter_heat_power": "power",
}
# unit scale to (l/h, kW) per manufacturer
SCALE = {1: {"flow": 1.0, "power": 1.0}, 2: {"flow": 1e-3, "power": 1e-3}}
MIN_COVERAGE = 0.6


def fetch(dest: Path) -> Path:
    """Download and extract the dataset into ``dest`` (checksum verified). Returns the dataset root."""
    dest = Path(dest)
    root = dest / "predist"
    if (root / "manufacturer 1" / "faults.csv").exists():
        return root
    dest.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(ZENODO_API, timeout=60) as r:  # noqa: S310 (fixed https URL)
        rec = json.load(r)
    entry = next(f for f in rec["files"] if f["key"] == ZIP_NAME)
    algo, _, want = entry["checksum"].partition(":")
    zpath = dest / ZIP_NAME
    if not zpath.exists():
        urllib.request.urlretrieve(entry["links"]["self"], zpath)  # noqa: S310
    h = hashlib.new(algo)
    with open(zpath, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    if h.hexdigest() != want:
        raise RuntimeError(f"checksum mismatch for {zpath}: {h.hexdigest()} != {want}")
    with zipfile.ZipFile(zpath) as zf:
        zf.extractall(root)
    return root


def _snake(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def load_events(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fault reports and normal events of both manufacturers, with a ``manufacturer`` column."""
    faults, normals = [], []
    for m in (1, 2):
        d = Path(root) / f"manufacturer {m}"
        f = pd.read_csv(d / "faults.csv", sep=";").rename(columns=_snake)
        f["manufacturer"] = m
        for c in ("report_date", "training_start", "training_end"):
            f[c] = pd.to_datetime(f[c], errors="coerce")
        n = pd.read_csv(d / "normal_events.csv", sep=";").rename(columns=_snake)
        n["manufacturer"] = m
        for c in ("event_start", "event_end", "training_start", "training_end"):
            n[c] = pd.to_datetime(n[c], errors="coerce")
        faults.append(f)
        normals.append(n)
    return pd.concat(faults, ignore_index=True), pd.concat(normals, ignore_index=True)


def load_hourly(root: Path, manufacturer: int, substation: int) -> pd.DataFrame:
    """Hourly meter frame (t_out, t_supply, t_return, flow l/h, power kW) of one substation."""
    path = Path(root) / f"manufacturer {manufacturer}" / "operational_data" / f"substation_{substation}.csv"
    head = pd.read_csv(path, sep=";", nrows=0).columns
    use = ["timestamp"] + [c for c in METER_COLS if c in head]
    df = pd.read_csv(path, sep=";", usecols=use, parse_dates=["timestamp"])
    df = df.rename(columns=METER_COLS).set_index("timestamp").sort_index()
    for c in METER_COLS.values():
        if c not in df.columns:
            df[c] = np.nan
    df = df[~df.index.duplicated()]
    hourly = df.resample("1h").mean()
    hourly["flow"] *= SCALE[manufacturer]["flow"]
    hourly["power"] *= SCALE[manufacturer]["power"]
    return hourly


def _model_frame(h: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=h.index)
    out["t_out"] = h["t_out"]
    out["t_return"] = h["t_return"]
    out["d_t"] = h["t_supply"] - h["t_return"]
    ok = (h["flow"] > 1.0) & (h["power"] > 0.1)
    out["log_flow"] = np.where(ok, np.log(h["flow"].clip(lower=1.0)), np.nan)
    out["log_power"] = np.where(ok, np.log(h["power"].clip(lower=0.1)), np.nan)
    out["night"] = ((h.index.hour >= 22) | (h.index.hour < 5)).astype(int)
    return out.dropna()


class BinnedZ:
    """Physics signature: per-signal median and MAD in outdoor-temperature bins x day/night."""

    SIGNALS = ("t_return", "d_t", "log_flow", "log_power")

    def __init__(self, width: float = 3.0) -> None:
        self.width = width
        self.b0 = 0
        self.med: dict[str, np.ndarray] = {}  # signal -> (2, bins)
        self.mad: dict[str, np.ndarray] = {}

    def _bin(self, t: np.ndarray) -> np.ndarray:
        return np.floor(np.asarray(t, dtype=float) / self.width).astype(int)

    def fit(self, f: pd.DataFrame) -> BinnedZ:
        b = self._bin(f["t_out"].to_numpy())
        night = f["night"].to_numpy()
        self.b0 = int(b.min())
        nb = int(b.max()) - self.b0 + 1
        for s in self.SIGNALS:
            v = f[s].to_numpy()
            med = np.full((2, nb), np.nan)
            mad = np.full((2, nb), np.nan)
            for ni in (0, 1):
                for bi in range(nb):
                    sel = (b - self.b0 == bi) & (night == ni)
                    if sel.sum() >= 6:
                        m = float(np.median(v[sel]))
                        med[ni, bi] = m
                        mad[ni, bi] = float(np.median(np.abs(v[sel] - m))) * 1.4826
            self.med[s], self.mad[s] = med, mad
        return self

    def _lookup(self, table: np.ndarray, b: np.ndarray, night: np.ndarray) -> np.ndarray:
        """Value of the nearest populated bin of the same day/night class (other class as fallback)."""
        out = np.full(len(b), np.nan)
        rel = b - self.b0
        for ni in (0, 1):
            sel = night == ni
            if not sel.any():
                continue
            valid = np.flatnonzero(~np.isnan(table[ni]))
            if len(valid) == 0:
                valid_other = np.flatnonzero(~np.isnan(table[1 - ni]))
                if len(valid_other) == 0:
                    continue
                idx = valid_other[np.abs(valid_other[None, :] - rel[sel, None]).argmin(axis=1)]
                out[sel] = table[1 - ni][idx]
            else:
                idx = valid[np.abs(valid[None, :] - rel[sel, None]).argmin(axis=1)]
                out[sel] = table[ni][idx]
        return out

    def z(self, f: pd.DataFrame) -> pd.DataFrame:
        b = self._bin(f["t_out"].to_numpy())
        night = f["night"].to_numpy()
        out = {}
        for s in self.SIGNALS:
            med = self._lookup(self.med[s], b, night)
            mad = self._lookup(self.mad[s], b, night)
            floor = 0.25 * np.nanmedian(self.mad[s]) if np.isfinite(self.mad[s]).any() else 1.0
            out[s] = (f[s].to_numpy() - med) / np.maximum(np.nan_to_num(mad, nan=1.0), max(floor, 1e-3))
        return pd.DataFrame(out, index=f.index)


@dataclass
class EventResult:
    kind: str  # "fault" or "normal"
    event_id: int
    manufacturer: int
    substation: int
    method: str
    evaluable: bool
    alarm: bool = False
    lead_days: float = float("nan")
    reason: str = ""
    label: str = ""
    monitoring_potential: float = float("nan")


def _daily_share(flags: pd.Series) -> pd.Series:
    return flags.groupby(flags.index.floor("D")).mean()


def _detect(
    method: str,
    hourly: pd.DataFrame,
    train: tuple[pd.Timestamp, pd.Timestamp],
    test: tuple[pd.Timestamp, pd.Timestamp],
) -> tuple[bool, pd.Timestamp | None, str]:
    """Train on ``train``, calibrate on its last 30 percent, return (alarm, first alarm time, reason)."""
    tr = _model_frame(hourly.loc[train[0] : train[1]])
    te = _model_frame(hourly.loc[test[0] : test[1]])
    if len(tr) < 24 * 10:
        return False, None, "training data < 10 days"
    if len(te) < 24 * 2:
        return False, None, "test data < 2 days"
    cut = int(len(tr) * 0.7)
    fit_part, cal_part = tr.iloc[:cut], tr.iloc[cut:]
    if method == "signature":
        model = BinnedZ().fit(fit_part)

        def score(f):
            return model.z(f).abs().max(axis=1)

        thr_hour = 4.5
    elif method == "isolation_forest":
        cols = ["t_out", "t_return", "d_t", "log_flow", "log_power"]
        iso = IsolationForest(n_estimators=150, random_state=0, n_jobs=1).fit(fit_part[cols])

        def score(f):
            return pd.Series(-iso.score_samples(f[cols]), index=f.index)

        thr_hour = float(np.quantile(-iso.score_samples(cal_part[cols]), 0.99))
    else:
        raise ValueError(f"unknown method {method}")
    cal_share = _daily_share(score(cal_part) > thr_hour)
    day_thr = max(0.3, float(cal_share.max()) * 1.25 if len(cal_share) else 0.3)
    share = _daily_share(score(te) > thr_hour)
    hit = share[share > day_thr]
    if hit.empty:
        return False, None, ""
    return True, hit.index[0], ""


def evaluate_events(
    root: Path,
    methods: tuple[str, ...] = ("signature", "isolation_forest"),
    max_events: int | None = None,
) -> pd.DataFrame:
    """Run every evaluable fault report and normal event; one row per (event, method)."""
    faults, normals = load_events(root)
    faults = faults[faults["efd_possible"].astype(str).str.lower() == "true"].copy()
    jobs = []
    for r in faults.itertuples():
        if pd.isna(r.training_start) or pd.isna(r.training_end):
            continue
        jobs.append(("fault", r))
    for r in normals.itertuples():
        jobs.append(("normal", r))
    if max_events:
        jobs = jobs[:max_events]
    cache: dict[tuple[int, int], pd.DataFrame] = {}
    rows: list[EventResult] = []
    for kind, r in jobs:
        m = int(r.manufacturer)
        sid = int(r.substation_id)
        key = (m, sid)
        path = Path(root) / f"manufacturer {m}" / "operational_data" / f"substation_{sid}.csv"
        if key not in cache:
            cache[key] = load_hourly(root, m, sid) if path.exists() else pd.DataFrame()
        hourly = cache[key]
        if kind == "fault":
            train = (r.training_start, r.training_end)
            test = (r.training_end, r.report_date)  # training end -> customer report
            ev_id, label = int(r.event_id), str(r.fault_label)
            mp = pd.to_numeric(r.monitoring_potential, errors="coerce")
        else:
            train = (r.training_start, r.training_end)
            test = (r.event_start, r.event_end)
            ev_id, label, mp = int(r.event_id), "", float("nan")
        for method in methods:
            res = EventResult(kind, ev_id, m, sid, method, False, label=label, monitoring_potential=float(mp))
            if hourly.empty or pd.isna(train[0]) or pd.isna(train[1]) or pd.isna(test[0]) or pd.isna(test[1]):
                res.reason = "missing data or window"
                rows.append(res)
                continue
            cov = hourly.loc[test[0] : test[1], ["t_return", "flow", "power"]].notna().all(axis=1).mean()
            cov_tr = hourly.loc[train[0] : train[1], ["t_return", "flow", "power"]].notna().all(axis=1).mean()
            if not (cov >= MIN_COVERAGE and cov_tr >= MIN_COVERAGE):
                res.reason = "meter data coverage below 60 percent"
                rows.append(res)
                continue
            alarm, t0, why = _detect(method, hourly, train, test)
            if why:
                res.reason = why
                rows.append(res)
                continue
            res.evaluable = True
            res.alarm = alarm
            if alarm and kind == "fault":
                res.lead_days = float((test[1] - t0).total_seconds() / 86400.0)
            rows.append(res)
    return pd.DataFrame([r.__dict__ for r in rows])


def summarise(results: pd.DataFrame, beta: float = 0.5) -> pd.DataFrame:
    """Per method: fault recall, normal-event false-alarm rate, eventwise F-beta, mean lead time."""
    out = []
    for method, g in results[results["evaluable"]].groupby("method"):
        f = g[g["kind"] == "fault"]
        n = g[g["kind"] == "normal"]
        tp = int(f["alarm"].sum())
        fn = int(len(f) - tp)
        fp = int(n["alarm"].sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        fb = (1 + beta**2) * prec * rec / (beta**2 * prec + rec) if prec + rec else 0.0
        lead = f.loc[f["alarm"], "lead_days"]
        out.append(
            {
                "method": method,
                "fault_events": len(f),
                "detected_before_report": tp,
                "recall": rec,
                "normal_events": len(n),
                "normal_with_alarm": fp,
                "precision": prec,
                f"F{beta}": fb,
                "mean_lead_days": float(lead.mean()) if len(lead) else float("nan"),
            }
        )
    return pd.DataFrame(out).set_index("method") if out else pd.DataFrame()
