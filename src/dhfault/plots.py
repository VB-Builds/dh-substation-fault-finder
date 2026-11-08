"""Figures: return-temperature signature (healthy cloud vs faulty substation), ranked list, energy signature."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402

from .features import EnergySignature  # noqa: E402
from .pipeline import Analysis  # noqa: E402
from .rules import LABELS  # noqa: E402
from .sim import HEALTHY  # noqa: E402

BG = "#16213A"
GRID = "#2A3750"
TEXT = "#F8FAFC"
MUTED = "#A9B6CB"
ACCENT = "#38BDF8"
GREEN = "#34D399"
AMBER = "#FBBF24"
RED = "#F87171"
GREY = "#64748B"
VIOLET = "#A78BFA"

_NAMES = {f.name for f in font_manager.fontManager.ttflist}
FONT = "Segoe UI" if "Segoe UI" in _NAMES else "DejaVu Sans"


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": FONT,
            "text.color": TEXT,
            "axes.labelcolor": MUTED,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.edgecolor": GRID,
            "axes.facecolor": BG,
            "figure.facecolor": BG,
            "grid.color": GRID,
            "axes.grid": True,
            "grid.linewidth": 0.8,
        }
    )


def daily_series(an: Analysis, i: int) -> dict[str, np.ndarray]:
    """Daily outdoor temperature, flow-weighted return temperature (measured and expected), energy."""
    a = an.arrays
    n_days = len(a.index) // 24
    pred = an.signature.predict(a)

    def day(x):
        return x[:, : n_days * 24].reshape(x.shape[0], n_days, 24)

    v = day(a.flow_lph)[i]
    w = np.maximum(v.sum(axis=1), 1e-9)
    return {
        "t_out": a.t_out[: n_days * 24].reshape(n_days, 24).mean(axis=1),
        "tr": (day(a.t_return)[i] * v).sum(axis=1) / w,
        "tr_hat": (day(pred["tr"])[i] * v).sum(axis=1) / w,
        "energy": day(a.power_kw)[i].sum(axis=1),
    }


def _signature_axes(ax, an: Analysis, i_hi: int, healthy_idx: np.ndarray, onset_day: int, big: bool) -> None:
    ref = an.ref_days
    rng = np.random.default_rng(0)
    pts_x, pts_y = [], []
    for j in rng.choice(healthy_idx, size=min(len(healthy_idx), 60), replace=False):
        s = daily_series(an, int(j))
        pts_x.append(s["t_out"][ref:])
        pts_y.append(s["tr"][ref:])
    ax.scatter(np.concatenate(pts_x), np.concatenate(pts_y), s=6 if big else 7, c=GREY, alpha=0.45, lw=0,
               label="healthy substations (daily)", rasterized=True)
    s = daily_series(an, i_hi)
    d = np.arange(len(s["tr"]))
    before = (d >= ref) & (d < ref + onset_day)
    after = d >= ref + onset_day
    ax.scatter(s["t_out"][before], s["tr"][before], s=12, c=ACCENT, lw=0, label="flagged substation, before fault")
    ax.scatter(s["t_out"][after], s["tr"][after], s=16, c=RED, lw=0, label="flagged substation, faulty")
    order = np.argsort(s["t_out"][ref:])
    smooth = pd.Series(s["tr_hat"][ref:][order]).rolling(21, center=True, min_periods=5).median()
    ax.plot(s["t_out"][ref:][order], smooth.to_numpy(), color=GREEN, lw=2.5, alpha=0.95,
            label="its own healthy signature")
    ax.set_xlabel("daily mean outdoor temperature (deg C)")
    ax.set_ylabel("return temperature (deg C)")


def _ranked_axes(ax, labels: list[str], values: np.ndarray, is_true: np.ndarray, fs: float) -> None:
    y = np.arange(len(labels))[::-1]
    cols = [GREEN if t else AMBER for t in is_true]
    ax.barh(y, values, color=cols, height=0.68)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=fs)
    for yy, v in zip(y, values, strict=True):
        ax.text(v + values.max() * 0.02, yy, f"{v / 1000:.1f}k", va="center", color=TEXT, fontsize=fs)
    ax.set_xlim(0, values.max() * 1.25)
    ax.set_xticks([])
    ax.grid(False)
    for sp in ("top", "right", "bottom"):
        ax.spines[sp].set_visible(False)


def _pick(ex) -> tuple[int, np.ndarray, np.ndarray]:
    """Highest-cost correctly diagnosed physical fault, healthy substations, and the return-temperature list.

    The list holds substations with a return-temperature alarm, by estimated CHF/year. Metering faults
    (sensor swap or drift) are excluded here: they have no physical penalty and are in ``worklist.csv``.
    """
    pop = ex.net.pop
    al = ex.alarms["rules"]
    rate = ex.costs["rate_chf_per_year"].to_numpy()
    sensor = ex.costs["sensor_label"].to_numpy()
    order = np.argsort(-rate, kind="stable")
    faulty = (pop["fault"] != HEALTHY).to_numpy()
    listed = np.array([i for i in order if al.day[i] >= 0 and not sensor[i]])
    hi = int(next(i for i in listed if faulty[i]))
    return hi, np.flatnonzero(~faulty), listed


def _labels(ex, order: np.ndarray, k: int) -> tuple[list[str], np.ndarray, np.ndarray]:
    pop = ex.net.pop
    al = ex.alarms["rules"]
    faulty = (pop["fault"] != HEALTHY).to_numpy()
    rate = ex.costs["rate_chf_per_year"].to_numpy()
    idx = order[:k]
    labs = []
    for i in idx:
        diag = LABELS[al.label[i]] if al.day[i] >= 0 else "no alarm"
        labs.append(f"{pop['substation'].iloc[i]}  {diag.replace('_', ' ')}")
    return labs, rate[idx], faulty[idx]


def figure_main(ex, path: Path) -> None:
    """1600 x 900 result figure: signature, energy signature, ranked work list."""
    _style()
    an = ex.analysis
    hi, healthy_idx, order = _pick(ex)
    pop = ex.net.pop
    onset = int(pop["onset_day"].iloc[hi])
    fig = plt.figure(figsize=(16, 9), dpi=100, facecolor=BG)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1], height_ratios=[1, 1], left=0.06, right=0.97,
                          top=0.86, bottom=0.09, hspace=0.42, wspace=0.28)
    ax1 = fig.add_subplot(gs[:, 0])
    _signature_axes(ax1, an, hi, healthy_idx, onset, big=True)
    ax1.legend(loc="upper right", frameon=False, fontsize=12, labelcolor=TEXT, markerscale=2)
    ax1.set_title("Return temperature vs outdoor temperature", loc="left", fontsize=15, color=TEXT)
    ax2 = fig.add_subplot(gs[0, 1])
    s = daily_series(an, hi)
    ref = an.ref_days
    sigfit = EnergySignature().fit(s["t_out"][:ref], s["energy"][:ref])
    xs = np.linspace(-8, 25, 100)
    ax2.scatter(s["t_out"][:ref], s["energy"][:ref] / 1000, s=8, c=GREY, lw=0, alpha=0.6, label="history")
    ax2.scatter(s["t_out"][ref:], s["energy"][ref:] / 1000, s=10, c=RED, lw=0, label="test year")
    ax2.plot(xs, sigfit.predict(xs) / 1000, color=GREEN, lw=2, label=f"fit, balance {sigfit.t_balance:.1f} C")
    ax2.set_xlabel("daily mean outdoor temperature (deg C)")
    ax2.set_ylabel("daily energy (MWh)")
    ax2.set_title("Energy signature with change-point (flagged substation)", loc="left", fontsize=13, color=TEXT)
    ax2.legend(frameon=False, fontsize=11, labelcolor=TEXT)
    ax3 = fig.add_subplot(gs[1, 1])
    labs, vals, is_true = _labels(ex, order, 10)
    _ranked_axes(ax3, labs, vals, is_true, 11)
    ax3.set_title("Work list: return-temperature penalty, CHF/year", loc="left", fontsize=13, color=TEXT)
    fig.text(0.06, 0.945, "District-heating substation fault finder", fontsize=22, fontweight="bold", color=TEXT)
    fig.text(
        0.06,
        0.905,
        "Simulated substations, real Zurich weather (Open-Meteo, 2022-2023). Green bars = true fault, "
        "amber = false alarm. CHF figures use an illustrative return-temperature tariff.",
        fontsize=12.5,
        color=MUTED,
    )
    fig.savefig(path, dpi=100, facecolor=BG)
    plt.close(fig)


def figure_post(ex, path: Path) -> None:
    """Exactly 960 x 700 dark post image: signature panel and ranked list."""
    _style()
    an = ex.analysis
    hi, healthy_idx, order = _pick(ex)
    pop = ex.net.pop
    onset = int(pop["onset_day"].iloc[hi])
    fig = plt.figure(figsize=(9.6, 7.0), dpi=100, facecolor=BG)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1], left=0.075, right=0.975, top=0.78, bottom=0.14, wspace=0.62)
    ax1 = fig.add_subplot(gs[0, 0])
    _signature_axes(ax1, an, hi, healthy_idx, onset, big=False)
    ax1.set_xlabel("outdoor temp (deg C)", fontsize=12)
    ax1.set_ylabel("return temp (deg C)", fontsize=12)
    ax1.tick_params(labelsize=12)
    h, lab = ax1.get_legend_handles_labels()
    keep = [0, 2, 3]
    ax1.legend([h[k] for k in keep], ["healthy substations", "faulty substation", "its healthy signature"],
               loc="upper right", frameon=False, fontsize=11.5, labelcolor=TEXT, markerscale=2.5)
    ax2 = fig.add_subplot(gs[0, 1])
    k = 8
    labs, vals, is_true = _labels(ex, order, k)
    _ranked_axes(ax2, [s.split("  ")[0] + "\n" + s.split("  ")[1] for s in labs], vals, is_true, 11.5)
    ax2.set_title("Work list, CHF/year", loc="left", fontsize=13, color=TEXT)
    fig.text(0.04, 0.925, "One substation returns water too hot", fontsize=19, fontweight="bold", color=TEXT)
    fig.text(0.04, 0.875, "Simulated substations, real Zurich weather (Open-Meteo)", fontsize=13, color=MUTED)
    fig.text(0.04, 0.04, "green = true fault, amber = false alarm  |  tariff is illustrative", fontsize=12, color=MUTED)
    fig.savefig(path, dpi=100, facecolor=BG)
    plt.close(fig)


def energy_signature_png(an: Analysis, i: int, path: Path) -> None:
    """Energy-signature plot for one substation (history fit with change-point, test days highlighted)."""
    _style()
    s = daily_series(an, i)
    ref = an.ref_days
    sig = EnergySignature().fit(s["t_out"][:ref], s["energy"][:ref])
    fig, ax = plt.subplots(figsize=(8, 5), dpi=100, facecolor=BG)
    xs = np.linspace(float(s["t_out"].min()), float(s["t_out"].max()), 100)
    ax.scatter(s["t_out"][:ref], s["energy"][:ref], s=8, c=GREY, lw=0, label="fault-free history")
    ax.scatter(s["t_out"][ref:], s["energy"][ref:], s=10, c=ACCENT, lw=0, label="test period")
    ax.plot(xs, sig.predict(xs), color=GREEN, lw=2, label=f"fit (balance {sig.t_balance:.1f} C)")
    ax.set_xlabel("daily mean outdoor temperature (deg C)")
    ax.set_ylabel("daily energy (kWh)")
    ax.set_title(f"Energy signature: {an.arrays.ids[i]}", loc="left", color=TEXT)
    ax.legend(frameon=False, labelcolor=TEXT)
    fig.savefig(path, dpi=100, facecolor=BG)
    plt.close(fig)
