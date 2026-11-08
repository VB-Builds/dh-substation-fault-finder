"""Technician work list and per-substation report cards (Markdown)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .cost import TariffConfig
from .pipeline import Alarms, Analysis, estimate_costs, rank_substations
from .rules import LABELS, explain

ACTIONS = {
    "fouled_hex": "Inspect and clean or descale the heat exchanger; check primary strainer and pressure drop.",
    "bypass": "Check bypass / short-circuit valve and any thermostatic bypass setting; close it if not required.",
    "stuck_valve": "Inspect primary control valve and actuator (stuck or hunting); test full travel.",
    "wrong_curve": "Review the heating curve and parallel shift in the controller; compare with building design.",
    "dhw_circulation": "Check DHW circulation pump run time, check valves and circulation balancing.",
    "sensor_swap": "Verify supply/return sensor wiring and meter installation direction; correct metered energy.",
    "sensor_drift": "Calibrate or replace the return temperature sensor; re-estimate billed energy.",
    "no_setback": "Enable or repair the night setback schedule in the controller.",
}
ENERGY_FLOOR_NOTE = "No physical return-temperature penalty; the cost shown is energy delivered but not metered."


def build_worklist(
    an: Analysis,
    alarms: Alarms,
    tariff: TariffConfig | None = None,
    pop: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Ranked list of substations: diagnosis, alarm date, estimated CHF/year, advised action."""
    tariff = tariff or TariffConfig()
    costs = estimate_costs(an, alarms.label, alarms.day, tariff)
    order = rank_substations(costs["rate_chf_per_year"].to_numpy())
    rows = []
    t0 = an.arrays.index[0]
    for rank, i in enumerate(order, start=1):
        label = LABELS[alarms.label[i]] if alarms.day[i] >= 0 else "healthy"
        d = int(alarms.day[i])
        row = {
            "rank": rank,
            "substation": an.arrays.ids[i],
            "diagnosis": label,
            "alarm_date": (t0 + pd.Timedelta(days=d)).date().isoformat() if d >= 0 else "",
            "est_chf_per_year": round(float(costs["rate_chf_per_year"].iloc[i]), 0),
            "cost_kind": "metering" if costs["sensor_label"].iloc[i] else "return-temperature penalty",
            "action": ACTIONS.get(label, ""),
        }
        if pop is not None:
            row["true_fault_sim"] = pop["fault"].iloc[i]
        rows.append(row)
    return pd.DataFrame(rows)


def card_markdown(
    an: Analysis, alarms: Alarms, i: int, rank: int, tariff: TariffConfig, true_fault: str | None = None
) -> str:
    """Report card for substation index ``i``."""
    costs = estimate_costs(an, alarms.label, alarms.day, tariff)
    sid = an.arrays.ids[i]
    label = LABELS[alarms.label[i]] if alarms.day[i] >= 0 else "healthy"
    d = int(alarms.day[i])
    t0 = an.arrays.index[0]
    raw_now = {k: float(v[i, an.n_days - 1]) for k, v in an.raw.items()}
    lines = [
        f"# Report card: {sid}",
        "",
        f"- Work-list rank: {rank}",
        f"- Status: {'ALARM' if d >= 0 else 'no alarm'} ({label.replace('_', ' ')})",
    ]
    if d >= 0:
        lines.append(f"- First alarm: {(t0 + pd.Timedelta(days=d)).date().isoformat()}")
    lines += [
        f"- Estimated cost rate (last 30 days, annualised): {costs['rate_chf_per_year'].iloc[i]:,.0f} CHF/year "
        f"({'metering exposure' if costs['sensor_label'].iloc[i] else 'return-temperature penalty'})",
        f"- Tariff used: {tariff.chf_per_km3:.3f} CHF per K*m3 (illustrative), "
        f"energy {tariff.energy_chf_per_kwh:.2f} CHF/kWh",
        "",
        "## Why",
        "",
        explain(label, {k: raw_now[k] for k in raw_now}) if d >= 0 else "Within this substation's own signature.",
        "",
        "## Window features (last 14 days, versus own fault-free signature)",
        "",
        "| feature | value |",
        "|---|---|",
        f"| excess return temperature (flow-weighted) | {raw_now['rT_all']:+.2f} K |",
        f"| flow deviation | {np.expm1(raw_now['rV_all']):+.1%} |",
        f"| energy deviation | {np.expm1(raw_now['rP_all']):+.1%} |",
        f"| night energy deviation | {np.expm1(raw_now['rP_night']):+.1%} |",
        f"| share of hours with supply - return < 1 K | {raw_now['neg_dT']:.1%} |",
        "",
    ]
    if d >= 0:
        lines += ["## Advised action", "", ACTIONS.get(label, ""), ""]
        if costs["sensor_label"].iloc[i]:
            lines += [ENERGY_FLOOR_NOTE, ""]
    if true_fault is not None:
        lines += ["## Ground truth (simulation only)", "", f"Injected fault: {true_fault}", ""]
    return "\n".join(lines)


def write_report_cards(
    an: Analysis,
    alarms: Alarms,
    outdir: Path,
    tariff: TariffConfig | None = None,
    top: int = 10,
    pop: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Write ``worklist.csv`` and report cards for the top ``top`` substations; return the work list."""
    tariff = tariff or TariffConfig()
    outdir = Path(outdir)
    cards = outdir / "cards"
    cards.mkdir(parents=True, exist_ok=True)
    wl = build_worklist(an, alarms, tariff, pop)
    wl.to_csv(outdir / "worklist.csv", index=False)
    idx = {s: i for i, s in enumerate(an.arrays.ids)}
    for r in wl.head(top).itertuples():
        i = idx[r.substation]
        tf = pop["fault"].iloc[i] if pop is not None else None
        text = card_markdown(an, alarms, i, r.rank, tariff, tf)
        (cards / f"{r.substation}.md").write_text(text, encoding="utf-8")
    return wl
