"""Full simulated experiment: tune network -> train baselines -> evaluate on a held-out network."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .cost import TariffConfig
from .evaluate import delay_days, detection_metrics, per_type_metrics, prf, ranking_metrics
from .pipeline import (
    Alarms,
    Analysis,
    AnomalyModel,
    SupervisedBaseline,
    analyse,
    estimate_costs,
    extract_alarms,
    rule_codes,
)
from .rules import LABELS
from .sim import HEALTHY, PHYSICAL_FAULTS, Network, SimConfig, simulate_network

TUNE_SEED = 11
EVAL_SEED = 21


@dataclass
class Experiment:
    net: Network
    analysis: Analysis
    alarms: dict[str, Alarms]  # method -> alarms on the evaluation network
    codes: dict[str, np.ndarray]
    costs: pd.DataFrame  # per substation, rule-based diagnosis
    results: dict


def _onset_abs(net: Network, ref_days: int) -> np.ndarray:
    o = net.pop["onset_day"].to_numpy()
    return np.where(o >= 0, o + ref_days, -1)


def run_experiment(
    n_tune: int = 240,
    per_type_tune: int = 5,
    n_eval: int = 300,
    per_type_eval: int = 6,
    tariff: TariffConfig | None = None,
    seed_tune: int = TUNE_SEED,
    seed_eval: int = EVAL_SEED,
) -> Experiment:
    """Simulate a tuning network and a held-out evaluation network and score every method.

    The rule thresholds were set by hand on the tuning network (seed 11). The isolation forest is fit on
    fault-free history of the evaluation network itself (unsupervised); the supervised baseline is
    trained on the tuning network only. Everything reported comes from the evaluation network.
    """
    tariff = tariff or TariffConfig()
    net_t = simulate_network(SimConfig(n_substations=n_tune, faults_per_type=per_type_tune, seed=seed_tune))
    an_t = analyse(net_t.meters, net_t.ref_days)
    sup = SupervisedBaseline().fit(an_t, net_t.pop)
    del net_t, an_t

    net = simulate_network(SimConfig(n_substations=n_eval, faults_per_type=per_type_eval, seed=seed_eval))
    an = analyse(net.meters, net.ref_days)
    onset = _onset_abs(net, net.ref_days)
    iso = AnomalyModel().fit(an)

    codes = {
        "rules": rule_codes(an),
        "supervised": sup.codes(an),
        "anomaly": iso.codes(an),
    }
    alarms = {k: extract_alarms(v, net.ref_days, onset) for k, v in codes.items()}
    results = score_experiment(net, an, alarms, tariff)
    costs = estimate_costs(an, alarms["rules"].label, alarms["rules"].day, tariff)
    return Experiment(net, an, alarms, codes, costs, results)


def score_experiment(net: Network, an: Analysis, alarms: dict[str, Alarms], tariff: TariffConfig) -> dict:
    pop = net.pop
    ref = net.ref_days
    res: dict = {
        "network": {
            "substations": int(len(pop)),
            "faulty": int((pop["fault"] != HEALTHY).sum()),
            "fault_types": int(pop["fault"].nunique() - 1),
            "test_days": int(an.n_days - ref),
            "weather": "Open-Meteo archive, Zurich, hourly 2 m temperature, 2022 (history) + 2023 (test)",
        },
        "tariff": {"chf_per_km3": tariff.chf_per_km3, "energy_chf_per_kwh": tariff.energy_chf_per_kwh},
    }
    res["detection"] = {k: detection_metrics(pop, a) for k, a in alarms.items()}
    per_type = {k: per_type_metrics(pop, alarms[k]) for k in ("rules", "supervised")}
    res["per_type"] = {k: v.round(4).reset_index().to_dict(orient="records") for k, v in per_type.items()}
    res["macro_f1_diagnosis"] = {k: float(v["f1"].mean()) for k, v in per_type.items()}
    # overall diagnosis accuracy among detected faults
    for k in ("rules", "supervised"):
        al = alarms[k]
        faulty = (pop["fault"] != HEALTHY).to_numpy() & (al.day >= 0)
        diag = np.array([LABELS[c] for c in al.label], dtype=object)
        res.setdefault("diagnosis_accuracy_of_detected", {})[k] = float(
            (diag[faulty] == pop["fault"].to_numpy()[faulty]).mean() if faulty.any() else 0.0
        )
    d = delay_days(pop, alarms["rules"], ref)
    by_type = d.groupby(pop.set_index("substation")["fault"]).median()
    res["delay_days"] = {
        "median": float(d.median()) if len(d) else float("nan"),
        "p90": float(d.quantile(0.9)) if len(d) else float("nan"),
        "median_by_type": {k: float(v) for k, v in by_type.items()},
    }
    da = delay_days(pop, alarms["anomaly"], ref)
    res["delay_days_anomaly_model"] = {"median": float(da.median()) if len(da) else float("nan")}

    costs = estimate_costs(an, alarms["rules"].label, alarms["rules"].day, tariff)
    rate = costs["rate_chf_per_year"].to_numpy()
    res["ranking"] = ranking_metrics(pop, rate, 0.10)
    res["ranking_top20pct"] = ranking_metrics(pop, rate, 0.20)
    # CHF recoverable: cost between alarm and end of year, physical faults only, vs ground truth
    phys = pop["fault"].isin(PHYSICAL_FAULTS).to_numpy()
    al = alarms["rules"]
    det_phys = phys & (al.day >= 0) & ~costs["sensor_label"].to_numpy()
    true_after = np.array(
        [
            tariff.chf_per_km3 * net.km3_daily[i, max(al.day[i] - ref, 0) :].sum() if al.day[i] >= 0 else 0.0
            for i in range(len(pop))
        ]
    )
    true_year = tariff.chf_per_km3 * net.truth["true_excess_km3_year"].to_numpy()
    est_after = costs["cost_after_alarm_chf"].to_numpy()
    res["chf"] = {
        "true_penalty_all_physical_faults_if_never_fixed": float(true_year[phys].sum()),
        "true_recoverable_after_alarm_detected_physical": float(true_after[det_phys].sum()),
        "estimated_recoverable_after_alarm_detected_physical": float(est_after[det_phys].sum()),
        "missed_or_late_physical": float(true_year[phys].sum() - true_after[det_phys].sum()),
        "relative_error_estimate": float(
            (est_after[det_phys].sum() - true_after[det_phys].sum()) / max(true_after[det_phys].sum(), 1e-9)
        ),
        "estimated_metering_exposure_sensor_faults": float(est_after[costs["sensor_label"].to_numpy()].sum()),
    }
    if det_phys.sum() > 2:
        from scipy.stats import spearmanr

        res["chf"]["spearman_est_vs_true_rate"] = float(
            spearmanr(est_after[det_phys], true_after[det_phys]).statistic
        )
    p, r, f = prf(
        res["detection"]["rules"]["tp"], res["detection"]["rules"]["fp"], res["detection"]["rules"]["fn"]
    )
    res["headline"] = {"detection_precision": p, "detection_recall": r, "detection_f1": f}
    return res
