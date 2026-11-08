"""Metrics: detection and diagnosis P/R/F1 per fault type, detection delay, ranking quality."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .pipeline import Alarms
from .rules import LABEL_CODE
from .sim import FAULT_TYPES, HEALTHY


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Precision, recall, F1 (0.0 where undefined)."""
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def detection_metrics(pop: pd.DataFrame, alarms: Alarms) -> dict:
    """Substation-level 'is there a fault' detection.

    TP: faulty substation with an alarm at/after its onset. FN: faulty without one.
    FP: healthy substation with any alarm, or a faulty one whose first alarm came before its onset.
    """
    faulty = (pop["fault"] != HEALTHY).to_numpy()
    alarm = alarms.day >= 0
    pre = alarms.pre_day >= 0
    tp = int((faulty & alarm).sum())
    fn = int((faulty & ~alarm).sum())
    fp = int((~faulty & alarm).sum() + (faulty & pre).sum())
    p, r, f = prf(tp, fp, fn)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f,
            "false_alarm_substations": int((~faulty & alarm).sum()), "n_healthy": int((~faulty).sum())}


def per_type_metrics(pop: pd.DataFrame, alarms: Alarms) -> pd.DataFrame:
    """Diagnosis quality per fault type.

    For type X: TP = substations with fault X diagnosed as X, FN = fault X not diagnosed as X
    (missed or mislabelled), FP = substations without fault X diagnosed as X.
    """
    rows = []
    truth = pop["fault"].to_numpy()
    diag = np.array([LABEL_CODE_INV[c] for c in alarms.label], dtype=object)
    diag = np.where(alarms.day >= 0, diag, HEALTHY)
    for ft in FAULT_TYPES:
        is_x = truth == ft
        pred_x = diag == ft
        tp = int((is_x & pred_x).sum())
        fn = int((is_x & ~pred_x).sum())
        fp = int((~is_x & pred_x).sum())
        p, r, f = prf(tp, fp, fn)
        detected = int((is_x & (alarms.day >= 0)).sum())
        rows.append({"fault": ft, "n": int(is_x.sum()), "detected": detected, "tp": tp, "fp": fp,
                     "fn": fn, "precision": p, "recall": r, "f1": f})
    return pd.DataFrame(rows).set_index("fault")


LABEL_CODE_INV = {v: k for k, v in LABEL_CODE.items()}


def delay_days(pop: pd.DataFrame, alarms: Alarms, ref_days: int) -> pd.Series:
    """Detection delay in days (alarm day minus onset day) for detected faults, indexed by substation."""
    faulty = (pop["fault"] != HEALTHY).to_numpy() & (alarms.day >= 0)
    onset = pop["onset_day"].to_numpy() + ref_days
    d = pd.Series(alarms.day - onset, index=pop["substation"])
    return d[faulty]


def ranking_metrics(pop: pd.DataFrame, rate: np.ndarray, frac: float = 0.10) -> dict:
    """Share of true faults among the top ``frac`` substations by estimated cost rate."""
    n = len(pop)
    k = max(int(round(frac * n)), 1)
    order = np.argsort(-rate, kind="stable")[:k]
    faulty = (pop["fault"] != HEALTHY).to_numpy()
    n_f = int(faulty.sum())
    hits = int(faulty[order].sum())
    return {"k": k, "hits": hits, "share_true_faults_top": hits / k,
            "recall_of_all_faults": hits / n_f if n_f else 0.0,
            "max_possible_share": min(n_f, k) / k, "base_rate": n_f / n}
