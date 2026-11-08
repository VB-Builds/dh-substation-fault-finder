import numpy as np
import pandas as pd
import pytest

from dhfault.cost import TariffConfig, excess_cost_chf, metering_exposure_chf, retail_value_per_km3
from dhfault.evaluate import detection_metrics, prf, ranking_metrics
from dhfault.features import (
    EnergySignature,
    Signature,
    ema,
    feature_sigma,
    hour_class,
    to_arrays,
    window_features,
    zscore_features,
)
from dhfault.pipeline import Alarms, extract_alarms, persistent_labels
from dhfault.rules import LABEL_CODE, LABELS, RuleConfig, apply_rules, explain


# ---------------------------------------------------------------- features
def test_to_arrays_shapes(small_net):
    a = to_arrays(small_net.meters)
    assert a.t_return.shape == (24, 2 * 365 * 24)
    assert a.t_out.shape == (2 * 365 * 24,)
    assert a.ids[0] == "S000"


def test_to_arrays_rejects_missing_columns(small_net):
    with pytest.raises(ValueError, match="missing columns"):
        to_arrays(small_net.meters.drop(columns=["flow_lph"]))


def test_to_arrays_rejects_flat_index(small_net):
    with pytest.raises(ValueError, match="indexed by"):
        to_arrays(small_net.meters.reset_index())


def test_to_arrays_rejects_unequal_timestamps(small_net):
    m = small_net.meters.iloc[:-5]
    with pytest.raises(ValueError, match="share the same"):
        to_arrays(m)


def test_ema_constant_and_step():
    assert ema(np.full(50, 7.0), 12.0) == pytest.approx(np.full(50, 7.0))
    step = np.r_[np.zeros(10), np.ones(200)]
    out = ema(step, 12.0)
    assert out[10] == pytest.approx(1 / 12)
    assert out[-1] == pytest.approx(1.0, abs=1e-6)


def test_hour_classes():
    idx = pd.date_range("2023-01-02 00:00", periods=24, freq="h")
    cls = hour_class(idx)
    assert cls[0] == 0 and cls[23] == 0 and cls[6] == 1 and cls[12] == 2 and cls[19] == 3


def test_energy_signature_recovers_parameters():
    rng = np.random.default_rng(1)
    t = rng.uniform(-8, 25, 400)
    e = 120 + 35 * np.maximum(15.5 - t, 0) + rng.normal(0, 5, 400)
    sig = EnergySignature().fit(t, e)
    assert sig.t_balance == pytest.approx(15.5, abs=1.0)
    assert sig.slope == pytest.approx(35, rel=0.05)
    assert sig.base == pytest.approx(120, rel=0.1)
    assert sig.predict(np.array([25.0]))[0] == pytest.approx(sig.base)
    assert sig.rmse < 8


def test_energy_signature_needs_data():
    with pytest.raises(ValueError):
        EnergySignature().fit(np.arange(5.0), np.arange(5.0))


def test_signature_predict_requires_fit(small_net):
    with pytest.raises(RuntimeError):
        Signature().predict(to_arrays(small_net.meters))


def test_healthy_residuals_are_small_after_fit(small_net):
    a = to_arrays(small_net.meters)
    sig = Signature().fit(a, slice(0, 365 * 24))
    f = window_features(a, sig)
    healthy = (small_net.pop["fault"] == "healthy").to_numpy()
    r_t = f["rT_all"][healthy][:, 365 + 20 :]
    assert np.nanmedian(np.abs(r_t)) < 0.8  # K
    assert np.isnan(f["rT_all"][:, :5]).all()  # warm-up days are NaN


def test_faulty_substations_stand_out(small_net):
    a = to_arrays(small_net.meters)
    sig = Signature().fit(a, slice(0, 365 * 24))
    f = window_features(a, sig)
    sg = feature_sigma({k: v[:, :365] for k, v in f.items()})
    z = zscore_features(f, sg)
    pop = small_net.pop
    i = int(np.flatnonzero(pop["fault"] == "bypass")[0])
    assert np.nanmax(z["rT_all"][i, 365:]) > 8
    j = int(np.flatnonzero(pop["fault"] == "sensor_swap")[0])
    assert np.nanmax(f["neg_dT"][j, 365:]) > 0.9


# ---------------------------------------------------------------- alarms
def test_persistent_labels_needs_five_of_seven():
    codes = np.zeros((1, 20), dtype=int)
    codes[0, 3:7] = 2  # only 4 days
    assert persistent_labels(codes)[0].max() == 0
    codes[0, 8] = 2  # day 8 sees days 2..8: days 3,4,5,6,8 -> 5 of 7
    assert persistent_labels(codes)[0, 8] == 2


def test_extract_alarms_diagnosis_and_pre_onset():
    codes = np.zeros((2, 100), dtype=int)
    codes[0, 50:100] = LABEL_CODE["bypass"]
    codes[1, 10:20] = LABEL_CODE["fouled_hex"]  # premature alarm
    codes[1, 60:100] = LABEL_CODE["wrong_curve"]
    al = extract_alarms(codes, 0, np.array([40, 50]))
    assert al.day[0] == 54 and LABELS[al.label[0]] == "bypass"
    assert al.pre_day[0] == -1
    assert al.pre_day[1] >= 10 and al.day[1] == 64
    assert LABELS[al.label[1]] == "wrong_curve"


def test_extract_alarms_no_alarm():
    al = extract_alarms(np.zeros((3, 40), dtype=int), 0)
    assert (al.day == -1).all() and (al.label == 0).all()


# ---------------------------------------------------------------- rules
KEYS = ["rT_all", "rT_lo", "rT_hi", "rV_all", "rV_lo", "rV_hi", "rP_all", "rP_lo", "rP_hi",
        "rP_night", "rP_day", "neg_dT"]


def _feats(**over):
    raw = {k: np.zeros((1, 1)) for k in KEYS}
    z = {k: np.zeros((1, 1)) for k in KEYS}
    for k, (r, zz) in over.items():
        raw[k] = np.full((1, 1), r, dtype=float)
        z[k] = np.full((1, 1), zz, dtype=float)
    return raw, z


def _label(**over):
    raw, z = _feats(**over)
    code, score = apply_rules(raw, z, RuleConfig())
    return LABELS[int(code[0, 0])], float(score[0, 0])


def test_rules_healthy():
    assert _label()[0] == "healthy"


def test_rule_sensor_swap_wins_over_everything():
    lab, _ = _label(neg_dT=(0.9, 90), rT_all=(40, 100), rP_all=(-1, -500))
    assert lab == "sensor_swap"


def test_rule_sensor_drift():
    assert _label(rT_all=(5, 12), rP_all=(-0.1, -3), rV_all=(0.0, 0.3))[0] == "sensor_drift"


def test_rule_bypass_all_load_levels():
    lab, _ = _label(rT_all=(6, 15), rT_lo=(6, 14), rT_hi=(6, 15), rV_all=(0.25, 6), rV_hi=(0.25, 6))
    assert lab == "bypass"


def test_rule_fouled_exchanger_only_at_high_load():
    lab, _ = _label(rT_all=(1.0, 2), rT_hi=(2.5, 7), rT_lo=(0.0, 0), rV_hi=(0.1, 2))
    assert lab == "fouled_hex"


def test_rule_wrong_curve_energy_up_with_hot_return():
    lab, _ = _label(rT_all=(3, 8), rT_hi=(3.5, 9), rV_all=(0.1, 4), rV_hi=(0.1, 5),
                    rP_all=(0.1, 5), rP_hi=(0.1, 6))
    assert lab == "wrong_curve"


def test_rule_dhw_circulation_base_load():
    lab, _ = _label(rT_all=(5, 10), rT_lo=(7, 20), rV_all=(0.3, 8), rV_lo=(0.4, 15), rP_all=(0.1, 3),
                    rP_lo=(0.2, 10), rP_night=(0.2, 9), rP_day=(0.1, 4))
    assert lab == "dhw_circulation"


def test_rule_stuck_valve_flow_up_but_return_not_hot():
    lab, _ = _label(rT_all=(-1, -3), rV_all=(0.3, 9), rP_all=(0.3, 8), rP_day=(0.3, 8), rP_night=(0.3, 9))
    assert lab == "stuck_valve"


def test_rule_no_setback_night_only():
    lab, _ = _label(rP_night=(0.15, 8), rP_day=(0.01, 0.5), rP_all=(0.05, 2))
    assert lab == "no_setback"


def test_rules_ignore_small_deviations():
    assert _label(rT_all=(0.8, 6), rV_all=(0.1, 2.0))[0] == "healthy"  # below min_rt_k
    assert _label(rT_all=(3, 3.0))[0] == "healthy"  # below sigma threshold


def test_rules_handle_nan_features():
    raw, z = _feats()
    for k in raw:
        raw[k][:] = np.nan
        z[k][:] = np.nan
    code, _ = apply_rules(raw, z)
    assert code[0, 0] == 0


def test_explain_mentions_numbers():
    txt = explain("bypass", {"rT_all": 6.2, "rV_all": np.log(1.25), "rP_all": 0.0})
    assert "6.2 K" in txt and "+25%" in txt
    assert explain("healthy", {}).startswith("No rule fired")


# ---------------------------------------------------------------- cost
def test_tariff_validation_and_values():
    with pytest.raises(ValueError):
        TariffConfig(chf_per_km3=-1)
    t = TariffConfig()
    assert excess_cost_chf(1000.0, t) == pytest.approx(1000 * t.chf_per_km3)
    assert excess_cost_chf(-5.0, t) == 0.0
    assert metering_exposure_chf(100.0, t) == pytest.approx(100 * t.energy_chf_per_kwh)
    assert retail_value_per_km3(t) == pytest.approx(1.163 * 0.17)


# ---------------------------------------------------------------- metrics
def test_prf_edge_cases():
    assert prf(0, 0, 0) == (0.0, 0.0, 0.0)
    p, r, f = prf(8, 2, 2)
    assert (p, r) == (0.8, 0.8) and f == pytest.approx(0.8)


def _pop(faults):
    return pd.DataFrame({"substation": [f"S{i}" for i in range(len(faults))], "fault": faults,
                         "onset_day": [10 if f != "healthy" else -1 for f in faults]})


def test_detection_metrics_counts_premature_alarm_as_false_positive():
    pop = _pop(["bypass", "bypass", "healthy", "healthy"])
    al = Alarms(day=np.array([20, -1, 30, -1]), pre_day=np.array([-1, -1, -1, -1]), label=np.zeros(4, dtype=int))
    m = detection_metrics(pop, al)
    assert (m["tp"], m["fn"], m["fp"]) == (1, 1, 1)
    al2 = Alarms(day=np.array([20, -1, -1, -1]), pre_day=np.array([5, -1, -1, -1]), label=np.zeros(4, dtype=int))
    assert detection_metrics(pop, al2)["fp"] == 1


def test_ranking_metrics():
    pop = _pop(["bypass"] * 3 + ["healthy"] * 17)
    rate = np.zeros(20)
    rate[[0, 1]] = [900, 800]
    rate[10] = 700
    m = ranking_metrics(pop, rate, 0.10)
    assert m["k"] == 2 and m["hits"] == 2 and m["share_true_faults_top"] == 1.0
    m3 = ranking_metrics(pop, rate, 0.15)
    assert m3["k"] == 3 and m3["hits"] == 2
