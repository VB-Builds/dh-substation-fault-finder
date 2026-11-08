import numpy as np
import pandas as pd
import pytest

from dhfault import predist
from dhfault.cli import main

EMPTY_FAULTS = ["Event ID", "substation ID", "Report date", "Training start", "Training end",
                "efd_possible", "Fault label", "Monitoring potential"]
EMPTY_NORMAL = ["Event ID", "substation ID", "Event start", "Event end", "Training start", "Training end"]


def _write_dataset(root, fault_in_test=True):
    d = root / "manufacturer 1"
    (d / "operational_data").mkdir(parents=True)
    idx = pd.date_range("2019-10-01", "2020-03-01", freq="10min")
    rng = np.random.default_rng(0)
    t_out = 5 + 8 * np.sin(np.arange(len(idx)) / (144 * 20)) + rng.normal(0, 0.5, len(idx))
    ret = 45 - 0.8 * t_out + rng.normal(0, 0.4, len(idx))
    sup = 80 - 0.6 * t_out
    power = 40 - 1.5 * t_out + rng.normal(0, 1, len(idx))
    flow = power / (1.163 * (sup - ret)) * 1000
    ret_f = ret.copy()
    if fault_in_test:
        ret_f[idx >= "2020-02-10"] += 12
    pd.DataFrame({
        "timestamp": idx, "outdoor_temperature": t_out, "p_net_supply_temperature": sup,
        "p_net_return_temperature": ret_f, "p_net_meter_flow": flow, "p_net_meter_heat_power": power,
    }).to_csv(d / "operational_data" / "substation_1.csv", sep=";", index=False)
    pd.DataFrame({
        "Event ID": [0, 1], "substation ID": [1, 1],
        "Report date": ["2020-02-20 10:00:00", "2020-02-20 10:00:00"],
        "Problem EN": ["no heat", "no heat"], "Event description EN": ["x", "y"],
        "Possible anomaly start": [None, None], "Possible anomaly end": [None, None],
        "Training start": ["2019-10-15 00:00:00", "2019-10-15 00:00:00"],
        "Training end": ["2020-02-06 10:00:00", "2020-02-06 10:00:00"],
        "efd_possible": [True, False], "Fault label": ["Valve", "Valve"], "Monitoring potential": [3.2, 3.2],
    }).to_csv(d / "faults.csv", sep=";", index=False)
    pd.DataFrame({
        "Event ID": [5], "substation ID": [1], "Event start": ["2019-12-10"], "Event end": ["2019-12-20"],
        "Training start": ["2019-10-15 00:00:00"], "Training end": ["2019-12-10"],
    }).to_csv(d / "normal_events.csv", sep=";", index=False)
    m2 = root / "manufacturer 2"
    m2.mkdir()
    pd.DataFrame(columns=EMPTY_FAULTS).to_csv(m2 / "faults.csv", sep=";", index=False)
    pd.DataFrame(columns=EMPTY_NORMAL).to_csv(m2 / "normal_events.csv", sep=";", index=False)
    return root


def test_load_events_parses_dates_and_names(tmp_path):
    faults, normals = predist.load_events(_write_dataset(tmp_path))
    assert len(faults) == 2 and len(normals) == 1
    assert pd.api.types.is_datetime64_any_dtype(faults["report_date"])
    assert normals.loc[0, "substation_id"] == 1


def test_load_hourly_resamples_and_converts_units(tmp_path):
    root = _write_dataset(tmp_path)
    h = predist.load_hourly(root, 1, 1)
    assert (h.index[1] - h.index[0]) == pd.Timedelta("1h")
    assert h["flow"].median() > 0 and h["power"].median() > 0
    # manufacturer 2 reports ml/h and W
    raw = pd.read_csv(root / "manufacturer 1" / "operational_data" / "substation_1.csv", sep=";")
    (root / "manufacturer 2" / "operational_data").mkdir(parents=True)
    raw.assign(p_net_meter_flow=raw["p_net_meter_flow"] * 1000,
               p_net_meter_heat_power=raw["p_net_meter_heat_power"] * 1000).to_csv(
        root / "manufacturer 2" / "operational_data" / "substation_1.csv", sep=";", index=False)
    h2 = predist.load_hourly(root, 2, 1)
    assert h2["flow"].median() == pytest.approx(h["flow"].median(), rel=1e-6)
    assert h2["power"].median() == pytest.approx(h["power"].median(), rel=1e-6)


@pytest.mark.parametrize("method", ["signature", "isolation_forest"])
def test_detector_flags_injected_fault_and_not_normal_period(tmp_path, method):
    h = predist.load_hourly(_write_dataset(tmp_path), 1, 1)
    train = (pd.Timestamp("2019-10-15"), pd.Timestamp("2020-02-06 10:00"))
    alarm, t0, _ = predist._detect(method, h, train, (train[1], pd.Timestamp("2020-02-20 10:00")))
    assert alarm and t0 >= pd.Timestamp("2020-02-10")
    quiet, _, _ = predist._detect(method, h, (pd.Timestamp("2019-10-15"), pd.Timestamp("2019-12-10")),
                                  (pd.Timestamp("2019-12-10"), pd.Timestamp("2019-12-20")))
    assert not quiet


def test_detector_refuses_tiny_training_window(tmp_path):
    h = predist.load_hourly(_write_dataset(tmp_path), 1, 1)
    alarm, _, why = predist._detect("signature", h, (pd.Timestamp("2019-10-15"), pd.Timestamp("2019-10-18")),
                                    (pd.Timestamp("2019-10-18"), pd.Timestamp("2019-10-25")))
    assert not alarm and "training" in why


def test_evaluate_events_end_to_end(tmp_path):
    res = predist.evaluate_events(_write_dataset(tmp_path), methods=("signature",))
    assert set(res["kind"]) == {"fault", "normal"}
    assert len(res[res["kind"] == "fault"]) == 1  # the efd_possible=False report is skipped
    row = predist.summarise(res).loc["signature"]
    assert row["fault_events"] == 1 and row["detected_before_report"] == 1
    assert row["normal_with_alarm"] == 0
    assert 0 < row["mean_lead_days"] <= 10.5  # alarm between 2020-02-10 and the report on 2020-02-20


def test_unknown_method_rejected(tmp_path):
    h = predist.load_hourly(_write_dataset(tmp_path), 1, 1)
    with pytest.raises(ValueError):
        predist._detect("magic", h, (pd.Timestamp("2019-10-15"), pd.Timestamp("2020-02-06")),
                        (pd.Timestamp("2020-02-06"), pd.Timestamp("2020-02-20")))


def test_cli_predist_missing_dataset_returns_2(tmp_path, capsys):
    assert main(["predist", "--data", str(tmp_path)]) == 2
    assert "dataset not found" in capsys.readouterr().out


def test_cli_simulate_and_analyse_roundtrip(tmp_path):
    csv = tmp_path / "m.csv"
    assert main(["simulate", "--out", str(csv), "--n", "16", "--per-type", "1", "--seed", "3"]) == 0
    assert csv.exists() and csv.with_suffix(".labels.csv").exists()
    out = tmp_path / "an"
    assert main(["analyse", str(csv), "--ref-days", "365", "--out", str(out), "--cards", "2"]) == 0
    wl = pd.read_csv(out / "worklist.csv")
    assert list(wl["rank"]) == list(range(1, 17))
    assert wl["est_chf_per_year"].is_monotonic_decreasing
    assert len(list((out / "cards").glob("*.md"))) == 2
