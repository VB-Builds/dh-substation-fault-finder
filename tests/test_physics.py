import numpy as np
import pytest

from dhfault.sim import (
    C_WATER,
    FAULT_TYPES,
    HEALTHY,
    SimConfig,
    _lmtd_cold_end,
    effectiveness_ntu,
    lmtd,
    make_population,
)
from dhfault.weather import load_hourly_temperature


def test_lmtd_known_value():
    assert lmtd(np.array([40.0]), np.array([20.0]))[0] == pytest.approx(20.0 / np.log(2.0))


def test_lmtd_equal_ends_is_the_difference():
    assert lmtd(np.array([15.0]), np.array([15.0]))[0] == pytest.approx(15.0)


def test_lmtd_cold_end_inverts_lmtd():
    d1 = np.array([30.0, 45.0, 12.0])
    d2 = np.array([5.0, 22.0, 3.0])
    back = _lmtd_cold_end(lmtd(d1, d2), d1)
    assert back == pytest.approx(d2, abs=1e-6)


def test_effectiveness_limits():
    ntu = 2.0
    ua = np.array([ntu])
    one = np.array([1.0])
    # Cr -> 0 (one stream has huge capacity): eps = 1 - exp(-NTU)
    assert effectiveness_ntu(ua, one, one * 1e9)[0] == pytest.approx(1 - np.exp(-ntu), rel=1e-6)
    # Cr = 1: eps = NTU / (1 + NTU)
    assert effectiveness_ntu(ua, one, one)[0] == pytest.approx(ntu / (1 + ntu))


def test_effectiveness_between_zero_and_one():
    rng = np.random.default_rng(0)
    e = effectiveness_ntu(rng.uniform(0.1, 20, 100), rng.uniform(0.5, 5, 100), rng.uniform(0.5, 5, 100))
    assert np.all((e > 0) & (e < 1))


def test_water_constant():
    assert C_WATER == pytest.approx(1.163)


def test_population_counts_and_labels():
    pop = make_population(SimConfig(n_substations=60, faults_per_type=3, seed=1))
    counts = pop["fault"].value_counts()
    assert counts[HEALTHY] == 60 - 3 * len(FAULT_TYPES)
    for ft in FAULT_TYPES:
        assert counts[ft] == 3
    assert (pop.loc[pop["fault"] == HEALTHY, "onset_day"] == -1).all()
    assert (pop.loc[pop["fault"] != HEALTHY, "onset_day"] >= 15).all()


def test_population_rejects_too_many_faults():
    with pytest.raises(ValueError):
        make_population(SimConfig(n_substations=5, faults_per_type=2))


def test_population_deterministic():
    a = make_population(SimConfig(n_substations=30, faults_per_type=1, seed=9))
    b = make_population(SimConfig(n_substations=30, faults_per_type=1, seed=9))
    assert a.equals(b)


def test_weather_file():
    wx = load_hourly_temperature()
    assert len(wx) == 2 * 365 * 24
    assert wx["temperature_2m"].notna().all()
    assert -20 < wx["temperature_2m"].min() < 0 < 25 < wx["temperature_2m"].max() < 40
    assert wx.index.is_monotonic_increasing and not wx.index.has_duplicates


def test_simulated_network_is_physical(small_net):
    m = small_net.meters
    assert m["t_return"].between(5, 90).all()
    assert (m["power_kw"] >= 0).all()
    healthy = small_net.pop.loc[small_net.pop["fault"] == HEALTHY, "substation"]
    mh = m.loc[healthy.tolist()]
    assert (mh["t_supply"] > mh["t_return"]).mean() > 0.999
    # energy is a cumulative counter
    e = m["energy_kwh"].unstack(level=0)
    assert (e.diff().dropna() >= -1e-3).all().all()


def test_energy_matches_flow_and_delta_t(small_net):
    m = small_net.meters.iloc[:5000].astype(float)
    p = C_WATER * (m["flow_lph"] / 1000.0) * (m["t_supply"] - m["t_return"])
    assert np.allclose(np.maximum(p, 0.0), m["power_kw"], rtol=1e-3, atol=1e-2)


def test_sensor_swap_gives_negative_delta_t_only_after_onset(small_net):
    pop = small_net.pop.set_index("substation")
    sid = pop.index[pop["fault"] == "sensor_swap"][0]
    s = small_net.meters.loc[sid]
    onset = (small_net.ref_days + int(pop.loc[sid, "onset_day"])) * 24
    dt = (s["t_supply"] - s["t_return"]).to_numpy()
    assert (dt[:onset] > 0).mean() > 0.999
    assert (dt[onset + 24 :] < 0).mean() > 0.999


def test_physical_faults_have_positive_true_excess_and_sensor_faults_zero(small_net):
    pop = small_net.pop.set_index("substation")
    t = small_net.truth.set_index("substation")["true_excess_km3_year"]
    for ft in ("bypass", "wrong_curve", "dhw_circulation", "fouled_hex"):
        sid = pop.index[pop["fault"] == ft][0]
        assert t[sid] > 0, ft
    for ft in ("sensor_swap", "sensor_drift"):
        sid = pop.index[pop["fault"] == ft][0]
        assert t[sid] == 0.0
    for sid in pop.index[pop["fault"] == HEALTHY]:
        assert t[sid] == 0.0


def test_bypass_raises_return_temperature(small_net):
    pop = small_net.pop.set_index("substation")
    sid = pop.index[pop["fault"] == "bypass"][0]
    s = small_net.meters.loc[sid]
    onset = (small_net.ref_days + int(pop.loc[sid, "onset_day"])) * 24
    day = (s.index.hour >= 8) & (s.index.hour < 18)
    cold = (s["t_out"] < 8).to_numpy()
    before = s["t_return"].to_numpy()[:onset][(day & cold)[:onset]].mean()
    after = s["t_return"].to_numpy()[onset:][(day & cold)[onset:]].mean()
    assert after > before + 2.0
