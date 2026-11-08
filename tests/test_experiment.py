import json

import matplotlib.image as mpimg
import pandas as pd
import pytest

from dhfault.cli import main
from dhfault.plots import energy_signature_png, figure_main, figure_post
from dhfault.report import build_worklist, write_report_cards
from dhfault.runner import run_experiment


@pytest.fixture(scope="module")
def experiment():
    """Smoke-sized experiment (not the README numbers): 48 tuning and 64 evaluation substations."""
    return run_experiment(n_tune=48, per_type_tune=2, n_eval=64, per_type_eval=3)


def test_detection_is_useful_even_on_a_tiny_network(experiment):
    d = experiment.results["detection"]["rules"]
    assert d["recall"] >= 0.8
    assert d["false_alarm_substations"] <= 0.1 * d["n_healthy"]


def test_results_have_all_sections(experiment):
    r = experiment.results
    for key in ("network", "detection", "per_type", "delay_days", "ranking", "chf", "tariff", "headline"):
        assert key in r
    assert r["network"]["faulty"] == 24
    assert {row["fault"] for row in r["per_type"]["rules"]} == {
        "fouled_hex", "bypass", "stuck_valve", "wrong_curve", "dhw_circulation",
        "sensor_swap", "sensor_drift", "no_setback",
    }
    json.dumps(r, default=float)  # serialisable


def test_ranking_puts_true_faults_first(experiment):
    r = experiment.results["ranking"]
    assert r["share_true_faults_top"] >= 0.8


def test_cost_estimate_close_to_ground_truth(experiment):
    chf = experiment.results["chf"]
    assert chf["true_recoverable_after_alarm_detected_physical"] > 0
    assert abs(chf["relative_error_estimate"]) < 0.35


def test_worklist_sorted_and_actionable(experiment):
    wl = build_worklist(experiment.analysis, experiment.alarms["rules"], pop=experiment.net.pop)
    assert wl["est_chf_per_year"].is_monotonic_decreasing
    assert wl.iloc[0]["diagnosis"] != "healthy" and wl.iloc[0]["action"]
    assert wl["rank"].tolist() == list(range(1, len(wl) + 1))


def test_report_cards_written(experiment, tmp_path):
    wl = write_report_cards(experiment.analysis, experiment.alarms["rules"], tmp_path, top=3, pop=experiment.net.pop)
    card = (tmp_path / "cards" / f"{wl.iloc[0]['substation']}.md").read_text(encoding="utf-8")
    assert card.startswith("# Report card:")
    assert "## Why" in card and "Advised action" in card and "Ground truth (simulation only)" in card
    assert (tmp_path / "worklist.csv").exists()


def test_figure_sizes_are_exact(experiment, tmp_path):
    figure_main(experiment, tmp_path / "figure.png")
    figure_post(experiment, tmp_path / "post.png")
    energy_signature_png(experiment.analysis, 0, tmp_path / "e.png")
    assert mpimg.imread(tmp_path / "figure.png").shape[:2] == (900, 1600)
    assert mpimg.imread(tmp_path / "post.png").shape[:2] == (700, 960)


def test_demo_cli_small(tmp_path, capsys):
    rc = main(["demo", "--small", "--out", str(tmp_path / "o"), "--figures", str(tmp_path / "f"), "--cards", "2"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Detection of" in out and "SIMULATED" in out
    assert (tmp_path / "f" / "figure.png").exists()
    res = pd.read_json(tmp_path / "o" / "results.json", typ="series")
    assert "detection" in res.index
