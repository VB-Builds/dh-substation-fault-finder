"""Command line: ``python -m dhfault demo | predist | analyse | simulate``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .cost import TariffConfig


def _print_table(df: pd.DataFrame) -> None:
    print(df.to_string(float_format=lambda x: f"{x:.2f}"))


def cmd_demo(args: argparse.Namespace) -> int:
    from .plots import energy_signature_png, figure_main, figure_post
    from .report import write_report_cards
    from .runner import run_experiment

    tariff = TariffConfig(chf_per_km3=args.chf_per_km3)
    if args.small:
        ex = run_experiment(n_tune=48, per_type_tune=2, n_eval=64, per_type_eval=3, tariff=tariff)
    else:
        ex = run_experiment(tariff=tariff)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(ex.results, indent=2, default=float), encoding="utf-8")
    wl = write_report_cards(ex.analysis, ex.alarms["rules"], out, tariff, top=args.cards, pop=ex.net.pop)
    r = ex.results
    print("Network (SIMULATED):", r["network"])
    print("\nDetection of 'is there a fault' (substation level)")
    _print_table(pd.DataFrame(r["detection"]).T[["tp", "fp", "fn", "precision", "recall", "f1"]])
    print("\nDiagnosis per fault type, rule library")
    cols = ["n", "detected", "precision", "recall", "f1"]
    _print_table(pd.DataFrame(r["per_type"]["rules"]).set_index("fault")[cols])
    print("\nDiagnosis per fault type, supervised baseline (trained on a different simulated network)")
    _print_table(pd.DataFrame(r["per_type"]["supervised"]).set_index("fault")[cols])
    print("\nDelay (days):", r["delay_days"])
    print("Ranking top 10%:", r["ranking"])
    print("CHF:", r["chf"])
    if args.figures:
        fig_dir = Path(args.figures)
        fig_dir.mkdir(parents=True, exist_ok=True)
        figure_main(ex, fig_dir / "figure.png")
        figure_post(ex, fig_dir / "post-figure.png")
        first = wl.iloc[0]["substation"]
        idx = ex.analysis.arrays.ids.index(first)
        energy_signature_png(ex.analysis, idx, fig_dir / "energy-signature.png")
        print(f"\nfigures written to {fig_dir}")
    print(f"\nwork list and report cards in {out}")
    return 0


def cmd_predist(args: argparse.Namespace) -> int:
    from .predist import evaluate_events, fetch, summarise

    root = fetch(Path(args.data)) if args.fetch else Path(args.data) / "predist"
    if not (root / "manufacturer 1" / "faults.csv").exists():
        print(f"dataset not found under {root}; run with --fetch to download it (about 270 MB)")
        return 2
    res = evaluate_events(root)
    summ = summarise(res)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "predist_events.csv", index=False)
    summ.to_csv(out / "predist_summary.csv")
    print(f"events: {len(res) // max(res['method'].nunique(), 1)}, evaluable: "
          f"{int(res.groupby('method')['evaluable'].sum().max())}")
    _print_table(summ)
    return 0


def cmd_analyse(args: argparse.Namespace) -> int:
    from .pipeline import analyse, extract_alarms, rule_codes
    from .report import write_report_cards

    df = pd.read_csv(args.csv, parse_dates=["time"]).set_index(["substation", "time"]).sort_index()
    an = analyse(df, args.ref_days)
    alarms = extract_alarms(rule_codes(an), args.ref_days)
    wl = write_report_cards(an, alarms, Path(args.out), TariffConfig(chf_per_km3=args.chf_per_km3), top=args.cards)
    _print_table(wl.head(15).drop(columns=["action"]))
    return 0


def cmd_simulate(args: argparse.Namespace) -> int:
    from .sim import SimConfig, simulate_network

    net = simulate_network(SimConfig(n_substations=args.n, faults_per_type=args.per_type, seed=args.seed))
    net.meters.reset_index().to_csv(args.out, index=False)
    net.pop.to_csv(Path(args.out).with_suffix(".labels.csv"), index=False)
    print(f"wrote {args.out} and labels")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="dhfault", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo", help="simulated network: train, evaluate, write work list, cards, figures")
    d.add_argument("--out", default="out/demo")
    d.add_argument("--figures", default=None, help="directory for figure.png / post-figure.png")
    d.add_argument("--small", action="store_true", help="tiny network (smoke test, not the README numbers)")
    d.add_argument("--cards", type=int, default=10)
    d.add_argument("--chf-per-km3", type=float, default=TariffConfig().chf_per_km3)
    d.set_defaults(func=cmd_demo)
    b = sub.add_parser("predist", help="evaluate on the public labelled benchmark (downloads with --fetch)")
    b.add_argument("--data", default="cache")
    b.add_argument("--fetch", action="store_true")
    b.add_argument("--out", default="out/predist")
    b.set_defaults(func=cmd_predist)
    a = sub.add_parser("analyse", help="run on your own hourly meter CSV")
    a.add_argument("csv", help="columns: substation,time,t_out,t_supply,t_return,flow_lph,power_kw")
    a.add_argument("--ref-days", type=int, default=365, help="days of fault-free history at the start")
    a.add_argument("--out", default="out/analyse")
    a.add_argument("--cards", type=int, default=10)
    a.add_argument("--chf-per-km3", type=float, default=TariffConfig().chf_per_km3)
    a.set_defaults(func=cmd_analyse)
    s = sub.add_parser("simulate", help="write a simulated meter CSV and labels")
    s.add_argument("--out", default="out/sim_meters.csv")
    s.add_argument("--n", type=int, default=60)
    s.add_argument("--per-type", type=int, default=2)
    s.add_argument("--seed", type=int, default=1)
    s.set_defaults(func=cmd_simulate)
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
