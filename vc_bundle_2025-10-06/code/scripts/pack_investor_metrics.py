#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pack investor-facing metrics from speedrun CSVs into one compact table.

Reads CSVs under ./kan_inf_outputs and writes ./kan_inf_outputs/investor_summary.csv

Dependencies: pandas
"""
from __future__ import annotations

import os
from pathlib import Path
import pandas as pd


KEEP_COLS = [
    # common
    "dataset", "method", "labels_per_class",
    # accuracy and timing
    "acc_all", "acc_test", "prep_time_sec", "predict_time_sec", "mem_gb",
    # softkan / temp
    "tau", "auto_tau", "bias_correct",
    # rcps fields (if present)
    "rcps_enabled", "rcps_tau", "rcps_alpha", "rcps_delta",
    "rcps_accept_fraction", "rcps_abstain_fraction", "rcps_acc_accepted", "rcps_final_acc",
]


def load_csvs(outdir: Path) -> pd.DataFrame:
    rows = []
    for p in outdir.glob("*.csv"):
        name = p.name
        if not ("agnews_" in name or "cifar10_" in name):
            continue
        try:
            df = pd.read_csv(p)
            rows.append(df)
        except Exception:
            pass
    if not rows:
        return pd.DataFrame()
    df_all = pd.concat(rows, ignore_index=True, sort=False)
    return df_all


def main():
    outdir = Path("./kan_inf_outputs")
    outdir.mkdir(parents=True, exist_ok=True)
    df_all = load_csvs(outdir)
    if df_all.empty:
        print("No speedrun CSVs found under ./kan_inf_outputs")
        return
    # Subselect columns present
    cols = [c for c in KEEP_COLS if c in df_all.columns]
    df = df_all[cols].copy()
    # Friendly ordering
    if "dataset" in df.columns and "method" in df.columns and "labels_per_class" in df.columns:
        df.sort_values(["dataset","method","labels_per_class"], inplace=True)
    out_path = outdir / "investor_summary.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

