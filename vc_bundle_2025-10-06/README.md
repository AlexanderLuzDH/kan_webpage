VC Bundle — 2025-10-06
======================

What’s inside
- code/rcps_mfa_breast_cancer.py — RCPS‑MFA demo (tabular KAN‑∞ front‑end with multi‑stage features + SVM fallback). Produces CSVs and plots.
- code/trust/rcps.py — RCPS utilities (one‑sided CP bound, gating, threshold selection, timing helpers).
- code/kan_infty_speedrun_v2.py — Updated speedrun with RCPS flags for AG News and CIFAR‑10.
- code/scripts/pack_investor_metrics.py — Aggregates speedrun CSVs into a compact investor summary.
- website_copy/ — Site copy, charts metadata, and assets.
- docs/press_release0{1,2,3}.md — Draft press releases.

Quickstart
1) Install (CPU ok):
   pip install -U numpy pandas scikit-learn matplotlib scipy tqdm torch torchvision open_clip_torch sentence-transformers datasets psutil

2) Tabular RCPS‑MFA demo:
   python code/rcps_mfa_breast_cancer.py --no-plot
   Outputs:
   - rcps_mfa_results_strict.csv (α=1%, δ=5% strict regime)
   - rcps_mfa_results_practical.csv (α=5%, δ=5%)

3) AG News with RCPS (10k subset example):
   python code/kan_infty_speedrun_v2.py --task agnews --method softkan --subset 10000 \
     --labels-per-class 5 --auto-tau --rcps-enable --rcps-alpha 0.05 --rcps-delta 0.05 \
     --st-model sentence-transformers/all-mpnet-base-v2 --outdir ./kan_inf_outputs

4) CIFAR‑10 with RCPS (few‑shot + text prototypes):
   python code/kan_infty_speedrun_v2.py --task cifar10 --method softkan \
     --labels-per-class 1 5 10 --text-proto --auto-tau --rcps-enable \
     --rcps-alpha 0.05 --rcps-delta 0.05 --outdir ./kan_inf_outputs

5) Investor summary table:
   python code/scripts/pack_investor_metrics.py
   - Writes: kan_inf_outputs/investor_summary.csv

RCPS Flags in speedrun
- --rcps-enable: turn on RCPS gating (Soft‑KAN path)
- --rcps-alpha, --rcps-delta: risk/confidence
- --rcps-gate {prob,margin}: gating statistic (top‑prob currently)
- --rcps-cal-frac: calibration fraction (default 0.25)

Notes
- Guarantees refer to accepted‑subset error ≤ α with confidence ≥ 1−δ using one‑sided Clopper–Pearson on a held‑out calibration split.
- CIFAR fallbacks use AMLE when labeled points exist; otherwise Soft‑KAN.
- Plots on the website can be derived from CSVs under ./kan_inf_outputs and website_copy/data.

