Title: Charts & Data Sources

Overview
- All charts are generated from CSV artifacts under `kan_inf_outputs/` using `scripts/plot_marketing_assets.py`.
- CSVs are downloadable to support reproducibility and investor due diligence.

Charts
- CIFAR‑10 few‑shot bars: `web_assets/cifar10_fewshot_bars.png`
  - Source: `data/cifar10_softkan_summary_v2.csv`
- CIFAR‑10 coverage vs certified accuracy: `web_assets/cifar10_coverage.png`
  - Source: `data/cifar10_coverage_L5.csv`, `data/cifar10_coverage_L10.csv`
- AG News label efficiency (mpnet): `web_assets/agnews_label_efficiency.png`
  - Source: `data/agnews_softkan_summary_v2.csv`
- AG News Soft‑KAN vs AMLE (10k): `web_assets/agnews_head_to_head.png`
  - Source: `data/agnews_softkan_summary_v2.csv`, `data/agnews_amle_summary_v2.csv`
- AG News coverage vs certified accuracy (10k mpnet): `web_assets/agnews_coverage.png`
  - Source: `data/agnews_coverage_L5.csv`, `data/agnews_coverage_L10.csv`
- AG News encoder trade‑off (MiniLM vs mpnet): `web_assets/agnews_mini_vs_mpnet.png`
  - Source: `data/agnews_softkan_summary_v2_mini.csv`, `data/agnews_softkan_summary_v2.csv`

Regen commands
- Charts: `python scripts/plot_marketing_assets.py`
- AG News coverage CSVs: `python scripts/agnews_coverage.py`
