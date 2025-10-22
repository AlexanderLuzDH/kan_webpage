Website Copy Pack

Contents
- Pages (Markdown)
  - homepage.md — hero, benefits, proof points, visuals
  - use_cases.md — fraud, moderation, tele‑triage wedges
  - pricing.md — pay‑per‑certified tiers, definitions, SLA
  - trust.md — Coverage, Certified Accuracy, Abstention, audit/data handling
  - faq.md — common buyer/investor questions
  - charts.md — chart → CSV mapping and regeneration notes
- Scripts/Widgets
  - roi_calculator.js — minimal ROI estimator (embed as a widget)
  - demo_schema.json — schema for certified prediction demo/API
  - energy_calculator.js — compute Wh and rough reduction vs fine-tuning (100–1000× framing)
- Assets
  - assets/*.png — all charts used by the pages (self‑contained)
  - data/*.csv — CSVs backing the charts (download links)

How to use
- Wire these Markdown files into your site/CMS; keep image paths as `assets/...` and CSV downloads as `data/...`.
- Optionally expose charts.md as a “Charts & Data” page or use it internally.

Optional (regeneration)
- If you want to regenerate charts from code, use the repository scripts and source CSVs; otherwise, this pack is ready to publish as static content.
