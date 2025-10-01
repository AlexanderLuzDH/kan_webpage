Title: Trust & Definitions — certified predictions with abstention by design

Core concepts
- Coverage (C): fraction of requests that return certified predictions at threshold τ.
- Certified Accuracy (Aτ): measured accuracy on the subset of certified predictions.
- Abstention Rate: 1 − C; these are routed to a review queue with full context.
- Threshold τ ("guarantee knob"): tighten to raise Aτ (accuracy) and lower C (coverage); loosen to raise C and lower Aτ.

Operational guarantees
- You select τ to target e.g., 95% or 99% certified accuracy.
- We monitor Aτ and C in real time; if data shifts, abstention rises first (safe‑by‑design).
- SLA option: if Aτ falls below target over a rolling window, we credit overages.

Audit and safety
- Every certified output includes: predicted label, score, τ, and certification status; every abstention includes a reason code.
- Full export: per‑request JSON lines for audit; roll‑up CSVs for compliance.

Data handling
- No gradient training on your data by default.
- PII: configurable redaction; optional on‑prem processing for sensitive workloads.

Visual aids (bundled)
- CIFAR‑10 coverage vs accuracy: `assets/cifar10_coverage.png`
- CIFAR‑10 reliability diagram (zero-shot, 2k cached): `assets/cifar10_reliability.png` (points in `data/cifar10_reliability_points.csv`)
- AG News coverage vs accuracy (10k subset, mpnet): `assets/agnews_coverage.png`
- Encoder trade‑off (MiniLM vs mpnet, 10k subset): `assets/agnews_mini_vs_mpnet.png`

Energy & Timing (assumptions and measured numbers)
- No gradient training: the training phase is eliminated entirely; only embedding + decision remain.
- Decision step on CPU (measured):
  - CIFAR-10 zero-shot (10k): ~0.18 s total; memory ~1.90 GB.
  - AG News Soft-KAN (10k mpnet): ~0.003–0.03 s total.
  - AG News Soft-KAN (50k mpnet): ~0.16–0.43 s total.
- Embedding on CPU (cacheable/streamable):
  - CIFAR-10 CLIP ViT-B/32: ~4.7 min for 10k (~28 ms/image).
  - AG News mpnet: ~6.3 min for 10k (~38 ms/doc); ~50 min for 50k.
- Energy framing (illustrative, see calculator):
  - GPU fine-tune baseline: 30 min at 300 W ≈ 150 Wh.
  - Our decision stage (CIFAR 10k): 0.18 s at 65 W ≈ 0.003 Wh → >50,000× smaller for the decision component.
  - Including CPU embeddings for the same 10k: ≈5 Wh total → ~30× smaller than the 150 Wh baseline. Larger fine-tunes and multi-epoch training typically yield 100–1000× reductions.
- Assumptions: CPU 65 W, GPU 300 W (replace with your hardware). Times sourced from the CSVs in `website_copy/data` and run logs.
- Calculator: use `website_copy/energy_calculator.js` or the ROI widget to plug your power/time.
