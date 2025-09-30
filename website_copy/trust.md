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

Visual aids (from repo)
- CIFAR‑10 coverage vs accuracy: `web_assets/cifar10_coverage.png`

