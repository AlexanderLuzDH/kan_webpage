Title: Only pay for certified predictions. Everything else safely abstains.

Subtitle: Pick your guarantee (95% or 99%). We deliver coverage at that bar — abstentions are free and routed to review.

Primary CTA: Get Early Access
Secondary CTA: Run the cost & coverage calculator

Key benefits
- Training‑free accuracy in minutes: ship models without gradient training, on CPU.
- Guaranteed confidence: tune a coverage/accuracy trade‑off via a single threshold (τ).
- Built‑in safety: when uncertain, the system abstains and hands off to your queue.
- Simple pricing: pay only for certified outputs that meet your target.

Proof points (from current artifacts)
- CIFAR‑10 Soft‑KAN few‑shot (test=2k, CLIP text prototypes, CPU):
  - 1 label/class: 89.5% acc; 5/class: 90.3%; 10/class: 91.2%
  - Predict time: ~0.36–0.40s; memory ~1.67 GB
- AG News (10k subset, mpnet encoder):
  - 1 label/class: 47.1%; 5/class: 70.8%; 10/class: 74.0%
  - Predict time: ~0.003–0.03s; memory ~7.3 GB
- Coverage curves show how accuracy rises as you tighten certification: see CIFAR‑10 L=5 and L=10 curves.

Visuals (use from repo)
- web_assets/cifar10_fewshot_bars.png
- web_assets/cifar10_coverage.png
- web_assets/agnews_label_efficiency.png
- web_assets/agnews_head_to_head.png

Explainer: How it works
- We compute decisions over a similarity graph with a safety knob τ.
- If confidence meets your target → return a certified prediction.
- If not → abstain (free), route to human/system review.
- This yields predictable spend and auditable accuracy at your chosen bar.

Short FAQ preview
- What happens to abstentions? They go to your queue with full context; you can auto‑approve common cases.
- Does it train? No gradient training; CPU‑friendly. You can add labels anytime to increase coverage.
- Can I self‑host? Yes, SDK and on‑prem options.
