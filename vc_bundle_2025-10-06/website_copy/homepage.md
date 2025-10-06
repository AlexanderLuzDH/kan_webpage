Title: Only pay for certified predictions. Everything else safely abstains.

Subtitle: Pick your guarantee (95% or 99%). We deliver coverage at that bar — abstentions are free and routed to review.

Primary CTA: Get Early Access
Secondary CTA: Run the cost & coverage calculator

Key benefits
- No gradient training: ship models without backprop; CPU‑friendly.
- Millisecond decision step on CPU; embeddings can be cached or streamed.
- Guaranteed confidence: tune a coverage/accuracy trade‑off via a single threshold (τ).
- Built‑in safety: when uncertain, the system abstains and hands off to your queue.
- Simple pricing: pay only for certified outputs that meet your target.

Proof points (from current artifacts)
- CIFAR-10 zero-shot (test=10k, CLIP text prototypes, CPU): 89.6% accuracy; decision step ~0.18s; memory ~1.90 GB
- CIFAR‑10 Soft‑KAN few‑shot (test=2k, CLIP text prototypes, CPU):
  - 1 label/class: 89.5% acc; 5/class: 90.3%; 10/class: 91.2%
  - Predict time: ~0.36–0.40s; memory ~1.67 GB
- AG News (50k subset, mpnet encoder):
  - 1 label/class: 53.4%; 5/class: 68.7%; 10/class: 79.6%
  - Predict time: ~0.03–0.43s; memory ~7.9–8.3 GB
- Coverage curves show how accuracy rises as you tighten certification: see CIFAR‑10 L=5 and L=10 curves.

Performance & energy note
- Decision stage runs on CPU in milliseconds; no gradient training.
- Energy math and assumptions (100–1000× vs fine‑tuning) on the Trust page.

Visuals (bundled in this folder)
- assets/cifar10_fewshot_bars.png
- assets/cifar10_coverage.png
- assets/agnews_label_efficiency.png
- assets/agnews_head_to_head.png
- assets/agnews_coverage.png
- assets/agnews_mini_vs_mpnet.png

Explainer: How it works
- We compute decisions over a similarity graph with a safety knob τ.
- If confidence meets your target → return a certified prediction.
- If not → abstain (free), route to human/system review.
- This yields predictable spend and auditable accuracy at your chosen bar.

Short FAQ preview
- What happens to abstentions? They go to your queue with full context; you can auto‑approve common cases.
- Does it train? No gradient training; CPU‑friendly. You can add labels anytime to increase coverage.
- Can I self‑host? Yes, SDK and on‑prem options.

