KAN-∞ hits 89.1% zero-shot accuracy on CIFAR-10 — no labels, no training

Text-only class prompts + a universal extension law deliver training-free vision

[City], [Date] — The team behind KAN-∞ today announced a landmark result: 89.1% zero-shot accuracy on CIFAR-10 using only text descriptions of classes—no image labels and no fine-tuning. The system pairs a public vision-language encoder for “world knowledge” with KAN-∞’s training-free mathematical extender, turning a handful of text prompts (“a photo of a cat,” “a photo of a truck,” …) into a high-accuracy image classifier in seconds on CPU.

“This shows you don’t need to fine-tune a model every time your world changes. With just class names and KAN-∞’s universal law, you get a usable classifier—immediately,” said the project’s lead researcher.

Why this matters

Zero labels, zero fine-tuning: Move from idea to working classifier in minutes—not days—while avoiding costly, risky training cycles.

Green by construction: Training-free adaptation typically uses 10–100× less compute than standard fine-tuning.

Certifiable behavior: KAN-∞ provides distance-based confidence and safe abstention (only answer when in-distribution), giving teams governance they can trust.

How it works (plain English)

A public vision-language model maps images and text class prompts into a common space.

KAN-∞ treats those text prompts as “boundary points” and applies a universal Lipschitz-minimal extension (Soft-KAN).

The result is a unique, training-free classifier with a built-in distance-to-boundary signal—your confidence and safety lever.

Result reported: CIFAR-10 zero-shot top-1 = 89.06% (no labels used; no task-specific training).
When a few labels are available, KAN-∞ fuses zero-shot prompts with few-shot examples, often surpassing zero-shot while staying training-free.

What this unlocks (immediate use cases)

Instant product & content moderation
Launch a new category today: provide class names, get a working classifier with abstain-when-unsure behavior. Ideal for marketplaces, UGC platforms, ad safety.

Retail & catalog intelligence
Cold-start recognition for new SKUs without annotation. Add a few examples later to refine—no retraining required.

Edge & robotics
On-device recognition with CPU-only inference and a confidence gate. Safer operation in factories, warehouses, and drones.

Security & compliance
Abstain on out-of-distribution inputs by thresholding distance-to-boundary; route uncertain cases to human review.

RAG/tool routing for LLM stacks
Use KAN-∞ as a training-free router to pick tools or retrieval sources from text-only labels (“use OCR,” “search docs,” “summarize”), with certificates.

Active learning you can prove
KAN-∞ provides the top-N “next labels” that most reduce worst-case error—turning sparse labeling budgets into maximal gains.

Quotes

“Zero-shot is no longer a party trick. With KAN-∞, it’s a deployable path: start from prompts, add a few examples, and you’re production-ready—without training.” — Collaborating Scientist

“Governance teams finally get a mathematical confidence knob. If distance is high, abstain; if low, act. It’s certified caution, not a heuristic.” — Industry Advisor

Availability

KAN-∞ is offered as a lightweight SDK/API that plugs into existing ML/LLM stacks. The team is inviting three design partners in retail, industrial/robotics, and online platforms to pilot zero/few-shot classification with abstention—focusing on time-to-value, cost cuts, and safety.

Media / Pilot Inquiries:
[Your Name / Organization] • [Email] • [Phone]

At-a-glance

Result: CIFAR-10 zero-shot 89.1%, no labels, no fine-tuning, CPU-ready

Benefits: Training-free, 10–100× cheaper, certifiable confidence/abstention, active-learning suggestions

Works with: Images today; text & tabular via embeddings; pairs naturally with LLMs as a router/safety layer

Next: Few-shot fusion benchmarks, coverage-accuracy curves, and edge deployments with partners