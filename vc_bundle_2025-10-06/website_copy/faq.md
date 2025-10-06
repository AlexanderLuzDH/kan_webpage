Title: FAQ — certified predictions and abstention

Is abstention a weakness?
- No. Abstention is a safety valve and cost shield. You only pay for certified predictions; abstentions route to your queue with full context.

How do you compute a “certified” prediction?
- We compute a confidence bound relative to a threshold τ. If the bound meets τ, we mark the output certified; otherwise we abstain.

What happens under distribution shift?
- Coverage drops first (more abstentions). Accuracy on certified outputs remains high. You can set alerts to switch into conservative mode.

Do you train on my data?
- No gradient training by default. You can add labels to improve coverage; the system updates without backprop.

How do I deploy?
- Options: cloud API, VPC, or on‑prem SDK. All provide the same certification semantics and audit trail.

What’s the ROI?
- Savings ≈ (legacy review rate − abstention‑driven review rate) × cost per review − certified decisions × price. Use our calculator.

