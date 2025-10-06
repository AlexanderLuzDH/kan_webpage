// Simple energy estimator for training-free vs fine-tune comparisons
// Energy (Wh) = Power (W) * Time (hours)

function energyCompare({
  gpuWatts = 300,          // e.g., 300 W for a single GPU
  gpuMinutes = 30,         // fine-tune duration in minutes
  cpuWatts = 65,           // e.g., 65 W CPU
  decisionSeconds = 0.18,  // total time for decision stage over a batch (e.g., 10k)
  embeddingSeconds = 0     // optional, set if recomputing embeddings
} = {}) {
  const gpuWh = (gpuWatts * (gpuMinutes / 60));
  const decWh = cpuWatts * (decisionSeconds / 3600);
  const embWh = cpuWatts * (embeddingSeconds / 3600);
  const totalWh = decWh + embWh;
  const ratio = gpuWh > 0 ? (gpuWh / Math.max(totalWh, 1e-12)) : Infinity;
  return {
    gpuWh: +gpuWh.toFixed(3),
    decisionWh: +decWh.toFixed(6),
    embeddingWh: +embWh.toFixed(3),
    totalWh: +totalWh.toFixed(3),
    reductionX: ratio === Infinity ? Infinity : +ratio.toFixed(1)
  };
}

// Example (CIFAR-10 zero-shot, 10k):
// energyCompare({ gpuWatts: 300, gpuMinutes: 30, cpuWatts: 65, decisionSeconds: 0.18, embeddingSeconds: 282 })

if (typeof module !== 'undefined') module.exports = { energyCompare };

