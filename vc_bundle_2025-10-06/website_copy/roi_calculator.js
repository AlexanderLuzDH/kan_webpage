// Minimal ROI calculator for pay-per-certified with abstention fallback
// Inputs:
// - volume: total decisions per month
// - legacyReviewRate: fraction (0..1) of items currently sent to manual review
// - ourCoverage: fraction (0..1) of items certified at chosen threshold (rest abstain)
// - costPerReview: $ per manual review
// - pricePerCertified: $ per certified decision (your tier)
// Returns: { monthlySavings, monthlySpend, monthlyReviewCost, notes }

function roiEstimate({ volume, legacyReviewRate, ourCoverage, costPerReview, pricePerCertified }) {
  const certified = volume * ourCoverage;
  const abstain = volume - certified;
  const legacyReviews = volume * legacyReviewRate;
  const ourReviews = abstain; // assume abstentions all go to review initially
  const monthlySpend = certified * pricePerCertified;
  const legacyReviewCost = legacyReviews * costPerReview;
  const ourReviewCost = ourReviews * costPerReview;
  const monthlySavings = legacyReviewCost - (ourReviewCost + monthlySpend);
  return {
    certified,
    abstain,
    monthlySpend: +monthlySpend.toFixed(2),
    monthlyReviewCost: +ourReviewCost.toFixed(2),
    monthlySavings: +monthlySavings.toFixed(2),
    notes: "Savings exclude second-order effects; tweak coverage to target accuracy."
  };
}

// Example:
// roiEstimate({ volume: 1000000, legacyReviewRate: 0.35, ourCoverage: 0.72, costPerReview: 2.5, pricePerCertified: 0.0005 })

if (typeof module !== 'undefined') module.exports = { roiEstimate };

