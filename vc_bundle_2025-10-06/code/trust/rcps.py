#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RCPS utilities: one-sided Clopper–Pearson selection and multi-stage gating.

Notes
- This module is agnostic to model type. Provide per-stage probability arrays and
  a calibration set to choose an acceptance threshold tau_acc that controls the
  accepted-subset error via a one-sided CP upper bound.
"""

from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np
from scipy.stats import beta


def cp_upper_one_sided(num_errors: int, n: int, delta: float) -> float:
    """One-sided Clopper–Pearson upper bound for a binomial proportion.

    Returns u s.t. P(Bin(n, u) <= num_errors) >= 1 - delta. If n=0, returns 0.
    """
    if n <= 0:
        return 0.0
    return float(beta.ppf(1.0 - delta, num_errors + 1, n - num_errors))


def singleton_accept_mask(proba: np.ndarray, tau_acc: float) -> np.ndarray:
    """Singleton acceptance: exactly one class with probability >= tau_acc."""
    return (proba >= tau_acc).sum(axis=1) == 1


def pipeline_accept_stage(probas_list: List[np.ndarray], tau_acc: float) -> np.ndarray:
    """For a given tau_acc, return the stage index each example is accepted at, or -1 if none.

    probas_list: list of arrays, each shape (n_samples, n_classes)
    """
    n = probas_list[0].shape[0]
    accepted_at = np.full(n, -1, dtype=int)
    remaining = np.arange(n)
    for j, proba in enumerate(probas_list):
        if remaining.size == 0:
            break
        mask = singleton_accept_mask(proba[remaining], tau_acc)
        acc_idxs = remaining[mask]
        accepted_at[acc_idxs] = j
        remaining = remaining[~mask]
    return accepted_at


def choose_tau_rcps(
    y_cal: np.ndarray,
    probas_cal: List[np.ndarray],
    alpha: float = 0.05,
    delta: float = 0.05,
    tau_grid: np.ndarray | None = None,
) -> Dict[str, float | int]:
    """Choose tau_acc to maximize calibration acceptance while keeping CP upper bound <= alpha.

    Returns dict with keys: tau, accepted (fraction), n_acc, err_cp_upper, num_err.
    """
    if tau_grid is None:
        tau_grid = np.linspace(0.5, 0.999, 80)

    best = {"tau": None, "accepted": 0.0, "n_acc": 0, "err_cp_upper": 1.0, "num_err": 0}
    n_total = len(y_cal)
    for tau in tau_grid:
        accepted_at = pipeline_accept_stage(probas_cal, tau)
        idx_acc = np.where(accepted_at >= 0)[0]
        n_acc = len(idx_acc)
        if n_acc == 0:
            ub = 0.0
            num_err = 0
        else:
            preds = []
            for i in idx_acc:
                j = accepted_at[i]
                preds.append(int(np.argmax(probas_cal[j][i])))
            preds = np.array(preds, dtype=int)
            num_err = int((preds != y_cal[idx_acc]).sum())
            ub = cp_upper_one_sided(num_err, n_acc, delta)
        if ub <= alpha:
            if n_acc > best["n_acc"] or (n_acc == best["n_acc"] and (best["tau"] is None or tau < best["tau"])):
                best = {
                    "tau": float(tau),
                    "accepted": float(n_acc / n_total),
                    "n_acc": int(n_acc),
                    "err_cp_upper": float(ub),
                    "num_err": int(num_err),
                }

    if best["tau"] is None:
        # No feasible tau found: fall back to abstain-all (valid but trivial)
        best = {"tau": 1.0, "accepted": 0.0, "n_acc": 0, "err_cp_upper": 0.0, "num_err": 0}
    return best


def run_pipeline_test(
    probas_list_test: List[np.ndarray],
    tau_acc: float,
    Xs_te_full: np.ndarray,
    y_test: np.ndarray,
    fallback_model,
    k_list: List[int],
) -> Tuple[np.ndarray, np.ndarray, float, float, float, Dict[str, float]]:
    """Apply RCPS-MFA gating on test, fallback to heavy model if unaccepted."""
    accepted_at = pipeline_accept_stage(probas_list_test, tau_acc)
    n = len(y_test)
    final_pred = np.empty(n, dtype=int)

    idx_acc = np.where(accepted_at >= 0)[0]
    for i in idx_acc:
        j = accepted_at[i]
        final_pred[i] = int(np.argmax(probas_list_test[j][i]))

    idx_fb = np.where(accepted_at < 0)[0]
    if idx_fb.size > 0:
        final_pred[idx_fb] = fallback_model.predict(Xs_te_full[idx_fb])

    if idx_acc.size > 0:
        test_err_rate_accepted = float((final_pred[idx_acc] != y_test[idx_acc]).mean())
    else:
        test_err_rate_accepted = 0.0

    stage_feat = np.array(k_list + [Xs_te_full.shape[1]])
    feat_per_sample = stage_feat[np.where(accepted_at >= 0, accepted_at, len(k_list))]
    avg_features = float(feat_per_sample.mean())
    feature_savings = 1.0 - avg_features / Xs_te_full.shape[1]

    shares = {}
    for s, k in enumerate(k_list):
        shares[f"stage{s+1}_k={k}"] = float((accepted_at == s).mean())
    shares["fallback_svm"] = float((accepted_at < 0).mean())

    return final_pred, accepted_at, test_err_rate_accepted, avg_features, feature_savings, shares


def time_pipeline(
    probas_list_test: List[np.ndarray],
    tau_acc: float,
    fallback_model,
    Xs_te_full: np.ndarray,
    k_list: List[int],
    n_repeat: int = 7,
) -> float:
    ts = []
    n = Xs_te_full.shape[0]
    for _ in range(n_repeat):
        import time
        t0 = time.perf_counter()
        accepted_at = pipeline_accept_stage(probas_list_test, tau_acc)
        idx_fb = np.where(accepted_at < 0)[0]
        if idx_fb.size > 0:
            _ = fallback_model.predict(Xs_te_full[idx_fb])
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts) / n)


def time_model(model, X: np.ndarray, n_repeat: int = 7) -> float:
    ts = []
    n = X.shape[0]
    for _ in range(n_repeat):
        import time
        t0 = time.perf_counter()
        _ = model.predict(X)
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts) / n)

