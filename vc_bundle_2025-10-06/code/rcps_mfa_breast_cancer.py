#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RCPS-MFA: Risk-Controlling Prediction Sets with Multi-Stage Feature Acquisition

Dataset: UCI Breast Cancer (via scikit-learn)
Guarantee: with prob >= (1 - δ), the true error among accepted predictions is <= α.

Stages (KAN-∞ style front-end):
- Rank features (L1-logistic if sufficiently sparse; else mutual information)
- Train tiny logistic models on top-k = {3, 6, 10}
- RCPS gate selects a global acceptance threshold tau_acc on calibration
- Accept when a stage yields a singleton set (one class prob >= tau_acc); else escalate
- If no stage accepts, fallback to RBF-SVM (strong, heavier)

Usage
- python rcps_mfa_breast_cancer.py
- Optional flags: --alpha 0.05 --delta 0.05 --no-plot

Dependencies: numpy pandas scikit-learn matplotlib scipy
"""

from __future__ import annotations

import argparse
import math
import time
import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import mutual_info_classif

import matplotlib.pyplot as plt

from trust.rcps import (
    cp_upper_one_sided,
    choose_tau_rcps,
    pipeline_accept_stage,
    run_pipeline_test,
    time_model,
    time_pipeline,
)


warnings.filterwarnings("ignore")


def standardize(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
    scaler = StandardScaler().fit(train_df.values)
    return (
        scaler.transform(train_df.values),
        scaler.transform(val_df.values),
        scaler.transform(test_df.values),
    )


def rank_features_l1_or_mi(Xs_tr, y_tr, min_k=10, random_state=0):
    """Rank features using L1 logistic if it yields >= min_k nonzeros; otherwise MI."""
    l1 = LogisticRegression(
        penalty="l1", solver="liblinear", C=0.2, max_iter=5000, random_state=random_state
    )
    l1.fit(Xs_tr, y_tr)
    coef = np.abs(l1.coef_.ravel())
    if (coef > 1e-8).sum() >= min_k:
        return np.argsort(-coef)
    mi = mutual_info_classif(Xs_tr, y_tr, random_state=random_state)
    return np.argsort(-mi)


def fit_stage_models(Xs_tr, y_tr, ranks, k_list: List[int]):
    """Fit a logistic regression on the top-k features for each stage."""
    models = []
    idxs = []
    for k in k_list:
        idx = ranks[:k]
        lr = LogisticRegression(penalty="none", solver="lbfgs", max_iter=5000, random_state=0)
        lr.fit(Xs_tr[:, idx], y_tr)
        models.append(lr)
        idxs.append(idx)
    return models, idxs


def compute_stage_probas(models, idxs, Xs):
    return [lr.predict_proba(Xs[:, idx]) for lr, idx in zip(models, idxs)]


def min_n_for_zero_errors(alpha: float, delta: float) -> int:
    # Need BetaInv(1-δ; 1, n) <= α  → (1-δ)^(1/n) >= 1 - α  → n >= ln(δ)/ln(1-α)
    return math.ceil(math.log(delta) / math.log(1.0 - alpha))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha-strict", type=float, default=0.01)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    # ----------------------------- Data -----------------------------
    data = load_breast_cancer()
    X_df = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target.astype(int)

    # 60/20/20 split: train / calibration / test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_df, y, test_size=0.20, stratify=y, random_state=42
    )
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_trainval, y_trainval, test_size=0.25, stratify=y_trainval, random_state=42
    )

    Xs_tr, Xs_cal, Xs_te = standardize(X_train, X_cal, X_test)

    # ----------------------------- KAN-∞ front-end -----------------------------
    ranks = rank_features_l1_or_mi(Xs_tr, y_train, min_k=10, random_state=42)
    k_list = [3, 6, 10]
    stage_models, idxs = fit_stage_models(Xs_tr, y_train, ranks, k_list)

    # Heavy fallback: RBF-SVM
    svm = SVC(C=10, gamma="scale", random_state=42)
    svm.fit(Xs_tr, y_train)

    # Probabilities on calibration & test
    probas_cal = compute_stage_probas(stage_models, idxs, Xs_cal)
    probas_test = compute_stage_probas(stage_models, idxs, Xs_te)

    # ----------------------------- STRICT setting (α=0.01, δ=0.05) -----------------------------
    alpha_strict = float(args.alpha_strict)
    delta = float(args.delta)
    best_strict = choose_tau_rcps(
        y_cal, probas_cal, alpha=alpha_strict, delta=delta, tau_grid=np.linspace(0.5, 0.999, 120)
    )

    print("\nRCPS-MFA — STRICT setting (α=%.2g, δ=%.2g)" % (alpha_strict, delta))
    print(
        f"Calibration τ* = {best_strict['tau']:.4f}; acceptance = {best_strict['accepted']:.3f}; CP upper = {best_strict['err_cp_upper']:.4f}"
    )
    print(
        f"With α={alpha_strict:.2g} & δ={delta:.2g}, you need ~{min_n_for_zero_errors(alpha_strict, delta)} accepted calibration points (with 0 errors) to certify —"
    )
    print(
        f"we only have {len(y_cal)} calibration samples total, so τ*=1.0 (accept none) can be the only certificate-respecting option with this dataset size."
    )

    # ----------------------------- PRACTICAL setting (α=0.05, δ=0.05) -----------------------------
    alpha = float(args.alpha)
    best = choose_tau_rcps(
        y_cal, probas_cal, alpha=alpha, delta=delta, tau_grid=np.linspace(0.5, 0.999, 120)
    )
    tau_star = best["tau"]

    final_pred, accepted_at, test_err_rate_accepted, avg_features, feature_savings, shares = run_pipeline_test(
        probas_test, tau_star, Xs_te, y_test, svm, k_list
    )
    acc = accuracy_score(y_test, final_pred)

    svm_acc = accuracy_score(y_test, svm.predict(Xs_te))
    svm_time = time_model(svm, Xs_te)
    pipe_time = time_pipeline(probas_test, tau_star, svm, Xs_te, k_list)
    speedup = svm_time / pipe_time if pipe_time > 0 else float("inf")

    print("\nRCPS-MFA — PRACTICAL setting (α=%.2g, δ=%.2g)" % (alpha, delta))
    print(
        f"Calibration τ* = {tau_star:.4f}; acceptance = {best['accepted']:.3f}; CP upper = {best['err_cp_upper']:.4f}  (guaranteed ≤ α)"
    )
    print(f"Test accuracy: RCPS-MFA {acc:.4f}  vs  SVM {svm_acc:.4f}")
    print(f"Accepted-subset test error (informational): {test_err_rate_accepted:.4f}")
    print(
        f"Avg features/sample: {avg_features:.2f} / {X_df.shape[1]}  (savings {feature_savings*100:.1f}%)"
    )
    print(
        f"Latency: SVM {svm_time:.6e} s/sample  |  RCPS-MFA {pipe_time:.6e} s/sample  |  speedup {speedup:.2f}×"
    )
    print("Stage shares:", shares)

    # ----------------------------- Plots -----------------------------
    if not args.no-plot:
        plt.figure()
        plt.bar([0, 1], [svm_acc, acc])
        plt.xticks([0, 1], ["SVM (baseline)", "RCPS-MFA (α=0.05)"])
        plt.ylabel("Test accuracy")
        plt.title("Accuracy — Baseline vs RCPS-MFA")
        plt.tight_layout()
        plt.show()

        plt.figure()
        plt.bar([0, 1], [svm_time, pipe_time])
        plt.xticks([0, 1], ["SVM (baseline)", "RCPS-MFA (α=0.05)"])
        plt.ylabel("Seconds per sample (lower is better)")
        plt.title("Latency — Baseline vs RCPS-MFA")
        plt.tight_layout()
        plt.show()

        plt.figure()
        plt.bar([0], [feature_savings])
        plt.ylim(0, 1)
        plt.xticks([0], ["Feature savings fraction"])
        plt.title("Average fraction of features NOT measured")
        plt.tight_layout()
        plt.show()

        taus = np.linspace(0.5, 0.999, 120)
        acc_frac = []
        cp_ub = []
        for t in taus:
            accepted_at_cal = pipeline_accept_stage(probas_cal, t)
            idx_acc = np.where(accepted_at_cal >= 0)[0]
            n_acc = len(idx_acc)
            if n_acc == 0:
                cp_ub.append(0.0)
                acc_frac.append(0.0)
                continue
            preds = []
            for i in idx_acc:
                j = accepted_at_cal[i]
                preds.append(int(np.argmax(probas_cal[j][i])))
            preds = np.array(preds, dtype=int)
            num_err = int((preds != y_cal[idx_acc]).sum())
            cp_ub.append(cp_upper_one_sided(num_err, n_acc, delta))
            acc_frac.append(n_acc / len(y_cal))

        plt.figure()
        plt.plot(taus, acc_frac, label="Acceptance fraction (calibration)")
        plt.plot(taus, cp_ub, label="CP upper bound (accepted-subset error)")
        plt.axhline(alpha, linestyle="--")
        plt.axvline(tau_star, linestyle="--")
        plt.xlabel("τ (singleton threshold)")
        plt.ylabel("Acceptance / Error bound")
        plt.title("RCPS calibration sweep")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # ----------------------------- Save CSV summaries -----------------------------
    pd.DataFrame(
        [
            {
                "alpha_risk": alpha_strict,
                "delta_confidence": delta,
                "tau_star": best_strict["tau"],
                "calib_accept_fraction": best_strict["accepted"],
                "calib_cp_upper": best_strict["err_cp_upper"],
                "note": "STRICT: expect τ*=1.0 given small calibration size",
            }
        ]
    ).to_csv("./rcps_mfa_results_strict.csv", index=False)

    pd.DataFrame(
        [
            {
                "alpha_risk": alpha,
                "delta_confidence": delta,
                "tau_star": tau_star,
                "calib_accept_fraction": best["accepted"],
                "calib_cp_upper": best["err_cp_upper"],
                "test_accuracy_rcps_mfa": acc,
                "test_accuracy_svm": svm_acc,
                "test_accepted_subset_error": test_err_rate_accepted,
                "avg_features_per_sample": avg_features,
                "feature_savings_fraction": feature_savings,
                "sec_per_sample_svm": svm_time,
                "sec_per_sample_rcps_mfa": pipe_time,
                "speedup_x": speedup,
                **shares,
            }
        ]
    ).to_csv("./rcps_mfa_results_practical.csv", index=False)

    print("\nSaved CSVs:")
    print("  ./rcps_mfa_results_strict.csv")
    print("  ./rcps_mfa_results_practical.csv")


if __name__ == "__main__":
    main()

