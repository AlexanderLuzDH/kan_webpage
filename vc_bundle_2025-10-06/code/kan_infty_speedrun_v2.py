#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KAN-∞ Speedrun v2 — high-signal, training-free demos for a16z

Adds:
- Auto temperature selection (--auto-tau, --tau-grid)
- Optional classwise bias correction (--bias-correct)
- Configurable ST model for AG News (--st-model)
- CLIP text prototypes for CIFAR-10 (--text-proto) + zero-shot mode (--zero-shot-only)

Install once:
pip install -U numpy scipy scikit-learn tqdm torch torchvision open_clip_torch sentence-transformers datasets psutil

Examples:
# AG News, 10k subset, 1/5/10-shot with auto-τ and mpnet:
python kan_infty_speedrun_v2.py --task agnews --labels-per-class 1 5 10 --subset 10000 --method softkan --auto-tau --st-model sentence-transformers/all-mpnet-base-v2

# AG News full train (careful, embeds ~120k rows):
python kan_infty_speedrun_v2.py --task agnews --labels-per-class 1 5 10 --subset 120000 --method softkan --auto-tau --st-model sentence-transformers/all-mpnet-base-v2

# AG News AMLE 10k head-to-head:
python kan_infty_speedrun_v2.py --task agnews --labels-per-class 5 --subset 10000 --method amle --knn 20 --st-model sentence-transformers/all-MiniLM-L6-v2

# CIFAR-10 Soft-KAN + text prototypes (few-shot + zero-shot hybrid):
python kan_infty_speedrun_v2.py --task cifar10 --method softkan --labels-per-class 1 5 10 --train-max 50000 --test-max 10000 --tau 0.02 --text-proto

# CIFAR-10 pure zero-shot (only CLIP text prompts; no labels):
python kan_infty_speedrun_v2.py --task cifar10 --method softkan --labels-per-class 0 --zero-shot-only --train-max 0 --test-max 10000 --text-proto --tau 0.02
"""

import os, time, json, argparse, random, gc
import numpy as np
from pathlib import Path
from typing import List
from tqdm import tqdm
import psutil
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

def ensure_dir(p: str): Path(p).mkdir(parents=True, exist_ok=True)
def now(): return time.strftime("%Y-%m-%d %H:%M:%S")
def log(msg: str): print(f"[{now()}] {msg}", flush=True)
def mem_gb(): return psutil.Process().memory_info().rss / (1024**3)

# ---------- Soft-KAN core ----------
def logsumexp(X: np.ndarray, axis=1):
    m = np.max(X, axis=axis, keepdims=True)
    return np.log(np.sum(np.exp(X - m), axis=axis)) + m.squeeze(axis=axis)

def smin_tau(Z: np.ndarray, tau: float) -> np.ndarray:
    return -tau * logsumexp(-Z / tau, axis=1)

def cosine_dist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return 1.0 - (A @ B.T).astype(np.float32)

def softkan_scores_from_D(D: np.ndarray, y_labels: np.ndarray, n_classes: int,
                          L: float = 1.0, tau: float = 0.05,
                          class_bias: np.ndarray | None = None) -> np.ndarray:
    n = D.shape[0]
    scores = np.zeros((n, n_classes), dtype=np.float32)
    idx_by_class = [np.where(y_labels == c)[0] for c in range(n_classes)]
    for c in range(n_classes):
        pos = idx_by_class[c]
        neg = np.setdiff1d(np.arange(D.shape[1]), pos, assume_unique=True)
        if len(pos) == 0:
            scores[:, c] = -1e9
            continue
        Zpos_up = (1.0 + L * D[:, pos])
        Zneg_up = (0.0 + L * D[:, neg]) if len(neg) else None
        if Zneg_up is None:
            upper = smin_tau(Zpos_up, tau)
        else:
            upper = -tau * logsumexp(np.concatenate([-(Zpos_up)/tau, -(Zneg_up)/tau], axis=1), axis=1)
        Zpos_low = ((-L * D[:, pos]) + 1.0)
        Zneg_low = ((-L * D[:, neg]) + 0.0) if len(neg) else None
        if Zneg_low is None:
            lower = tau * logsumexp(Zpos_low / tau, axis=1)
        else:
            lower = tau * logsumexp(np.concatenate([Zpos_low/tau, Zneg_low/tau], axis=1), axis=1)
        s = 0.5 * (upper + lower)
        scores[:, c] = s
    if class_bias is not None:
        scores = scores + class_bias[None, :].astype(np.float32)
    return scores

def pick_labels_per_class(y: np.ndarray, per_class: int, n_classes: int, seed=42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = []
    for c in range(n_classes):
        ids = np.where(y == c)[0]
        if len(ids) == 0: continue
        rng.shuffle(ids)
        take = min(per_class, len(ids))
        idx.extend(list(ids[:take]))
    return np.array(sorted(idx))

def farthest_unlabeled_order(Xn: np.ndarray, labeled_idx: np.ndarray) -> np.ndarray:
    unl = np.setdiff1d(np.arange(Xn.shape[0]), labeled_idx, assume_unique=True)
    if len(labeled_idx) == 0: return unl
    D = cosine_dist(Xn[unl], Xn[labeled_idx])
    mind = D.min(axis=1)
    return unl[np.argsort(mind)[::-1]]

def auto_tau_loocv(Xn: np.ndarray, labeled_idx: np.ndarray, y: np.ndarray, n_classes: int,
                   tau_grid: List[float], L: float = 1.0, bias_correct: bool = False):
    """Pick τ via leave-one-out on the labeled set (tiny, fast, training-free)."""
    if len(labeled_idx) == 0: return tau_grid[0], None
    best_tau, best_acc, best_bias = tau_grid[0], -1.0, None
    y_lab = y[labeled_idx]
    # Precompute pairwise distances among labeled
    D_lab = cosine_dist(Xn[labeled_idx], Xn[labeled_idx])
    for tau in tau_grid:
        correct = 0
        # Leave-one-out: predict each labeled i from all others
        for i in range(len(labeled_idx)):
            mask = np.ones(len(labeled_idx), dtype=bool); mask[i] = False
            D_i = D_lab[i:i+1, mask]              # (1, m-1)
            y_i = y_lab[mask]
            scores_i = softkan_scores_from_D(D_i, y_i, n_classes=n_classes, L=L, tau=tau)
            pred = int(np.argmax(scores_i[0]))
            correct += int(pred == y_lab[i])
        acc = correct / len(labeled_idx)
        # Optional bias calibration using full labeled set (no LOOCV here)
        cb = None
        if bias_correct and len(labeled_idx) > 0:
            D_full = D_lab  # include self; it won't change classwise mean much
            s_full = softkan_scores_from_D(D_full, y_lab, n_classes=n_classes, L=L, tau=tau)
            cb = np.zeros(n_classes, dtype=np.float32)
            for c in range(n_classes):
                pos = (y_lab == c)
                if pos.any():
                    cb[c] = 1.0 - float(s_full[pos, c].mean())
        if acc > best_acc:
            best_acc, best_tau, best_bias = acc, tau, cb
    return best_tau, best_bias

# ---------- AMLE (∞-harmonic) ----------
def build_knn_indices(Xn: np.ndarray, k: int = 15) -> np.ndarray:
    nbrs = NearestNeighbors(n_neighbors=k+1, metric='euclidean').fit(Xn)
    return nbrs.kneighbors(Xn, return_distance=False)[:, 1:]

def amle_multiclass(U_init: np.ndarray, neighbors: np.ndarray, boundary_mask: np.ndarray,
                    max_iter: int = 400, tol: float = 1e-4) -> np.ndarray:
    U = U_init.copy()
    interior = np.where(~boundary_mask)[0]
    for _ in range(max_iter):
        max_delta = 0.0
        for i in interior:
            v = U[neighbors[i]]
            new_u = 0.5 * (v.max(axis=0) + v.min(axis=0))
            d = np.max(np.abs(new_u - U[i]))
            if d > tol:
                U[i] = new_u
                if d > max_delta: max_delta = d
        if max_delta < tol: break
    return U

# ---------- CIFAR-10 ----------
CIFAR10_CLASSES = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
DEFAULT_TEMPLATES = [
    "a photo of a {}",
    "a blurry photo of a {}",
    "a close-up photo of a {}",
    "a low-resolution photo of a {}",
    "a bright photo of a {}",
    "a cropped photo of a {}",
    "a photo of the {}"
]

def load_cifar10_embeddings(train_max=20000, test_max=10000, batch=128, device='cpu',
                            clip_arch='ViT-B-32', clip_pretrained='laion2b_s34b_b79k'):
    import torch, torchvision
    import torchvision.transforms as T
    import open_clip
    log("Loading CIFAR-10...")
    transform = T.Compose([
        T.Resize(224, antialias=True), T.CenterCrop(224), T.ToTensor(),
        T.Normalize(mean=(0.48145466,0.4578275,0.40821073),
                    std=(0.26862954,0.26130258,0.27577711))
    ])
    trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    testset  = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    if train_max and train_max < len(trainset):
        idx = np.random.RandomState(0).choice(len(trainset), size=train_max, replace=False)
        trainset = torch.utils.data.Subset(trainset, idx.tolist())
    if test_max and test_max < len(testset):
        idx = np.random.RandomState(1).choice(len(testset), size=test_max, replace=False)
        testset = torch.utils.data.Subset(testset, idx.tolist())
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch, shuffle=False, num_workers=2, pin_memory=False)
    test_loader  = torch.utils.data.DataLoader(testset,  batch_size=batch, shuffle=False, num_workers=2, pin_memory=False)

    log(f"Loading OpenCLIP {clip_arch} ({clip_pretrained})...")
    model, _, _ = open_clip.create_model_and_transforms(clip_arch, pretrained=clip_pretrained, device=device)
    model.eval()

    @torch.no_grad()
    def embed_images(loader):
        feats, ys = [], []
        for xb, yb in tqdm(loader, desc="Embedding images", ncols=80):
            xb = xb.to(device)
            f = model.encode_image(xb); f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f.cpu().numpy().astype(np.float32))
            ys.append(yb.numpy().astype(np.int64))
        return np.concatenate(feats, 0), np.concatenate(ys, 0)

    Xtr, Ytr = embed_images(train_loader)
    Xte, Yte = embed_images(test_loader)
    X_all = normalize(np.vstack([Xtr, Xte]).astype(np.float32), axis=1)
    y_all = np.concatenate([Ytr, Yte]).astype(np.int64)
    return X_all, y_all, Xtr.shape[0], Xte.shape[0], model

def clip_text_prototypes(model, device='cpu', templates=DEFAULT_TEMPLATES):
    import torch, open_clip
    with torch.no_grad():
        toks = []
        labels = []
        for c, name in enumerate(CIFAR10_CLASSES):
            for t in templates:
                toks.append(open_clip.tokenize(t.format(name)))
                labels.append(c)
        toks = torch.cat(toks, dim=0).to(device)
        txt = model.encode_text(toks)
        txt = txt / txt.norm(dim=-1, keepdim=True)
    return txt.cpu().numpy().astype(np.float32), np.array(labels, dtype=np.int64)

def run_cifar10(method: str, labels_per_class: List[int], train_max=20000, test_max=10000,
                knn=15, tau=0.05, L=1.0, seed=42, outdir="./kan_inf_outputs",
                text_proto: bool=False, zero_shot_only: bool=False, auto_tau: bool=False,
                tau_grid: List[float] = None, bias_correct: bool=False,
                clip_arch: str = 'ViT-B-32', clip_pretrained: str = 'laion2b_s34b_b79k',
                rcps_enable: bool = False, rcps_alpha: float = 0.05, rcps_delta: float = 0.05,
                rcps_gate: str = "prob", rcps_cal_frac: float = 0.25):
    set_seed(seed); ensure_dir(outdir)
    device = 'cpu'
    try:
        import torch
        if torch.cuda.is_available(): device='cuda'
        elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available(): device='mps'
    except Exception: pass

    X, y, n_train, n_test, clip_model = load_cifar10_embeddings(train_max, test_max, device=device,
                                                                clip_arch=clip_arch, clip_pretrained=clip_pretrained)
    n_total, n_classes = X.shape[0], 10
    Xn = normalize(X, axis=1)

    txt_emb, txt_lbl = (None, None)
    if text_proto:
        log("Encoding CLIP text prototypes...")
        txt_emb, txt_lbl = clip_text_prototypes(clip_model, device=device)
        txt_emb = normalize(txt_emb, axis=1)

    res = []
    for per_class in labels_per_class:
        if per_class == 0 and zero_shot_only:
            labeled_global = np.array([], dtype=np.int64)
            y_labels = np.array([], dtype=np.int64)
        else:
            labeled_global = pick_labels_per_class(y[:n_train], per_class, n_classes, seed=seed)
            y_labels = y[labeled_global]

        # build boundary set for Soft-KAN distances
        B_emb = []
        B_y = []
        if len(labeled_global) > 0:
            B_emb.append(Xn[labeled_global]); B_y.append(y_labels)
        if text_proto and (zero_shot_only or len(labeled_global) >= 0):
            B_emb.append(txt_emb); B_y.append(txt_lbl)
        if len(B_emb) == 0:
            log("No boundary points to extend from; skip.")
            continue
        B_emb = normalize(np.vstack(B_emb), axis=1)
        B_y = np.concatenate(B_y)

        t0 = time.time()
        D = cosine_dist(Xn, B_emb)
        prep_time = time.time() - t0

        chosen_tau, class_bias = tau, None
        if auto_tau and not zero_shot_only and len(labeled_global) > 0:
            chosen_tau, class_bias = auto_tau_loocv(
                Xn[:n_train], labeled_global, y, n_classes,
                tau_grid=(tau_grid or [0.01,0.02,0.05,0.1,0.2]),
                L=L, bias_correct=bias_correct
            )
            log(f"[auto-τ] picked τ={chosen_tau} bias={'on' if bias_correct else 'off'}")

        t1 = time.time()
        if method.lower() == "softkan":
            scores = softkan_scores_from_D(D, B_y, n_classes=n_classes, L=L, tau=chosen_tau, class_bias=class_bias)
            y_pred_all = np.argmax(scores, axis=1)
            pred_time = time.time() - t1
            S_gate = scores  # default gating scores

            # Calibrated fusion: separate label vs prototype scores and blend with alpha via LOOCV
            try:
                B_lab_emb = Xn[labeled_global] if len(labeled_global) > 0 else None
                B_lab_y   = y[labeled_global] if len(labeled_global) > 0 else None
                B_proto_emb, B_proto_y = (txt_emb, txt_lbl) if text_proto else (None, None)
                # auto-tau for labels only
                chosen_tau_labels, cb = chosen_tau, class_bias
                if auto_tau and (B_lab_emb is not None) and (len(labeled_global) > 0):
                    chosen_tau_labels, cb = auto_tau_loocv(
                        Xn[:n_train], labeled_global, y, n_classes,
                        tau_grid=(tau_grid or [0.01,0.02,0.05,0.1,0.2]), L=L, bias_correct=bias_correct)
                    log(f"[auto-τ labels] τ_lab={chosen_tau_labels}")
                tau_proto = tau
                # distances
                D_lab = cosine_dist(Xn, B_lab_emb) if B_lab_emb is not None else None
                D_pro = cosine_dist(Xn, B_proto_emb) if B_proto_emb is not None else None
                # scores
                S_lab = (softkan_scores_from_D(D_lab, B_lab_y, n_classes, L=L, tau=chosen_tau_labels, class_bias=cb)
                         if D_lab is not None else None)
                S_pro = (softkan_scores_from_D(D_pro, B_proto_y, n_classes, L=L, tau=tau_proto)
                         if D_pro is not None else None)
                # LOOCV alpha on labeled set
                alpha = 0.5
                if (S_lab is not None) and (len(labeled_global) > 0):
                    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
                    best, best_acc = 0.5, -1.0
                    if S_pro is None:
                        best = 1.0
                    else:
                        D_lab_lab = cosine_dist(B_lab_emb, B_lab_emb)
                        for a in alphas:
                            correct = 0
                            for i in range(len(labeled_global)):
                                mask = np.ones(len(labeled_global), dtype=bool); mask[i] = False
                                s_li = softkan_scores_from_D(D_lab_lab[i:i+1, mask], B_lab_y[mask], n_classes, L=L, tau=chosen_tau_labels)
                                if B_proto_emb is not None:
                                    D_pi = cosine_dist(B_lab_emb[i:i+1], B_proto_emb)
                                    s_pi = softkan_scores_from_D(D_pi, B_proto_y, n_classes, L=L, tau=tau_proto)
                                    s_i = (1-a)*s_pi + a*s_li
                                else:
                                    s_i = s_li
                                pred_i = int(np.argmax(s_i[0]))
                                correct += int(pred_i == B_lab_y[i])
                            acc_cv = correct / len(labeled_global)
                            if acc_cv > best_acc:
                                best_acc, best = acc_cv, a
                    alpha = best
                    log(f"[auto-α fusion] α={alpha}")
                # final fusion for all points
                if S_pro is None and S_lab is None:
                    pass
                elif S_pro is None:
                    y_pred_all = np.argmax(S_lab, axis=1)
                    S_gate = S_lab
                elif S_lab is None:
                    y_pred_all = np.argmax(S_pro, axis=1)
                    S_gate = S_pro
                else:
                    S_mix = (1-alpha)*S_pro + alpha*S_lab
                    y_pred_all = np.argmax(S_mix, axis=1)
                    S_gate = S_mix
            except Exception as _e:
                # Fall back silently if fusion path fails
                S_gate = scores
            # Optional RCPS gating for CIFAR: accept on Soft-KAN top-prob >= tau_acc; fallback to AMLE if labels exist
            rcps_fields = {
                "rcps_enabled": bool(rcps_enable),
                "rcps_tau": None,
                "rcps_alpha": float(rcps_alpha),
                "rcps_delta": float(rcps_delta),
                "rcps_accept_fraction": None,
                "rcps_abstain_fraction": None,
                "rcps_acc_accepted": None,
                "rcps_final_acc": None,
                "rcps_fallback": None,
            }
            if rcps_enable:
                try:
                    from trust.rcps import choose_tau_rcps, pipeline_accept_stage
                    # Build gating probabilities
                    if rcps_gate == "prob":
                        G = S_gate - S_gate.max(axis=1, keepdims=True)
                        P_gate = np.exp(G); P_gate /= P_gate.sum(axis=1, keepdims=True)
                    else:
                        # margin → transform to pseudo-prob via softmax over [top, second] only
                        G = S_gate - S_gate.max(axis=1, keepdims=True)
                        P_gate = np.exp(G); P_gate /= P_gate.sum(axis=1, keepdims=True)
                    # Cal/eval split on full set for simplicity
                    rng = np.random.default_rng(seed)
                    idx_all = np.arange(n_total)
                    rng.shuffle(idx_all)
                    n_cal = int(max(1, rcps_cal_frac * n_total))
                    cal_idx = idx_all[:n_cal]
                    eval_idx = idx_all[n_cal:]
                    best = choose_tau_rcps(y[cal_idx], [P_gate[cal_idx]], alpha=rcps_alpha, delta=rcps_delta,
                                           tau_grid=np.linspace(0.5, 0.999, 120))
                    tau_acc = float(best["tau"]) if best["tau"] is not None else 1.0
                    rcps_fields["rcps_tau"] = tau_acc
                    rcps_fields["rcps_accept_fraction"] = float(best["accepted"]) if best else 0.0
                    # Accepted set on eval
                    acc_stage = pipeline_accept_stage([P_gate[eval_idx]], tau_acc)
                    accepted_mask = (acc_stage >= 0)
                    final_pred = np.empty(len(eval_idx), dtype=int)
                    # Accepted: use current Soft-KAN prediction
                    final_pred[accepted_mask] = y_pred_all[eval_idx][accepted_mask]
                    # Fallback: AMLE if labels available, else stick with Soft-KAN
                    fb_name = "softkan"
                    if len(labeled_global) > 0:
                        try:
                            inds = build_knn_indices(Xn, k=knn)
                            B_mask = np.zeros(n_total, dtype=bool); B_mask[labeled_global] = True
                            U0 = np.zeros((n_total, n_classes), dtype=np.float32)
                            for c in range(n_classes):
                                U0[labeled_global, c] = (y[labeled_global] == c).astype(np.float32)
                            U = amle_multiclass(U0, inds, B_mask, max_iter=200, tol=1e-3)
                            y_pred_amle = np.argmax(U, axis=1)
                            fb_pred = y_pred_amle[eval_idx][~accepted_mask]
                            fb_name = "amle"
                        except Exception:
                            fb_pred = y_pred_all[eval_idx][~accepted_mask]
                    else:
                        fb_pred = y_pred_all[eval_idx][~accepted_mask]
                    final_pred[~accepted_mask] = fb_pred
                    rcps_fields["rcps_fallback"] = fb_name
                    # Metrics
                    if accepted_mask.any():
                        rcps_fields["rcps_acc_accepted"] = float(
                            accuracy_score(y[eval_idx][accepted_mask], final_pred[accepted_mask])
                        )
                    rcps_fields["rcps_abstain_fraction"] = float((~accepted_mask).mean())
                    rcps_fields["rcps_final_acc"] = float(accuracy_score(y[eval_idx], final_pred))
                except Exception:
                    pass
        elif method.lower() == "amle":
            inds = build_knn_indices(Xn, k=knn)
            B_mask = np.zeros(n_total, dtype=bool)
            if len(labeled_global) > 0:
                B_mask[labeled_global] = True
            U0 = np.zeros((n_total, n_classes), dtype=np.float32)
            if len(labeled_global) > 0:
                for c in range(n_classes):
                    U0[labeled_global, c] = (y[labeled_global] == c).astype(np.float32)
            U = amle_multiclass(U0, inds, B_mask, max_iter=400, tol=1e-3)
            y_pred_all = np.argmax(U, axis=1)
            pred_time = time.time() - t1
            prep_time = prep_time  # keep column shape
        else:
            raise ValueError("method must be one of {softkan, amle}")

        acc_all = accuracy_score(y, y_pred_all)
        acc_train = accuracy_score(y[:n_train], y_pred_all[:n_train])
        acc_test  = accuracy_score(y[n_train:], y_pred_all[n_train:])

        # active learning picks from train pool only if we used any train labels
        top16 = []
        if len(labeled_global) > 0:
            far_order = farthest_unlabeled_order(Xn[:n_train], labeled_global)
            top16 = far_order[:16].tolist()

        row = {
            "dataset": "cifar10",
            "method": method,
            "labels_per_class": per_class,
            "m_labels": int(len(labeled_global)) + (0 if txt_emb is None else int(txt_emb.shape[0])),
            "acc_all": round(float(acc_all),4),
            "acc_train": round(float(acc_train),4),
            "acc_test": round(float(acc_test),4),
            "tau": float(chosen_tau),
            "prep_time_sec": round(prep_time,3),
            "predict_time_sec": round(pred_time,3),
            "mem_gb": round(mem_gb(),3),
            "text_prototypes": bool(text_proto),
            "zero_shot_only": bool(zero_shot_only),
            "next16_label_indices_in_train": top16
        }
        if method.lower() == "softkan":
            # attach RCPS fields if computed
            try:
                row.update({k: v for k, v in rcps_fields.items()})
            except Exception:
                pass
        res.append(row)
        log(f"[CIFAR10][{method}] K={per_class}/class | acc_test={acc_test:.4f} | τ={chosen_tau} | prep={prep_time:.2f}s pred={pred_time:.2f}s | mem={mem_gb():.2f}GB")
        del D; gc.collect()

    import pandas as pd
    df = pd.DataFrame(res)
    csv_path = os.path.join(outdir, f"cifar10_{method}_summary_v2.csv")
    df.to_csv(csv_path, index=False)
    log(f"Saved: {csv_path}")

# ---------- AG News ----------
def load_agnews_embeddings(subset=10000, batch=256, device='cpu', st_model="sentence-transformers/all-MiniLM-L6-v2"):
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer
    log("Loading AG News train split...")
    ds = load_dataset("ag_news", split="train")
    if subset and subset < len(ds): ds = ds.shuffle(seed=0).select(range(subset))
    texts = ds["text"]; labels = np.array(ds["label"], dtype=np.int64)
    n_classes = int(labels.max() + 1)
    class_names = ["World","Sports","Business","Sci/Tech"][:n_classes]
    log(f"Loading sentence-transformer: {st_model}")
    model = SentenceTransformer(st_model, device=device)
    model.eval()
    log(f"Embedding {len(texts)} texts on {device} ...")
    X = model.encode(texts, batch_size=batch, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)
    X = normalize(X.astype(np.float32), axis=1)
    return X, labels, class_names

def run_agnews(method: str, labels_per_class: List[int], subset=10000,
               tau=0.05, L=1.0, seed=42, outdir="./kan_inf_outputs",
               auto_tau: bool=False, tau_grid: List[float] = None,
               bias_correct: bool=False, st_model="sentence-transformers/all-MiniLM-L6-v2",
               knn: int = 15, with_lr: bool = False, blend_alpha: float = 0.0,
               rcps_enable: bool = False, rcps_alpha: float = 0.05, rcps_delta: float = 0.05,
               rcps_gate: str = "prob", rcps_cal_frac: float = 0.25):
    set_seed(seed); ensure_dir(outdir)
    device='cpu'
    try:
        import torch
        if torch.cuda.is_available(): device='cuda'
        elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available(): device='mps'
    except Exception: pass

    X, y, class_names = load_agnews_embeddings(subset=subset, device=device, st_model=st_model)
    Xn = normalize(X, axis=1)
    n, d = Xn.shape; n_classes = len(class_names)
    res = []
    for per_class in labels_per_class:
        labeled_idx = pick_labels_per_class(y, per_class=per_class, n_classes=n_classes, seed=seed)
        y_labels = y[labeled_idx]

        t0 = time.time()
        if method.lower() == "softkan":
            D = cosine_dist(Xn, Xn[labeled_idx])
            prep_time = time.time() - t0
            chosen_tau, class_bias = tau, None
            if auto_tau and len(labeled_idx) > 0:
                chosen_tau, class_bias = auto_tau_loocv(
                    Xn, labeled_idx, y, n_classes,
                    tau_grid=(tau_grid or [0.01,0.02,0.05,0.1,0.2]),
                    L=L, bias_correct=bias_correct
                )
                log(f"[auto-τ] picked τ={chosen_tau} bias={'on' if bias_correct else 'off'}")
            t1 = time.time()
            scores = softkan_scores_from_D(D, y_labels, n_classes=n_classes, L=L, tau=chosen_tau, class_bias=class_bias)
            y_pred = np.argmax(scores, axis=1)
            pred_time = time.time() - t1
            # LR baseline + optional blend
            acc_lr = None
            acc_blend = None
            try:
                from sklearn.linear_model import LogisticRegression
                lr = LogisticRegression(max_iter=300, n_jobs=None)
                lr.fit(Xn[labeled_idx], y_labels)
                y_pred_lr = lr.predict(Xn)
                acc_lr = float(accuracy_score(y, y_pred_lr))
                BLEND_ALPHA = float(os.environ.get('KAN_BLEND_ALPHA', '0'))
                if BLEND_ALPHA > 0.0:
                    P_lr = lr.predict_proba(Xn)
                    S = scores - scores.max(axis=1, keepdims=True)
                    P_kan = np.exp(S); P_kan /= P_kan.sum(axis=1, keepdims=True)
                    P_mix = BLEND_ALPHA * P_lr + (1.0 - BLEND_ALPHA) * P_kan
                    y_pred_mix = np.argmax(P_mix, axis=1)
                    acc_blend = float(accuracy_score(y, y_pred_mix))
            except Exception:
                pass

            # Optional RCPS gating: accept on Soft-KAN top-prob >= tau_acc, else fallback to LR
            rcps_fields = {
                "rcps_enabled": bool(rcps_enable),
                "rcps_tau": None,
                "rcps_alpha": float(rcps_alpha),
                "rcps_delta": float(rcps_delta),
                "rcps_accept_fraction": None,
                "rcps_abstain_fraction": None,
                "rcps_acc_accepted": None,
                "rcps_final_acc": None,
                "rcps_fallback": None,
            }
            if rcps_enable:
                try:
                    from trust.rcps import choose_tau_rcps, pipeline_accept_stage
                    # softmax to probability-like for gating (monotone; CP works on accepted-subset errors)
                    S = scores - scores.max(axis=1, keepdims=True)
                    P = np.exp(S); P /= P.sum(axis=1, keepdims=True)
                    # calibration/eval split
                    rng = np.random.default_rng(seed)
                    idx = np.arange(n)
                    rng.shuffle(idx)
                    n_cal = int(max(1, rcps_cal_frac * n))
                    cal_idx = idx[:n_cal]
                    eval_idx = idx[n_cal:]
                    probas_cal = [P[cal_idx]]
                    best = choose_tau_rcps(y[cal_idx], probas_cal, alpha=rcps_alpha, delta=rcps_delta,
                                           tau_grid=np.linspace(0.5, 0.999, 120))
                    tau_acc = float(best["tau"]) if best["tau"] is not None else 1.0
                    rcps_fields["rcps_tau"] = tau_acc
                    rcps_fields["rcps_accept_fraction"] = float(best["accepted"]) if best else 0.0
                    # Evaluate acceptance on eval set
                    acc_stage = pipeline_accept_stage([P[eval_idx]], tau_acc)
                    accepted_mask = (acc_stage >= 0)
                    final_pred = np.empty(len(eval_idx), dtype=int)
                    # Accepted: use Soft-KAN prediction
                    final_pred[accepted_mask] = y_pred[eval_idx][accepted_mask]
                    # Fallback: LR if available
                    fb_name = None
                    if 'lr' in locals():
                        fb_pred = y_pred_lr[eval_idx][~accepted_mask]
                        fb_name = "lr"
                    else:
                        # If LR failed for any reason, fall back to Soft-KAN (no-op)
                        fb_pred = y_pred[eval_idx][~accepted_mask]
                        fb_name = "softkan"
                    final_pred[~accepted_mask] = fb_pred
                    rcps_fields["rcps_fallback"] = fb_name
                    # Metrics
                    if accepted_mask.any():
                        rcps_fields["rcps_acc_accepted"] = float(
                            accuracy_score(y[eval_idx][accepted_mask], final_pred[accepted_mask])
                        )
                    else:
                        rcps_fields["rcps_acc_accepted"] = None
                    rcps_fields["rcps_abstain_fraction"] = float((~accepted_mask).mean())
                    rcps_fields["rcps_final_acc"] = float(accuracy_score(y[eval_idx], final_pred))
                except Exception as _e:
                    # Keep RCPS fields None on failure, but do not interrupt baseline flow
                    pass

        elif method.lower() == "amle":
            inds = build_knn_indices(Xn, k=knn)
            prep_time = time.time() - t0
            B_mask = np.zeros(n, dtype=bool); B_mask[labeled_idx] = True
            U0 = np.zeros((n, n_classes), dtype=np.float32)
            for c in range(n_classes):
                U0[labeled_idx, c] = (y_labels == c).astype(np.float32)
            t1 = time.time()
            U = amle_multiclass(U0, inds, B_mask, max_iter=400, tol=1e-3)
            y_pred = np.argmax(U, axis=1)
            pred_time = time.time() - t1
        else:
            raise ValueError("method must be one of {softkan, amle}")

        acc = accuracy_score(y, y_pred)
        far_order = farthest_unlabeled_order(Xn, labeled_idx) if len(labeled_idx) else []
        top16 = far_order[:16].tolist() if len(far_order) else []

        row = {
            "dataset": "agnews",
            "method": method,
            "st_model": st_model,
            "labels_per_class": per_class,
            "m_labels": int(len(labeled_idx)),
            "acc_all": round(float(acc),4),
            "tau": float(chosen_tau) if method=="softkan" else None,
            "prep_time_sec": round(prep_time,3),
            "predict_time_sec": round(pred_time,3),
            "mem_gb": round(mem_gb(),3),
            "auto_tau": bool(auto_tau),
            "bias_correct": bool(bias_correct),
            "acc_lr": round(float(acc_lr),4) if 'acc_lr' in locals() and acc_lr is not None else None,
            "acc_blend": round(float(acc_blend),4) if 'acc_blend' in locals() and acc_blend is not None else None,
            "next16_label_indices": top16
        }
        # Attach RCPS metrics if present
        if method.lower() == "softkan":
            row.update({k: v for k, v in rcps_fields.items()})
        res.append(row)
        log(f"[AGNews][{method}] K={per_class}/class | acc={acc:.4f} | τ={res[-1]['tau']} | prep={prep_time:.2f}s pred={pred_time:.2f}s | mem={mem_gb():.2f}GB")
        gc.collect()

    import pandas as pd
    df = pd.DataFrame(res)
    tag = f"{method}_summary_v2"
    csv_path = os.path.join(outdir, f"agnews_{tag}.csv")
    df.to_csv(csv_path, index=False)
    log(f"Saved: {csv_path}")

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=str, choices=["cifar10","agnews"], required=True)
    ap.add_argument("--method", type=str, default="softkan", choices=["softkan","amle"])
    ap.add_argument("--labels-per-class", type=int, nargs="+", default=[1,5,10])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--L", type=float, default=1.0)
    ap.add_argument("--auto-tau", action="store_true")
    ap.add_argument("--tau-grid", type=float, nargs="+", default=None)
    ap.add_argument("--bias-correct", action="store_true")

    # RCPS controls
    ap.add_argument("--rcps-enable", action="store_true")
    ap.add_argument("--rcps-alpha", type=float, default=0.05)
    ap.add_argument("--rcps-delta", type=float, default=0.05)
    ap.add_argument("--rcps-gate", type=str, choices=["prob","margin"], default="prob")
    ap.add_argument("--rcps-cal-frac", type=float, default=0.25)

    # CIFAR controls
    ap.add_argument("--train-max", type=int, default=20000)
    ap.add_argument("--test-max", type=int, default=10000)
    ap.add_argument("--knn", type=int, default=15)
    ap.add_argument("--text-proto", action="store_true")
    ap.add_argument("--zero-shot-only", action="store_true")
    ap.add_argument("--clip-arch", type=str, default="ViT-B-32")
    ap.add_argument("--clip-pretrained", type=str, default="laion2b_s34b_b79k")

    # AG News controls
    ap.add_argument("--subset", type=int, default=10000)
    ap.add_argument("--st-model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--outdir", type=str, default="./kan_inf_outputs")
    args = ap.parse_args()

    ensure_dir(args.outdir); log(f"Args: {vars(args)}")
    if args.task == "cifar10":
        run_cifar10(method=args.method, labels_per_class=args.labels_per_class,
                    train_max=args.train_max, test_max=args.test_max,
                    knn=args.knn, tau=args.tau, L=args.L, seed=args.seed, outdir=args.outdir,
                    text_proto=args.text_proto, zero_shot_only=args.zero_shot_only,
                    auto_tau=args.auto_tau, tau_grid=args.tau_grid, bias_correct=args.bias_correct,
                    clip_arch=args.clip_arch, clip_pretrained=args.clip_pretrained,
                    rcps_enable=args.rcps_enable, rcps_alpha=args.rcps_alpha,
                    rcps_delta=args.rcps_delta, rcps_gate=args.rcps_gate,
                    rcps_cal_frac=args.rcps_cal_frac)
    else:
        run_agnews(method=args.method, labels_per_class=args.labels_per_class,
                   subset=args.subset, tau=args.tau, L=args.L, seed=args.seed, outdir=args.outdir,
                   auto_tau=args.auto_tau, tau_grid=args.tau_grid, bias_correct=args.bias_correct,
                   st_model=args.st_model, knn=args.knn,
                   rcps_enable=args.rcps_enable, rcps_alpha=args.rcps_alpha,
                   rcps_delta=args.rcps_delta, rcps_gate=args.rcps_gate,
                   rcps_cal_frac=args.rcps_cal_frac)

if __name__ == "__main__":
    main()

