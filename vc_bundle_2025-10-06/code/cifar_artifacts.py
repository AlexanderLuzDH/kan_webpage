#!/usr/bin/env python3
import os, json, argparse
import numpy as np
from pathlib import Path
from typing import List, Tuple

import torch
import torchvision
import torchvision.transforms as T
import open_clip
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


CIFAR10_CLASSES = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
TEMPLATES = [
    "a photo of a {}",
    "a blurry photo of a {}",
    "a close-up photo of a {}",
    "a low-resolution photo of a {}",
    "a bright photo of a {}",
    "a cropped photo of a {}",
    "a photo of the {}"
]


def ensure_dir(p: str): Path(p).mkdir(parents=True, exist_ok=True)


def load_cifar_subset(train_max=5000, test_max=2000, device='cpu',
                      clip_arch='ViT-B-32', clip_pretrained='laion2b_s34b_b79k',
                      cache_npz: str | None = None) -> Tuple[np.ndarray, np.ndarray, int, int, any]:
    if cache_npz and os.path.exists(cache_npz):
        data = np.load(cache_npz)
        X_all = data['X_all'].astype(np.float32)
        y_all = data['y_all'].astype(np.int64)
        n_train = int(data['n_train'])
        n_test = int(data['n_test'])
        # Load a CLIP model for text prototypes
        model, _, _ = open_clip.create_model_and_transforms(clip_arch, pretrained=clip_pretrained, device=device)
        return X_all, y_all, n_train, n_test, model

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
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=False, num_workers=2, pin_memory=False)
    test_loader  = torch.utils.data.DataLoader(testset,  batch_size=128, shuffle=False, num_workers=2, pin_memory=False)

    model, _, _ = open_clip.create_model_and_transforms(clip_arch, pretrained=clip_pretrained, device=device)
    model.eval()

    @torch.no_grad()
    def embed_images(loader):
        feats, ys = [], []
        for xb, yb in loader:
            xb = xb.to(device)
            f = model.encode_image(xb)
            f = f / f.norm(dim=-1, keepdim=True)
            feats.append(f.cpu().numpy().astype(np.float32))
            ys.append(yb.numpy().astype(np.int64))
        return np.concatenate(feats, 0), np.concatenate(ys, 0)

    Xtr, Ytr = embed_images(train_loader)
    Xte, Yte = embed_images(test_loader)
    X_all = normalize(np.vstack([Xtr, Xte]).astype(np.float32), axis=1)
    y_all = np.concatenate([Ytr, Yte]).astype(np.int64)
    n_train = Xtr.shape[0]
    n_test = Xte.shape[0]
    if cache_npz:
        np.savez(cache_npz, X_all=X_all, y_all=y_all, n_train=n_train, n_test=n_test)
    return X_all, y_all, n_train, n_test, model


def clip_text_prototypes(model, device='cpu') -> Tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        toks = []
        labels = []
        for c, name in enumerate(CIFAR10_CLASSES):
            for t in TEMPLATES:
                toks.append(open_clip.tokenize(t.format(name)))
                labels.append(c)
        toks = torch.cat(toks, dim=0).to(device)
        txt = model.encode_text(toks)
        txt = txt / txt.norm(dim=-1, keepdim=True)
    return txt.cpu().numpy().astype(np.float32), np.array(labels, dtype=np.int64)


def cosine_dist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return 1.0 - (A @ B.T).astype(np.float32)


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


def logsumexp(X: np.ndarray, axis=1):
    m = np.max(X, axis=axis, keepdims=True)
    return np.log(np.sum(np.exp(X - m), axis=axis)) + m.squeeze(axis=axis)


def smin_tau(Z: np.ndarray, tau: float) -> np.ndarray:
    return -tau * logsumexp(-Z / tau, axis=1)


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
        if len(neg):
            Zneg_up = (0.0 + L * D[:, neg])
            upper = -tau * logsumexp(np.concatenate([-(Zpos_up)/tau, -(Zneg_up)/tau], axis=1), axis=1)
        else:
            upper = smin_tau(Zpos_up, tau)
        Zpos_low = ((-L * D[:, pos]) + 1.0)
        if len(neg):
            Zneg_low = ((-L * D[:, neg]) + 0.0)
            lower = tau * logsumexp(np.concatenate([Zpos_low/tau, Zneg_low/tau], axis=1), axis=1)
        else:
            lower = tau * logsumexp(Zpos_low / tau, axis=1)
        scores[:, c] = 0.5 * (upper + lower)
    if class_bias is not None:
        scores = scores + class_bias[None, :].astype(np.float32)
    return scores


def auto_tau_loocv(Xn: np.ndarray, labeled_idx: np.ndarray, y: np.ndarray, n_classes: int,
                   tau_grid: List[float], L: float = 1.0, bias_correct: bool = False):
    if len(labeled_idx) == 0: return tau_grid[0], None
    best_tau, best_acc, best_bias = tau_grid[0], -1.0, None
    y_lab = y[labeled_idx]
    D_lab = cosine_dist(Xn[labeled_idx], Xn[labeled_idx])
    for tau in tau_grid:
        correct = 0
        for i in range(len(labeled_idx)):
            mask = np.ones(len(labeled_idx), dtype=bool); mask[i] = False
            D_i = D_lab[i:i+1, mask]
            y_i = y_lab[mask]
            scores_i = softkan_scores_from_D(D_i, y_i, n_classes=n_classes, L=L, tau=tau)
            pred = int(np.argmax(scores_i[0]))
            correct += int(pred == y_lab[i])
        acc = correct / len(labeled_idx)
        cb = None
        if bias_correct and len(labeled_idx) > 0:
            s_full = softkan_scores_from_D(D_lab, y_lab, n_classes=n_classes, L=L, tau=tau)
            cb = np.zeros(n_classes, dtype=np.float32)
            for c in range(n_classes):
                pos = (y_lab == c)
                if pos.any():
                    cb[c] = 1.0 - float(s_full[pos, c].mean())
        if acc > best_acc:
            best_acc, best_tau, best_bias = acc, tau, cb
    return best_tau, best_bias


def coverage_curve(y_true: np.ndarray, y_pred: np.ndarray, dmin: np.ndarray,
                   qs: np.ndarray) -> np.ndarray:
    rows = []
    for q in qs:
        t = np.quantile(dmin, q)
        keep = dmin <= t
        cov = float(keep.mean())
        acc = float(accuracy_score(y_true[keep], y_pred[keep])) if keep.any() else np.nan
        rows.append((float(q), cov, acc))
    return np.array(rows, dtype=np.float32)


def make_tile(images: List[np.ndarray], out_png: str, cols: int = 4):
    rows = (len(images) + cols - 1) // cols
    plt.figure(figsize=(cols*3, rows*3))
    for i, img in enumerate(images):
        ax = plt.subplot(rows, cols, i+1)
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-max', type=int, default=5000)
    ap.add_argument('--test-max', type=int, default=2000)
    ap.add_argument('--per-class', type=int, nargs='+', default=[5,10])
    ap.add_argument('--outdir', type=str, default='./kan_inf_outputs')
    ap.add_argument('--clip-arch', type=str, default='ViT-B-32')
    ap.add_argument('--clip-pretrained', type=str, default='laion2b_s34b_b79k')
    ap.add_argument('--tau', type=float, default=0.05)
    ap.add_argument('--bias-correct', action='store_true')
    ap.add_argument('--cache', type=str, default='./kan_inf_outputs/cifar10_ViT-B-32_5000_2000.npz')
    args = ap.parse_args()

    ensure_dir(args.outdir)
    device = 'cpu'
    X, y, n_train, n_test, model = load_cifar_subset(args.train_max, args.test_max, device=device,
                                                     clip_arch=args.clip_arch, clip_pretrained=args.clip_pretrained,
                                                     cache_npz=args.cache)
    Xn = normalize(X, axis=1)
    txt_emb, txt_lbl = clip_text_prototypes(model, device=device)
    txt_emb = normalize(txt_emb, axis=1)

    # raw images for tiles
    raw_transform = T.Compose([T.Resize(224, antialias=True), T.CenterCrop(224)])
    train_raw = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=raw_transform)
    if args.train_max < len(train_raw):
        idx = np.random.RandomState(0).choice(len(train_raw), size=args.train_max, replace=False)
        train_raw = torch.utils.data.Subset(train_raw, idx.tolist())

    for k in args.per_class:
        labeled_idx = pick_labels_per_class(y[:n_train], per_class=k, n_classes=10, seed=42)
        # auto-τ on labels
        tau_lab, class_bias = auto_tau_loocv(Xn[:n_train], labeled_idx, y[:n_train], 10,
                                             tau_grid=[0.01,0.02,0.05,0.1,0.2], L=1.0, bias_correct=args.bias_correct)
        # label vs proto distances for TEST set
        test_idx = np.arange(n_train, n_train+n_test)
        D_lab_te = cosine_dist(Xn[test_idx], Xn[labeled_idx]) if len(labeled_idx) else None
        D_pro_te = cosine_dist(Xn[test_idx], txt_emb)
        # scores
        S_lab_te = softkan_scores_from_D(D_lab_te, y[labeled_idx], n_classes=10, L=1.0, tau=tau_lab, class_bias=class_bias) if D_lab_te is not None else None
        S_pro_te = softkan_scores_from_D(D_pro_te, txt_lbl, n_classes=10, L=1.0, tau=args.tau)
        # pick alpha by LOOCV on labeled set
        alpha = 0.5
        if len(labeled_idx):
            alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
            best, best_acc = 0.5, -1.0
            D_ll = cosine_dist(Xn[labeled_idx], Xn[labeled_idx])
            for a in alphas:
                correct = 0
                for i in range(len(labeled_idx)):
                    mask = np.ones(len(labeled_idx), dtype=bool); mask[i] = False
                    s_li = softkan_scores_from_D(D_ll[i:i+1, mask], y[labeled_idx][mask], 10, L=1.0, tau=tau_lab)
                    D_pi = cosine_dist(Xn[labeled_idx[i]:labeled_idx[i]+1], txt_emb)
                    s_pi = softkan_scores_from_D(D_pi, txt_lbl, 10, L=1.0, tau=args.tau)
                    s_i = (1-a)*s_pi + a*s_li
                    pred = int(np.argmax(s_i[0]))
                    correct += int(pred == y[labeled_idx[i]])
                acc_cv = correct / len(labeled_idx)
                if acc_cv > best_acc: best_acc, best = acc_cv, a
            alpha = best

        # fused predictions and coverage
        if S_lab_te is None:
            S_fused = S_pro_te
        else:
            S_fused = (1-alpha)*S_pro_te + alpha*S_lab_te
        y_pred_te = np.argmax(S_fused, axis=1)
        y_true_te = y[test_idx]
        # dmin to boundary
        dmin_lab = D_lab_te.min(axis=1) if D_lab_te is not None else np.full(n_test, np.inf, dtype=np.float32)
        dmin_pro = D_pro_te.min(axis=1)
        dmin = np.minimum(dmin_lab, dmin_pro)
        rows = coverage_curve(y_true_te, y_pred_te, dmin, np.linspace(0,1,21))
        out_csv = os.path.join(args.outdir, f"cifar10_coverage_L{k}.csv")
        np.savetxt(out_csv, rows, delimiter=",", header="quantile,coverage,accuracy", comments="")

        # next 16 to label: farthest from labeled set (train pool)
        if len(labeled_idx):
            D_train = cosine_dist(Xn[:n_train], Xn[labeled_idx])
            mind = D_train.min(axis=1)
            order = np.argsort(mind)[::-1]
            top16 = order[:16]
            # build tile
            imgs = []
            for idx in top16:
                img, _ = train_raw[idx]
                # Normalize to float [0,1] regardless of PIL or Tensor input
                if isinstance(img, torch.Tensor):
                    arr = img.detach().cpu()
                    if arr.ndim == 3 and arr.shape[0] in (1,3):
                        arr = arr.permute(1,2,0)
                    arr = arr.numpy()
                else:
                    arr = np.array(img)
                if arr.dtype == np.uint8:
                    arr = arr.astype(np.float32) / 255.0
                arr = np.clip(arr, 0.0, 1.0)
                imgs.append(arr)
            out_png = os.path.join(args.outdir, f"cifar10_next16_L{k}.png")
            make_tile(imgs, out_png, cols=4)

    print("Artifacts written to", args.outdir)


if __name__ == '__main__':
    main()
