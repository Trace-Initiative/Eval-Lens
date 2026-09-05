#!/usr/bin/env python3
"""
STEP 3 (figures) — CPU. Generate the 4 write-up figures from saved activations.
Saves PNGs to data/figures/.  Run:  python make_figures.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

PROJ = os.path.dirname(os.path.abspath(__file__))
ACTS = os.path.join(PROJ, "data", "acts")
FIG = os.path.join(PROJ, "data", "figures")
os.makedirs(FIG, exist_ok=True)
TAG = "7b"
SEED = 0
rng = np.random.default_rng(SEED)
FACTORS = ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8"]
CLEAN = {"F1", "F3", "F5"}
WRAP = {"F6", "F7", "F8"}


def dom(X, y, L):
    d = X[y == 1, L].mean(0) - X[y == 0, L].mean(0)
    return d / (np.linalg.norm(d) + 1e-8)


def au(y, s):
    return roc_auc_score(y, s)


def au2(y, s):
    a = roc_auc_score(y, s); return max(a, 1 - a)


def placebo(X, y, L, n=25):
    H = X.shape[2]
    return max(au2(y, X[:, L] @ (r / (np.linalg.norm(r) + 1e-8))) for r in rng.standard_normal((n, H)))


core = np.load(os.path.join(ACTS, f"core_{TAG}.npz"))
Xc = core["X"].astype(np.float32); y = core["label"].astype(int); task = core["task"]
tasks = sorted(set(task.tolist())); rng.shuffle(tasks)
ntest = int(len(tasks) * 0.30)
test_t, train_t = set(tasks[:ntest]), set(tasks[ntest:])
tr = np.array([t in train_t for t in task]); te = ~tr

flips = np.load(os.path.join(ACTS, f"flips_{TAG}.npz"))
Xf = flips["X"].astype(np.float32); ftask = flips["task"]; ffac = flips["factor"]
base_m = ffac == "none"
nL = Xc.shape[1]


def factor_curve(F):
    fm = ffac == F
    X = np.concatenate([Xf[base_m], Xf[fm]])
    yl = np.concatenate([np.zeros(base_m.sum()), np.ones(fm.sum())]).astype(int)
    tk = np.concatenate([ftask[base_m], ftask[fm]])
    trm = np.array([t in train_t for t in tk]); tem = ~trm
    A, P = [], []
    for L in range(1, nL):
        d = dom(X[trm], yl[trm], L)
        A.append(au(yl[tem], X[tem, L] @ d)); P.append(placebo(X[tem], yl[tem], L))
    return np.array(A), np.array(P), (X, yl, trm, tem)


def factor_dir_at(F, L):
    """Per-factor direction at a SPECIFIC layer (for a valid same-space cosine comparison)."""
    fm = ffac == F
    X = np.concatenate([Xf[base_m], Xf[fm]])
    yl = np.concatenate([np.zeros(base_m.sum()), np.ones(fm.sum())]).astype(int)
    tk = np.concatenate([ftask[base_m], ftask[fm]])
    trm = np.array([t in train_t for t in tk])
    return dom(X[trm], yl[trm], L)


# ---- Fig 1: per-factor best AUROC vs placebo ----
best_au, best_pl, colors = [], [], []
dirs = {}
for F in FACTORS:
    A, P, (X, yl, trm, tem) = factor_curve(F)
    gap = A - P
    Lb = int(np.argmax(gap)) + 1
    best_au.append(A[Lb - 1]); best_pl.append(P[Lb - 1])
    dirs[F] = dom(X[trm], yl[trm], Lb)
    colors.append("#2a9d8f" if F in CLEAN else ("#e76f51" if F in WRAP else "#e9c46a"))
x = np.arange(8)
plt.figure(figsize=(8, 4))
plt.bar(x - 0.2, best_au, 0.4, label="probe AUROC", color=colors)
plt.bar(x + 0.2, best_pl, 0.4, label="random placebo", color="#bbbbbb")
plt.xticks(x, FACTORS); plt.ylim(0.4, 1.02); plt.axhline(0.5, ls=":", c="k", lw=0.8)
plt.ylabel("held-out AUROC"); plt.title("Per-factor probe vs placebo (best layer)\ngreen=length-neutral, yellow=length, red=wrapper")
plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig1_per_factor.png"), dpi=130); plt.close()

# ---- Fig 2: layer profiles for F1, F5, F3 ----
plt.figure(figsize=(8, 4.5))
for F, c in [("F1", "#264653"), ("F5", "#2a9d8f"), ("F3", "#e76f51")]:
    A, P, _ = factor_curve(F)
    Ls = np.arange(1, nL)
    plt.plot(Ls, A, "-o", ms=3, c=c, label=f"{F} AUROC")
    plt.plot(Ls, P, "--", c=c, alpha=0.5, label=f"{F} placebo")
plt.xlabel("layer"); plt.ylabel("AUROC"); plt.ylim(0.45, 1.02)
plt.title("Layer profile: genuine signal = AUROC clearly above placebo at MID layers")
plt.legend(fontsize=8, ncol=3); plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig2_layer_profile.png"), dpi=130); plt.close()

# ---- Fig 3: cosine matrix — directions at ONE COMMON layer (valid same-space comparison) ----
L_COS = 14   # mid-layer, where the genuine per-factor signals live
cdirs = {F: factor_dir_at(F, L_COS) for F in FACTORS}
M = np.array([[float(cdirs[a] @ cdirs[b]) for b in FACTORS] for a in FACTORS])
plt.figure(figsize=(5.5, 4.8))
im = plt.imshow(M, vmin=-0.2, vmax=1, cmap="viridis")
plt.colorbar(im, fraction=0.046); plt.xticks(range(8), FACTORS); plt.yticks(range(8), FACTORS)
for i in range(8):
    for j in range(8):
        plt.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                 color="white" if M[i, j] < 0.6 else "black", fontsize=7)
plt.title(f"Per-factor direction cosine at L{L_COS} (mean off-diag {M[~np.eye(8,dtype=bool)].mean():.2f})\nlow = 'not one capability'")
plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig3_cosine.png"), dpi=130); plt.close()

# ---- Fig 4: base-rate precision vs recall ----
wc_path = os.path.join(ACTS, f"wildchat_{TAG}.npz")
if os.path.exists(wc_path):
    Xw = np.load(wc_path)["X"].astype(np.float32)
    Lg = 14
    d = dom(Xc[tr], y[tr], Lg)
    pos = Xc[(te) & (y == 1), Lg] @ d
    neg = Xw[:, Lg] @ d
    Npos = max(1, round(len(neg) * 0.01 / 0.99))
    pos = pos[:Npos]
    recs = np.linspace(0.05, 1.0, 20)
    precs = []
    for r in recs:
        thr = np.quantile(pos, 1 - r)
        tp = (pos >= thr).sum(); fp = (neg >= thr).sum()
        precs.append(tp / (tp + fp + 1e-9))
    plt.figure(figsize=(7, 4))
    plt.plot(recs, precs, "-o", ms=3, c="#e76f51")
    plt.xlabel("recall (fraction of evals caught)"); plt.ylabel("precision")
    plt.ylim(0, 1); plt.title(f"Base-rate reliability at ~1% prevalence\n(AUROC high, but precision low: most flags are false positives)")
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig4_base_rate.png"), dpi=130); plt.close()
    print("wrote fig4_base_rate.png")

print("figures ->", FIG)
for f in sorted(os.listdir(FIG)):
    print("  ", f)
