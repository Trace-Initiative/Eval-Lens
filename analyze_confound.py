#!/usr/bin/env python3
"""
STEP 3 (analyze) — CPU, re-runnable. Runs the pre-registered confirmatory analysis on the saved
activations from extract_confound.py. No GPU.

Implements the pre-registration:
  * split BY TASK (70/30), layer chosen by CV on TRAIN tasks only
  * core probe: held-out AUROC + random placebo + Cohen's d + task-clustered bootstrap CI + perm null
  * baselines: length-only classifier + black-box "ask Qwen"
  * per-factor attribution (F1..F8): which cues the probe reads; length-neutral F1/F3/F5 are the clean test
  * cosine matrix of per-factor directions: one eval-awareness direction, or many?

  python analyze_confound.py
"""

import os
import json
import numpy as np
from sklearn.metrics import roc_auc_score

PROJ = os.path.dirname(os.path.abspath(__file__))
ACTS = os.path.join(PROJ, "data", "acts")
PROMPTS = os.path.join(PROJ, "data", "prompts")
TAG = "7b"
SEED = 0
TEST_FRAC = 0.30
FACTORS = ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8"]
CLEAN_FACTORS = {"F1", "F3", "F5"}          # length-neutral, non-wrapper -> the honest test
WRAPPER_FACTORS = {"F6", "F7", "F8"}
rng = np.random.default_rng(SEED)


def npz(name):
    return np.load(os.path.join(ACTS, f"{name}_{TAG}.npz"), allow_pickle=False)


def word_len(jsonl):
    return np.array([len(json.loads(l)["user_text"].split()) for l in open(os.path.join(PROMPTS, jsonl))])


def dom_dir(X, y, L):
    d = X[y == 1, L].mean(0) - X[y == 0, L].mean(0)
    return d / (np.linalg.norm(d) + 1e-8)


def au(y, s):
    return roc_auc_score(y, s)


def au2(y, s):                                # orientation-free (for baselines/placebo)
    a = roc_auc_score(y, s)
    return max(a, 1 - a)


def placebo(X, y, L, n=50):
    H = X.shape[2]
    a = [au2(y, X[:, L] @ (r / (np.linalg.norm(r) + 1e-8))) for r in rng.standard_normal((n, H))]
    return float(np.mean(a)), float(np.max(a))


def cohend(a, b):
    sp = np.sqrt(((len(a)-1)*a.var(ddof=1) + (len(b)-1)*b.var(ddof=1)) / (len(a)+len(b)-2) + 1e-12)
    return float((a.mean() - b.mean()) / sp)


# ------------------------------------------------------------------ load core
core = npz("core")
Xc = core["X"].astype(np.float32)            # (N, nLayers+1, H)
y = core["label"].astype(int)
task = core["task"]
verbal = core["verbal_p_eval"].astype(float)
clen = word_len("confound_core.jsonl")
assert len(clen) == len(y), "length/order mismatch — re-run extract_confound.py"

tasks = sorted(set(task.tolist()))
rng.shuffle(tasks)
ntest = int(len(tasks) * TEST_FRAC)
test_tasks, train_tasks = set(tasks[:ntest]), set(tasks[ntest:])
tr = np.array([t in train_tasks for t in task])
te = ~tr
print(f"tasks: {len(train_tasks)} train / {len(test_tasks)} test")
print(f"core rows: {tr.sum()} train / {te.sum()} test  (labels test: {y[te].sum()} eval / {(y[te]==0).sum()} deploy)")

# ------------------------------------------------------------------ layer selection (CV on train)
train_list = [t for t in tasks if t in train_tasks]
folds = np.array_split(rng.permutation(train_list), 5)
nLp = Xc.shape[1]
best, Lstar = -1, None
for L in range(1, nLp):
    aus = []
    for k in range(5):
        vs = set(folds[k].tolist())
        mv = np.array([t in vs for t in task]) & tr
        mf = tr & ~mv
        if mv.sum() < 6 or (y[mf] == 1).sum() < 2 or (y[mf] == 0).sum() < 2 or len(set(y[mv])) < 2:
            continue
        aus.append(au(y[mv], Xc[mv, L] @ dom_dir(Xc[mf], y[mf], L)))
    if aus and np.mean(aus) > best:
        best, Lstar = np.mean(aus), L
print(f"selected layer L{Lstar}  (train-CV AUROC {best:.3f})")

# ------------------------------------------------------------------ core probe on TEST
d_core = dom_dir(Xc[tr], y[tr], Lstar)
sc = Xc[te, Lstar] @ d_core
core_auroc = au(y[te], sc)
pl_mean, pl_max = placebo(Xc[te], y[te], Lstar)
d_eff = cohend(sc[y[te] == 1], sc[y[te] == 0])

te_task_list = [t for t in tasks if t in test_tasks]
boot = []
for _ in range(2000):
    samp = rng.choice(te_task_list, size=len(te_task_list), replace=True)
    idx = np.concatenate([np.where(task == t)[0] for t in samp])
    if len(set(y[idx])) < 2:
        continue
    boot.append(au(y[idx], Xc[idx, Lstar] @ d_core))
ci_lo, ci_hi = np.percentile(boot, 2.5), np.percentile(boot, 97.5)

obs = core_auroc
cnt = sum(au(rng.permutation(y[te]), sc) >= obs for _ in range(1000))
pval = (cnt + 1) / 1001

# baselines
len_auroc = au2(y[te], clen[te].astype(float))
verbal_auroc = au2(y[te], verbal[te])
verbal_raw = au(y[te], verbal[te])

print("\n" + "=" * 72)
print("CORE PROBE (content-matched eval-like vs deploy-like), held-out by task")
print("=" * 72)
print(f"  probe AUROC        = {core_auroc:.3f}   95% CI [{ci_lo:.3f}, {ci_hi:.3f}]   perm p={pval:.3f}")
print(f"  Cohen's d          = {d_eff:.2f}")
print(f"  random placebo     = mean {pl_mean:.3f}, max {pl_max:.3f}   (should be ~0.5 now, unlike confounded 0.76)")
print(f"  baseline: length   = {len_auroc:.3f}   (probe must beat this)")
print(f"  baseline: ask-Qwen = {verbal_auroc:.3f}   (raw, oriented: {verbal_raw:.3f})")
print(f"  H1 signal-exists   : {'PASS' if core_auroc > 0.7 and pl_max < 0.65 else 'weak/na'}")
print(f"  H3 beats baselines : {'PASS' if core_auroc > len_auroc and core_auroc > verbal_auroc else 'NO — probe ~ baseline'}")

# ------------------------------------------------------------------ per-factor attribution
flips = npz("flips")
Xf = flips["X"].astype(np.float32)
ftask = flips["task"]
ffac = flips["factor"]
base_m = ffac == "none"
print("\n" + "=" * 72)
print("PER-FACTOR ATTRIBUTION (base vs single-factor flip), held-out by task")
print("  clean = length-neutral non-wrapper (F1,F3,F5) -> a hit here can't be dismissed as length")
print("=" * 72)
factor_dirs = {}
for F in FACTORS:
    fm = ffac == F
    X = np.concatenate([Xf[base_m], Xf[fm]])
    yl = np.concatenate([np.zeros(base_m.sum()), np.ones(fm.sum())]).astype(int)
    tk = np.concatenate([ftask[base_m], ftask[fm]])
    trm = np.array([t in train_tasks for t in tk])
    tem = ~trm
    d = dom_dir(X[trm], yl[trm], Lstar)
    factor_dirs[F] = d
    a = au(yl[tem], X[tem, Lstar] @ d)
    _, plx = placebo(X[tem], yl[tem], Lstar, n=30)
    kind = "CLEAN" if F in CLEAN_FACTORS else ("wrapper" if F in WRAPPER_FACTORS else "length")
    flag = " <<" if a > plx and a > 0.7 else ""
    print(f"  {F} [{kind:7s}] AUROC={a:.3f}  placebo_max={plx:.3f}{flag}")

# ------------------------------------------------------------------ cosine: one direction or many?
print("\n" + "=" * 72)
print("COSINE between per-factor directions (low = factor-specific 'not one capability'; high = shared)")
print("=" * 72)
print("      " + "  ".join(FACTORS))
cos = np.zeros((8, 8))
for i, A in enumerate(FACTORS):
    for j, B in enumerate(FACTORS):
        cos[i, j] = float(factor_dirs[A] @ factor_dirs[B])
    print(f"  {A} " + "  ".join(f"{cos[i,j]:+.2f}" for j in range(8)))
off = cos[~np.eye(8, dtype=bool)]
print(f"\n  mean off-diagonal cosine = {off.mean():.2f}   "
      f"(clean-factor cosines are the meaningful ones)")
print(f"  cosine(core direction vs F1/F3/F5): "
      + ", ".join(f"{F}={float(d_core@factor_dirs[F]):+.2f}" for F in ['F1', 'F3', 'F5']))

# ------------------------------------------------------------------ layer profile (clean factors)
print("\n" + "=" * 72)
print("LAYER PROFILE for CLEAN factors (F1, F3, F5): AUROC vs placebo across depth")
print("  early-layer signal = surface/token separability; a MID-LATE peak clearly above placebo")
print("  = genuine internal representation of the cue (not just token matching).")
print("=" * 72)
for F in ["F1", "F3", "F5"]:
    fm = ffac == F
    X = np.concatenate([Xf[base_m], Xf[fm]])
    yl = np.concatenate([np.zeros(base_m.sum()), np.ones(fm.sum())]).astype(int)
    tk = np.concatenate([ftask[base_m], ftask[fm]])
    trm = np.array([t in train_tasks for t in tk]); tem = ~trm
    print(f"\n  {F}:")
    best_gap, best_L = -1, None
    for L in range(2, nLp, 2):
        d = dom_dir(X[trm], yl[trm], L)
        a = au(yl[tem], X[tem, L] @ d)
        _, plx = placebo(X[tem], yl[tem], L, n=20)
        gap = a - plx
        mark = "  << gap" if gap > 0.10 and a > 0.7 else ""
        print(f"    L{L:2d}  AUROC={a:.3f}  placebo={plx:.3f}  gap={gap:+.3f}{mark}")
        if gap > best_gap:
            best_gap, best_L = gap, L
    zone = "early(surface)" if best_L <= 8 else ("mid" if best_L <= 18 else "late")
    print(f"    -> best gap {best_gap:+.3f} at L{best_L} [{zone}]")

print("\n" + "=" * 72)
print("READ vs PRE-REGISTRATION: H1 (signal+placebo~0.5), H2 (F1/F3/F5 above placebo), "
      "H3 (beats length+ask-Qwen), H4 (cosine). Report all pass/fail honestly.")
print("=" * 72)
