#!/usr/bin/env python3
"""
STEP 3 (extra) — CPU. The last three pre-registered checks, on saved activations:
  H5  internal probe vs the model's verbalized "is this an eval?" judgment (Thread A)
  H6  base-rate / false-positive test: precision at a realistic ~1% eval prevalence (WildChat stream)
  H7  cross-dataset: does the CONFOUNDED EvalDetectBench direction fail on the clean pairs? does the
      clean direction transfer to realistic data?

Needs: data/acts/{core_7b,test_7b,wildchat_7b}.npz and data/probe/direction_7b.npz
(run extract_probe.py, extract_confound.py, extract_wildchat.py first).

  python analyze_extra.py
"""

import os
import numpy as np
from sklearn.metrics import roc_auc_score

PROJ = os.path.dirname(os.path.abspath(__file__))
ACTS = os.path.join(PROJ, "data", "acts")
PROBE = os.path.join(PROJ, "data", "probe")
TAG = "7b"
SEED = 0
TEST_FRAC = 0.30
rng = np.random.default_rng(SEED)


def npz(path):
    return np.load(path, allow_pickle=False)


def dom_dir(X, y, L):
    d = X[y == 1, L].mean(0) - X[y == 0, L].mean(0)
    return d / (np.linalg.norm(d) + 1e-8)


def au2(y, s):
    a = roc_auc_score(y, s)
    return max(a, 1 - a)


# ---- load ----
core = npz(os.path.join(ACTS, f"core_{TAG}.npz"))
Xc = core["X"].astype(np.float32); y = core["label"].astype(int)
task = core["task"]; verbal = core["verbal_p_eval"].astype(float)

# same task split as analyze_confound.py
tasks = sorted(set(task.tolist())); rng.shuffle(tasks)
ntest = int(len(tasks) * TEST_FRAC)
test_tasks, train_tasks = set(tasks[:ntest]), set(tasks[ntest:])
tr = np.array([t in train_tasks for t in task]); te = ~tr

dconf = npz(os.path.join(PROBE, f"direction_{TAG}.npz"))
Lc = int(dconf["layer"]); d_conf = dconf["direction"].astype(np.float32)

conf_test = npz(os.path.join(ACTS, f"test_{TAG}.npz"))
Xtc = conf_test["X"].astype(np.float32); ytc = conf_test["y"].astype(int)

# =====================================================================  H7
print("=" * 72)
print("H7 — CROSS-DATASET (does the confounded direction capture clean eval-awareness?)")
print("=" * 72)
# confounded direction -> clean core test (at the confounded layer Lc)
a_conf_on_clean = au2(y[te], Xc[te, Lc] @ d_conf)
# clean core direction (train) at Lc -> confounded test
d_clean_Lc = dom_dir(Xc[tr], y[tr], Lc)
a_clean_on_conf = au2(ytc, Xtc[:, Lc] @ d_clean_Lc)
# sanity
a_conf_on_conf = au2(ytc, Xtc[:, Lc] @ d_conf)
a_clean_on_clean = au2(y[te], Xc[te, Lc] @ d_clean_Lc)
print(f"  confounded dir  -> confounded test : {a_conf_on_conf:.3f}  (sanity, ~0.92)")
print(f"  confounded dir  -> CLEAN pairs     : {a_conf_on_clean:.3f}  "
      f"({'FAILS -> it was confounded' if a_conf_on_clean < 0.7 else 'transfers'})")
print(f"  clean dir       -> clean test      : {a_clean_on_clean:.3f}  (sanity)")
print(f"  clean dir       -> confounded test : {a_clean_on_conf:.3f}  "
      f"({'transfers to realistic' if a_clean_on_conf > 0.7 else 'does not transfer'})")

# =====================================================================  H5
print("\n" + "=" * 72)
print("H5 — INTERNAL PROBE vs VERBALIZED self-report (on core test)")
print("=" * 72)
probe_score = Xc[te, Lc] @ d_clean_Lc
a_probe = au2(y[te], probe_score)
a_verbal = au2(y[te], verbal[te])
# correlation between internal score and verbalized p(eval)
corr = float(np.corrcoef(probe_score, verbal[te])[0, 1])
# divergence: probe confident-eval but model verbally says deploy (p_eval<0.5)
hi = probe_score > np.median(probe_score)
disagree = np.mean((hi) & (verbal[te] < 0.5))
print(f"  internal probe AUROC   = {a_probe:.3f}")
print(f"  verbalized AUROC       = {a_verbal:.3f}   (model asked directly)")
print(f"  corr(probe, verbal)    = {corr:+.2f}")
print(f"  divergence (probe-high but verbal<0.5) = {disagree:.0%}")
print("  -> internal representation is present even where the model does not verbalize it"
      if a_probe > a_verbal else "  -> verbal >= internal here")

# =====================================================================  H6
print("\n" + "=" * 72)
print("H6 — BASE-RATE / FALSE POSITIVES (probe on a realistic ~1% eval stream)")
print("=" * 72)
wc_path = os.path.join(ACTS, f"wildchat_{TAG}.npz")
if not os.path.exists(wc_path):
    print("  (wildchat acts not found — run extract_wildchat.py first)")
else:
    Xw = npz(wc_path)["X"].astype(np.float32)
    Lg = 14  # genuine mid-layer where clean signal peaks
    d_clean_g = dom_dir(Xc[tr], y[tr], Lg)
    # positives = clean eval-like test prompts; negatives = WildChat (real deployment)
    pos = Xc[(te) & (y == 1), Lg] @ d_clean_g
    neg_wc = Xw[:, Lg] @ d_clean_g
    Nneg = len(neg_wc)
    Npos = max(1, round(Nneg * 0.01 / 0.99))          # ~1% prevalence
    pos = pos[:Npos] if len(pos) >= Npos else pos
    scores = np.concatenate([neg_wc, pos])
    labels = np.concatenate([np.zeros(Nneg), np.ones(len(pos))])
    auroc_stream = roc_auc_score(labels, scores)
    print(f"  stream: {Nneg} real deploy (WildChat) + {len(pos)} eval  = {len(pos)/len(scores):.1%} prevalence")
    print(f"  AUROC on the stream = {auroc_stream:.3f}  (rank quality; base-rate-independent)")
    print("  precision at fixed recall (this is what a deployed monitor actually gets):")
    for rec in [0.5, 0.8, 0.95]:
        thr = np.quantile(pos, 1 - rec)                # threshold catching `rec` of evals
        tp = (pos >= thr).sum()
        fp = (neg_wc >= thr).sum()
        prec = tp / (tp + fp + 1e-9)
        print(f"    recall={rec:.0%}: threshold catches {tp}/{len(pos)} evals but {fp}/{Nneg} WildChat "
              f"false-positives -> precision = {prec:.1%}")
    print("  -> even a high-AUROC probe can be unreliable at realistic prevalence (many false positives).")

print("\n" + "=" * 72)
print("Done. With H1-H7 in hand, the write-up can report the full pre-registered result honestly.")
print("=" * 72)
