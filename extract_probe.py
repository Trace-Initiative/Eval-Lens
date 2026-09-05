#!/usr/bin/env python3
"""
STEP 2 — extract residual-stream activations for the eval/deploy prompts, fit the
eval-awareness probe (difference-of-means), and evaluate it HELD-OUT against a
random-direction placebo baseline.

Why it's built this way (grounded in Neel's stated methodology):
  * "Start by doing the obvious thing" -> difference-of-means direction, not SAEs.
  * "Compare to baselines (replace your vector with a random one)" -> random-direction placebo.
  * "Report held-out results; most exciting results are false" -> AUROC on the TEST split only.
  * Save activations once -> every later experiment (confound, base-rate FP, internal-vs-verbalized)
    reuses them, so we run the GPU a single time.

Usage:
  python extract_probe.py --smoke     # ~20 prompts, ~1-2 min: confirms model + hooks work on the 4050
  python extract_probe.py             # full run: extract + save activations + probe report
  python extract_probe.py --model Qwen/Qwen2.5-3B-Instruct --tag 3b   # scale check (stretch)
"""

import os
import json
import time
import argparse
import numpy as np

PROJ = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(PROJ, "data")
PROMPTS = os.path.join(DATA, "prompts")
ACTS = os.path.join(DATA, "acts")
OUTP = os.path.join(DATA, "probe")
os.makedirs(ACTS, exist_ok=True)
os.makedirs(OUTP, exist_ok=True)

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
MAX_TOKENS = 1024          # cap prompt length (VRAM safety on 6 GB); long agentic prompts get truncated
N_PLACEBO = 30             # random directions for the baseline


def load_prompts(split, limit=None):
    fp = os.path.join(PROMPTS, f"{split}.jsonl")
    rows = [json.loads(l) for l in open(fp)]
    return rows[:limit] if limit else rows


def cohens_d(a, b):
    na, nb = len(a), len(b)
    va, vb = a.var(ddof=1), b.var(ddof=1)
    sp = np.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2) + 1e-12)
    return (a.mean() - b.mean()) / sp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="run on 20 prompts to verify the pipeline")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--tag", default="7b", help="filename tag for saved activations")
    args = ap.parse_args()
    smoke = args.smoke
    lim = 20 if smoke else None

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from sklearn.metrics import roc_auc_score

    print(f"[load] {args.model} (4-bit nf4) ...")
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.truncation_side = "left"        # keep the END (generation point) for last-token probing
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=bnb, device_map={"": 0}).eval()
    print(f"  layers={model.config.num_hidden_layers}  hidden={model.config.hidden_size}")

    def extract(rows, name):
        A, y, srcs, ntok = [], [], [], []
        t0 = time.time()
        for i, r in enumerate(rows):
            enc = tok.apply_chat_template(
                r["messages"], add_generation_prompt=True, return_tensors="pt",
                return_dict=True, truncation=True, max_length=MAX_TOKENS).to(model.device)
            with torch.no_grad():
                hs = model(**enc, output_hidden_states=True).hidden_states  # tuple len L+1
            last = np.stack([hs[L][0, -1].float().cpu().numpy() for L in range(len(hs))])
            A.append(last); y.append(int(r["label"])); srcs.append(r.get("source", ""))
            ntok.append(int(enc["input_ids"].shape[1]))
            if (i + 1) % 25 == 0 or (smoke and i + 1 == len(rows)):
                print(f"  [{name}] {i+1}/{len(rows)}  {(time.time()-t0)/(i+1):.2f}s/prompt")
        return np.stack(A).astype(np.float32), np.array(y), srcs, np.array(ntok)

    print("[extract] train ...")
    Xtr, ytr, str_src, str_tok = extract(load_prompts("train", lim), "train")
    print("[extract] test ...")
    Xte, yte, ste_src, ste_tok = extract(load_prompts("test", lim), "test")

    print(f"\n[tokens] train max={str_tok.max()} median={int(np.median(str_tok))}"
          f" | truncated@{MAX_TOKENS}: {(str_tok>=MAX_TOKENS).sum()} train, {(ste_tok>=MAX_TOKENS).sum()} test")

    if not smoke:
        np.savez_compressed(os.path.join(ACTS, f"train_{args.tag}.npz"),
                            X=Xtr, y=ytr, src=np.array(str_src), ntok=str_tok)
        np.savez_compressed(os.path.join(ACTS, f"test_{args.tag}.npz"),
                            X=Xte, y=yte, src=np.array(ste_src), ntok=ste_tok)
        print("[save] activations ->", ACTS)

    # ---------- probe: diff-of-means per layer + random-direction placebo ----------
    def probe_layer(L):
        etr, dtr = Xtr[ytr == 1, L], Xtr[ytr == 0, L]
        d = etr.mean(0) - dtr.mean(0)
        d /= (np.linalg.norm(d) + 1e-8)
        sc = Xte[:, L] @ d
        au = roc_auc_score(yte, sc)
        return au, d, sc

    def placebo_layer(L, n=N_PLACEBO, seed=0):
        rng = np.random.default_rng(seed)
        H = Xte.shape[2]
        aus = []
        for _ in range(n):
            r = rng.standard_normal(H); r /= (np.linalg.norm(r) + 1e-8)
            a = roc_auc_score(yte, Xte[:, L] @ r)
            aus.append(max(a, 1 - a))          # best random orientation
        aus = np.array(aus)
        return aus.mean(), aus.max()

    print("\n" + "=" * 72)
    print("PROBE — diff-of-means eval-awareness direction  (held-out AUROC vs random placebo)")
    print("=" * 72)
    results = []
    for L in range(1, Xtr.shape[1]):
        au, d, sc = probe_layer(L)
        pm, pmax = placebo_layer(L)
        results.append((L, au, pm, pmax))
        flag = "  <<< real >> placebo" if (au > pmax and au > 0.7) else ""
        print(f"  L{L:2d}  AUROC={au:.3f}   placebo(mean={pm:.3f}, max={pmax:.3f}){flag}")

    best_L, best_au, _, best_pmax = max(results, key=lambda r: r[1])
    au, dbest, scbest = probe_layer(best_L)
    d = cohens_d(scbest[yte == 1], scbest[yte == 0])
    print("\n" + "-" * 72)
    print(f"BEST layer L{best_L}:  held-out AUROC={best_au:.3f}   "
          f"placebo-max={best_pmax:.3f}   Cohen's d={d:.2f}")
    verdict = ("the probe genuinely separates eval vs deploy and beats random"
               if best_au > best_pmax and best_au > 0.7 else
               "WEAK / not clearly above random — that itself is a finding, report honestly")
    print("VERDICT:", verdict)

    if not smoke:
        np.savez(os.path.join(OUTP, f"direction_{args.tag}.npz"),
                 layer=best_L, direction=dbest)
        json.dump({"model": args.model, "best_layer": int(best_L),
                   "auroc": float(best_au), "placebo_max": float(best_pmax),
                   "cohens_d": float(d)},
                  open(os.path.join(OUTP, f"probe_summary_{args.tag}.json"), "w"), indent=2)
        print("[save] direction + summary ->", OUTP)

    print("\nSMOKE OK — pipeline works, now run without --smoke" if smoke else "\nDONE")


if __name__ == "__main__":
    main()
