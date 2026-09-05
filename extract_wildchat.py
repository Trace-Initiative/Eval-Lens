#!/usr/bin/env python3
"""
STEP 3 (H6 extract) — GPU. Extract last-token activations for the WildChat deployment stream,
used by the base-rate / false-positive test (analyze_extra.py). Saves data/acts/wildchat_7b.npz.

  python extract_wildchat.py --smoke   # 20 rows
  python extract_wildchat.py           # full (~2000, ~12 min)
"""

import os
import json
import time
import argparse
import numpy as np

PROJ = os.path.dirname(os.path.abspath(__file__))
WC = os.path.join(PROJ, "data", "wildchat_sample.jsonl")
ACTS = os.path.join(PROJ, "data", "acts")
os.makedirs(ACTS, exist_ok=True)
MODEL = "Qwen/Qwen2.5-7B-Instruct"
MAX_TOKENS = 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--tag", default="7b")
    ap.add_argument("--limit", type=int, default=2000)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    rows = [json.loads(l) for l in open(WC)]
    rows = [r for r in rows if r.get("prompt")][: (20 if args.smoke else args.limit)]
    print(f"[load] {args.model} (4-bit); {len(rows)} WildChat prompts")
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.truncation_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=bnb,
                                                 device_map={"": 0}).eval()

    A = []
    t0 = time.time()
    for i, r in enumerate(rows):
        msgs = [{"role": "user", "content": r["prompt"]}]
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt",
                                      return_dict=True, truncation=True, max_length=MAX_TOKENS).to(model.device)
        with torch.no_grad():
            hs = model(**enc, output_hidden_states=True).hidden_states
        A.append(np.stack([hs[L][0, -1].float().cpu().numpy() for L in range(len(hs))]).astype(np.float16))
        if (i + 1) % 50 == 0 or (args.smoke and i + 1 == len(rows)):
            print(f"  {i+1}/{len(rows)}  {(time.time()-t0)/(i+1):.2f}s/prompt")
    X = np.stack(A)
    if args.smoke:
        print("smoke shape:", X.shape, "\nSMOKE OK")
        return
    np.savez_compressed(os.path.join(ACTS, f"wildchat_{args.tag}.npz"), X=X)
    print(f"[save] {ACTS}/wildchat_{args.tag}.npz  X={X.shape}")
    print("DONE — next: analyze_extra.py (H5 internal-vs-verbal, H6 base-rate, H7 cross-dataset)")


if __name__ == "__main__":
    main()
