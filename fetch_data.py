#!/usr/bin/env python3
"""
STEP 1 (data) — download AND inspect the eval-awareness datasets.

We deliberately DOWNLOAD + PRINT THE STRUCTURE (files, columns, a couple of example
rows) rather than assume a schema, so we can confirm each dataset really contains
labeled eval-vs-deployment prompts before we build experiments on it (Step 2).

Saves everything under ./data/. Nothing here needs a GPU.
Run inside the venv:  python fetch_data.py
"""

import os
import json

PROJ = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(PROJ, "data")
os.makedirs(DATA, exist_ok=True)


def peek_dir(d):
    """List files in a downloaded dataset dir and peek at the first json/jsonl/csv."""
    files = []
    for root, _dirs, fs in os.walk(d):
        for f in fs:
            files.append(os.path.join(root, f))
    print(f"    {len(files)} files. sample:")
    for f in sorted(files)[:15]:
        try:
            kb = os.path.getsize(f) // 1024
        except OSError:
            kb = 0
        print(f"      {os.path.relpath(f, d)}  ({kb} KB)")
    for f in sorted(files):
        if f.lower().endswith((".json", ".jsonl", ".csv")):
            print(f"    --- peek: {os.path.relpath(f, d)} ---")
            try:
                with open(f, "r", errors="replace") as fh:
                    for i, ln in enumerate(fh):
                        if i >= 2:
                            break
                        print("      " + ln.strip()[:300])
            except Exception as e:
                print("      (could not read:", e, ")")
            break


def snapshot(repo_id):
    """Pull a HF dataset repo's raw files (robust to odd schemas) and inspect it."""
    from huggingface_hub import snapshot_download
    print(f"\n>>> dataset repo: {repo_id}")
    try:
        p = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=os.path.join(DATA, repo_id.replace("/", "__")),
        )
        print("    saved to:", p)
        peek_dir(p)
        return repo_id
    except Exception as e:
        print(f"    [FAILED] {repo_id}: {type(e).__name__}: {e}")
        print("    -> may need `huggingface-cli login` or accepting terms on the dataset's HF page.")
        return None


# --- eval / deployment datasets (we only need one or two of these to work) ---
EVAL_REPOS = [
    "jjpn2/eval_awareness",       # canonical: "LLMs Often Know When They Are Being Evaluated"
    "aisa-group/EvalAwareBench",  # decomposes eval-awareness (format vs context) -> confound test
    "el7982/aware-bench",         # EvalDetectBench
]
downloaded = [r for r in (snapshot(r) for r in EVAL_REPOS) if r]

# --- optional: WildChat sample = realistic deployment stream (base-rate test) ---
print("\n>>> optional: WildChat-1M sample (deployment stream for the 1% base-rate test)")
try:
    from datasets import load_dataset
    ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
    out = os.path.join(DATA, "wildchat_sample.jsonl")
    n = 0
    with open(out, "w") as fh:
        for ex in ds:
            conv = ex.get("conversation") or ex.get("messages") or []
            first_user = next((m.get("content") for m in conv if m.get("role") == "user"), None)
            if first_user:
                fh.write(json.dumps({"prompt": first_user}) + "\n")
                n += 1
            if n >= 2000:
                break
    print(f"    saved {n} deployment prompts -> {out}")
except Exception as e:
    print(f"    [skipped] WildChat ({type(e).__name__}: {e})")
    print("    -> fine: we can use the deployment-labeled prompts inside the eval datasets instead.")

# --- summary --------------------------------------------------------------
print("\n== FETCH SUMMARY ==")
print("eval datasets OK :", downloaded if downloaded else "NONE (check hf login / dataset terms)")
print("data dir         :", DATA)
print("\nNext: paste this output back. Once at least one eval dataset downloaded and its")
print("structure looks like labeled eval-vs-deployment prompts, we start Step 2 (experiments).")
