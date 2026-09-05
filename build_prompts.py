#!/usr/bin/env python3
"""
STEP 2 (data prep) — turn the raw EvalDetectBench + WildChat downloads into ONE clean,
labeled, balanced prompt table for probing.

Design choices (deliberate, so results are trustworthy):
  * Label by FOLDER (unambiguous): deployment_categories/* = deploy(0); benchmark evals = eval(1).
  * Keep the full input MESSAGE LIST (system+user) so the probe can apply the chat template properly.
  * Join EvalDetectBench's judge `p_eval` (black-box baseline our white-box probe must beat).
  * Dedup identical prompts (same prompt appears once per model in the trajectories).
  * Balance eval vs deploy, then a fixed-seed train/test split.

This script is CPU-only and just reads/writes JSON — run it first and READ THE REPORT it prints,
before we run anything on the GPU.

    python build_prompts.py
"""

import os
import re
import glob
import json
import random

PROJ = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(PROJ, "data")
EDB = os.path.join(DATA, "el7982__aware-bench", "collected_trajectories")
CSV_DIR = os.path.join(DATA, "el7982__aware-bench", "paper_replication")
WILDCHAT = os.path.join(DATA, "wildchat_sample.jsonl")
OUT = os.path.join(DATA, "prompts")
os.makedirs(OUT, exist_ok=True)

random.seed(0)
N_PER_CLASS = 600          # cap per class before splitting (plenty for a probe; keeps runs fast)
TEST_FRAC = 0.3
MAX_CHARS = 6000           # truncate absurdly long agentic prompts (keeps VRAM sane)


# --------------------------------------------------------------------------- helpers
def msg_to_text(content):
    """Messages store content as a string OR a list of {type:text, text:...}."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for c in content:
            if isinstance(c, dict) and c.get("type") == "text":
                parts.append(c.get("text", ""))
            elif isinstance(c, str):
                parts.append(c)
        return "\n".join(parts)
    return str(content)


def extract_input_messages(item):
    """Return the PROMPT as a list of {role, content} up to (not including) the first assistant turn."""
    inp = item.get("input")
    msgs = []
    if isinstance(inp, list):
        for m in inp:
            role = m.get("role")
            if role in ("system", "user"):
                txt = msg_to_text(m.get("content", ""))
                if txt.strip():
                    msgs.append({"role": role, "content": txt})
            elif role == "assistant":
                break
    elif isinstance(inp, str):
        msgs = [{"role": "user", "content": inp}]
    return msgs


def user_blob(msgs):
    """Flattened text (for dedup + quick reading)."""
    return " ".join(m["content"] for m in msgs)[:MAX_CHARS]


def load_trajectory_dir(patterns, label, label_name):
    """Load every json under the given glob patterns; return list of prompt dicts."""
    rows, seen = [], set()
    files = []
    for pat in patterns:
        files.extend(glob.glob(pat, recursive=True))
    for fp in files:
        base = os.path.basename(fp)
        if base in ("all.json",):          # 'all.json' duplicates the per-benchmark files
            continue
        try:
            data = json.load(open(fp, "r", errors="replace"))
        except Exception:
            continue
        if not isinstance(data, list):
            continue
        for item in data:
            msgs = extract_input_messages(item)
            if not msgs:
                continue
            blob = user_blob(msgs)
            key = re.sub(r"\s+", " ", blob).strip().lower()[:400]
            if not key or key in seen:
                continue
            seen.add(key)
            rows.append({
                "id": item.get("id", ""),
                "label": label,                 # 1 = eval, 0 = deploy
                "label_name": label_name,
                "source": os.path.splitext(base)[0],
                "messages": msgs,
                "user_text": blob,
            })
    return rows


def load_peval():
    """Map transcript_id -> judge p_eval (black-box baseline), if the CSVs are present."""
    peval = {}
    try:
        import pandas as pd
        for csv in glob.glob(os.path.join(CSV_DIR, "measure_*_scored.csv")):
            df = pd.read_csv(csv)
            id_col = "transcript_id" if "transcript_id" in df.columns else None
            if id_col and "p_eval" in df.columns:
                for _i, r in df.iterrows():
                    peval[str(r[id_col])] = float(r["p_eval"])
    except Exception as e:
        print("  (p_eval join skipped:", e, ")")
    return peval


def load_wildchat(n):
    rows = []
    if os.path.exists(WILDCHAT):
        for ln in open(WILDCHAT, errors="replace"):
            try:
                p = json.loads(ln).get("prompt")
            except Exception:
                continue
            if p and p.strip():
                rows.append({"id": "", "label": 0, "label_name": "deploy",
                             "source": "wildchat",
                             "messages": [{"role": "user", "content": p[:MAX_CHARS]}],
                             "user_text": p[:MAX_CHARS]})
            if len(rows) >= n:
                break
    return rows


# --------------------------------------------------------------------------- build
print("=" * 70)
print("BUILD PROMPTS — eval vs deployment table")
print("=" * 70)

eval_rows = load_trajectory_dir(
    [os.path.join(EDB, "*", "*.json"),                 # per-model benchmark evals
     os.path.join(EDB, "non_inspect_chat_evals", "*.json")],
    label=1, label_name="eval")
# remove anything that actually lives under deployment_categories from the eval glob
eval_rows = [r for r in eval_rows if r["source"] not in {"author_chats", "hh_rlhf_redteam",
             "toxic_chat", "crowd_sourced_code", "dataclaw", "codechat", "traced_run",
             "wildbench", "open_deep_research", "reddit_wayback", "openhands", "cdial_bias"}]

deploy_rows = load_trajectory_dir(
    [os.path.join(EDB, "deployment_categories", "*.json")],
    label=0, label_name="deploy")
deploy_rows += load_wildchat(400)

print(f"\nraw unique prompts:  eval={len(eval_rows)}  deploy={len(deploy_rows)}")
print("eval sources   :", sorted({r['source'] for r in eval_rows}))
print("deploy sources :", sorted({r['source'] for r in deploy_rows}))

# attach p_eval baseline
peval = load_peval()
hit = 0
for r in eval_rows + deploy_rows:
    v = peval.get(str(r["id"]))
    r["p_eval_blackbox"] = v
    if v is not None:
        hit += 1
print(f"p_eval baseline joined for {hit}/{len(eval_rows)+len(deploy_rows)} prompts",
      "(0 is fine — we can compute our own black-box baseline by asking Qwen)")

# balance + split
random.shuffle(eval_rows); random.shuffle(deploy_rows)
n = min(N_PER_CLASS, len(eval_rows), len(deploy_rows))
eval_rows, deploy_rows = eval_rows[:n], deploy_rows[:n]
print(f"\nbalanced to {n} per class")

def split(rows):
    k = int(len(rows) * (1 - TEST_FRAC))
    return rows[:k], rows[k:]

etr, ete = split(eval_rows)
dtr, dte = split(deploy_rows)
train, test = etr + dtr, ete + dte
random.shuffle(train); random.shuffle(test)

for name, rows in [("train", train), ("test", test)]:
    fp = os.path.join(OUT, f"{name}.jsonl")
    with open(fp, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    ne = sum(x["label"] for x in rows)
    print(f"  wrote {fp}: {len(rows)} rows ({ne} eval / {len(rows)-ne} deploy)")

# --------------------------------------------------------------------------- verify
print("\n" + "=" * 70)
print("VERIFICATION — read these before we probe")
print("=" * 70)
print(f"train={len(train)}  test={len(test)}  (balanced, seed=0)")
print("\n--- example EVAL prompt ---")
ex_e = next(r for r in train if r["label"] == 1)
print("source:", ex_e["source"], "| p_eval:", ex_e["p_eval_blackbox"])
print(ex_e["user_text"][:400])
print("\n--- example DEPLOY prompt ---")
ex_d = next(r for r in train if r["label"] == 0)
print("source:", ex_d["source"], "| p_eval:", ex_d["p_eval_blackbox"])
print(ex_d["user_text"][:400])
print("\nIf eval looks like benchmark/test prompts and deploy looks like real user chat,")
print("the labels are right and we can build the probe next.")
