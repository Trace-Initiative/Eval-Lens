#!/usr/bin/env python3
"""
STEP 2a — DIAGNOSTIC (no GPU). Understand the factors & confounds in the eval/deploy data
BEFORE we probe, so we know what a probe result would really mean.

This follows Neel's Explore -> Understand -> Distill: gain surface area on the data first.
It answers: is the eval-vs-deploy label secretly confounded with something trivial
(length, agentic-ness, having a system prompt, a few give-away words)? Those are the things a
probe could latch onto instead of "eval-awareness", so we must see them now.

    python analyze_factors.py
"""

import os
import json
import re
import statistics as st
from collections import Counter

PROJ = os.path.dirname(os.path.abspath(__file__))
PROMPTS = os.path.join(PROJ, "data", "prompts")

# crude agentic cue: agent/tool framing in the prompt
AGENTIC_CUES = ["you have access to", "bash tool", "tool call", "function call", "```bash",
                "run shell", "you are a", "agent", "tools available", "sqlite3", "jq "]
AGENTIC_SOURCES = {"petri", "agentharm", "agentic_misalignment", "swebench_verified_mini",
                   "gaia_level2", "cybench", "openhands", "open_deep_research", "traced_run",
                   "nl2bash", "browse_comp"}


def load(split):
    return [json.loads(l) for l in open(os.path.join(PROMPTS, f"{split}.jsonl"))]


def q(vals):
    vals = sorted(vals)
    return (min(vals), st.median(vals), max(vals),
            vals[int(len(vals) * 0.25)], vals[int(len(vals) * 0.75)])


def describe(rows, tag):
    print("\n" + "=" * 72)
    print(f"FACTOR DIAGNOSTIC — {tag}  (n={len(rows)})")
    print("=" * 72)
    ev = [r for r in rows if r["label"] == 1]
    dp = [r for r in rows if r["label"] == 0]
    print(f"eval={len(ev)}  deploy={len(dp)}")

    # 1) length confound
    def wc(r):
        return len(r["user_text"].split())
    e_len, d_len = [wc(r) for r in ev], [wc(r) for r in dp]
    print("\n[1] LENGTH (words)  -- if eval and deploy differ a lot, length is a confound")
    print(f"    eval  : min/med/max = {q(e_len)[0]}/{q(e_len)[1]}/{q(e_len)[2]}   IQR {q(e_len)[3]}-{q(e_len)[4]}")
    print(f"    deploy: min/med/max = {q(d_len)[0]}/{q(d_len)[1]}/{q(d_len)[2]}   IQR {q(d_len)[3]}-{q(d_len)[4]}")
    ratio = (st.median(e_len) + 1) / (st.median(d_len) + 1)
    print(f"    median ratio eval/deploy = {ratio:.2f}   -> {'CONFOUND: control length' if ratio>1.5 or ratio<0.67 else 'ok, similar'}")

    # 2) agentic confound
    def is_agentic(r):
        if r["source"] in AGENTIC_SOURCES:
            return True
        t = r["user_text"].lower()
        return any(c in t for c in AGENTIC_CUES)
    e_ag = sum(is_agentic(r) for r in ev) / max(1, len(ev))
    d_ag = sum(is_agentic(r) for r in dp) / max(1, len(dp))
    print("\n[2] AGENTIC fraction (heuristic)  -- if eval is much more agentic, that's a confound")
    print(f"    eval agentic  = {e_ag:.0%}    deploy agentic = {d_ag:.0%}")
    print(f"    -> {'CONFOUND: control agentic-vs-chat (also join is_agentic from the CSV)' if abs(e_ag-d_ag)>0.25 else 'ok, similar'}")

    # 3) system-prompt / format confound
    def has_sys(r):
        return any(m["role"] == "system" for m in r["messages"])
    e_sys = sum(has_sys(r) for r in ev) / max(1, len(ev))
    d_sys = sum(has_sys(r) for r in dp) / max(1, len(dp))
    print("\n[3] HAS SYSTEM PROMPT  -- a format cue the probe could exploit")
    print(f"    eval = {e_sys:.0%}    deploy = {d_sys:.0%}"
          f"   -> {'CONFOUND: format differs' if abs(e_sys-d_sys)>0.25 else 'ok, similar'}")

    # 4) p_eval baseline distribution
    e_pe = [r["p_eval_blackbox"] for r in ev if r.get("p_eval_blackbox") is not None]
    d_pe = [r["p_eval_blackbox"] for r in dp if r.get("p_eval_blackbox") is not None]
    print("\n[4] p_eval (black-box judge baseline)  -- how well the JUDGE already separates them")
    if e_pe and d_pe:
        print(f"    eval  mean p_eval = {st.mean(e_pe):.2f}  (n={len(e_pe)})")
        print(f"    deploy mean p_eval= {st.mean(d_pe):.2f}  (n={len(d_pe)})")
        print("    -> our white-box probe should aim to MATCH or BEAT this separation")
    else:
        print("    (insufficient p_eval coverage; we'll compute our own baseline by asking Qwen)")

    # 5) lexical leakage: words most over-represented in eval vs deploy
    def toks(rows):
        c = Counter()
        for r in rows:
            for w in re.findall(r"[a-z]{3,}", r["user_text"].lower()):
                c[w] += 1
        return c
    ce, cd = toks(ev), toks(dp)
    Ne, Nd = sum(ce.values()) + 1, sum(cd.values()) + 1
    scores = {}
    for w in set(ce) | set(cd):
        if ce[w] + cd[w] < 5:
            continue
        scores[w] = (ce[w] / Ne) / ((cd[w] / Nd) + 1e-9)
    top_eval = sorted(scores.items(), key=lambda x: -x[1])[:15]
    top_dep = sorted(scores.items(), key=lambda x: x[1])[:15]
    print("\n[5] SURFACE-WORD LEAKAGE  -- if labels are separable by a few give-away words,")
    print("    a probe may exploit lexical cues, not 'eval-awareness'. Watch for obvious tells.")
    print("    eval-tell words :", ", ".join(w for w, _ in top_eval))
    print("    deploy-tell words:", ", ".join(w for w, _ in top_dep))


def main():
    tr = load("train")
    describe(tr, "TRAIN")
    describe(load("test"), "TEST")
    print("\n" + "=" * 72)
    print("WHAT TO DO WITH THIS")
    print("=" * 72)
    print("Any factor flagged CONFOUND must be controlled in Exp 3 (e.g. match on length, or")
    print("test the probe within agentic-only / chat-only subsets, or use EvalAwareBench's")
    print("format-controlled pairs). Obvious give-away words tell us to check the probe isn't")
    print("just doing lexical matching. Paste this output and we lock the confirmatory design.")


if __name__ == "__main__":
    main()
