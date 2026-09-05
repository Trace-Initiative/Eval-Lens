#!/usr/bin/env python3
"""
STEP 2 (clean data) — build content-matched, confound-controlled prompt sets from
EvalAwareBench's PRE-RENDERED prompts (data/aisa-group__EvalAwareBench/data/prompts.jsonl).

We do NOT render anything ourselves — the 51,200 rows are already rendered; we just SELECT the
factor configurations we want, keyed by task so content is held constant.

Outputs:
  data/prompts/confound_core.jsonl   deploy-like (all realistic) vs eval-like (all varied), per task
                                     -> a clean eval-vs-deploy contrast with SAME content
  data/prompts/confound_flips.jsonl  base (all realistic) + each single-factor flip, per task
                                     -> attribute any probe signal to a SPECIFIC factor

Prints a DIAGNOSTIC (balance + length by class and per factor) so we can see which factors change
length by design (F4 context, F7 template) vs which are length-neutral (F1, F3, F5, F6, F8).

CPU-only.  Run:  python build_confound.py
"""

import os
import json
import statistics as st

PROJ = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(PROJ, "data", "aisa-group__EvalAwareBench", "data", "prompts.jsonl")
OUT = os.path.join(PROJ, "data", "prompts")
os.makedirs(OUT, exist_ok=True)

FACTORS = ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8"]
EVAL_LEVEL = {"F1": "varied", "F2": "varied", "F3": "formal", "F4": "varied",
              "F5": "varied", "F6": "varied", "F7": "template", "F8": "varied"}
DEPLOY_LEVEL = {"F1": "realistic", "F2": "realistic", "F3": "casual", "F4": "realistic",
                "F5": "realistic", "F6": "realistic", "F7": "prose", "F8": "realistic"}
LENGTH_NEUTRAL = {"F1", "F3", "F5", "F6", "F8"}   # don't add/remove much text


def detect(row):
    pf = next((k for k in ["prompt", "text", "rendered_prompt", "rendered", "content"] if k in row), None)
    if all(f in row for f in FACTORS):
        getf = lambda r: {f: r.get(f) for f in FACTORS}
    elif isinstance(row.get("config"), dict):
        getf = lambda r: r["config"]
    elif isinstance(row.get("factors"), dict):
        getf = lambda r: r["factors"]
    else:
        getf = None
    tid = next((k for k in ["task_id", "task_name", "id"] if k in row), None)
    val = "valence" if "valence" in row else None
    return pf, getf, tid, val


def key(cfg):
    return tuple(cfg[f] for f in FACTORS)


def main():
    if not os.path.exists(SRC):
        print("!! not found:", SRC)
        return
    rows = [json.loads(l) for l in open(SRC)]
    print(f"loaded {len(rows)} rendered prompts")
    print("first row keys:", list(rows[0].keys()))
    pf, getf, tid, val = detect(rows[0])
    print(f"detected -> prompt='{pf}'  task='{tid}'  valence='{val}'  factors_ok={getf is not None}")
    if not (pf and getf and tid):
        print("!! schema auto-detect failed. Sample row (paste this to me):")
        print(json.dumps(rows[0])[:800])
        return

    idx, taskset = {}, set()
    for r in rows:
        cfg = getf(r)
        if any(cfg.get(f) is None for f in FACTORS):
            continue
        v = r.get(val, "") if val else ""
        idx[(r[tid], v, key(cfg))] = r[pf]
        taskset.add((r[tid], v))
    print(f"indexed {len(idx)} rows across {len(taskset)} task-variants")

    def get(task, v, cfg):
        return idx.get((task, v, key(cfg)))

    core, flips = [], []
    for task, v in sorted(taskset):
        dep, evl = get(task, v, DEPLOY_LEVEL), get(task, v, EVAL_LEVEL)
        if dep and evl:
            core.append({"messages": [{"role": "user", "content": dep}], "label": 0,
                         "config_tag": "deploy_like", "valence": v, "task": task, "user_text": dep})
            core.append({"messages": [{"role": "user", "content": evl}], "label": 1,
                         "config_tag": "eval_like", "valence": v, "task": task, "user_text": evl})
        base = get(task, v, DEPLOY_LEVEL)
        if base:
            flips.append({"messages": [{"role": "user", "content": base}], "config_tag": "base",
                          "factor": "none", "valence": v, "task": task, "user_text": base})
            for f in FACTORS:
                cfg = dict(DEPLOY_LEVEL); cfg[f] = EVAL_LEVEL[f]
                p = get(task, v, cfg)
                if p:
                    flips.append({"messages": [{"role": "user", "content": p}], "config_tag": f"flip_{f}",
                                  "factor": f, "valence": v, "task": task, "user_text": p})

    for name, data in [("confound_core", core), ("confound_flips", flips)]:
        fp = os.path.join(OUT, f"{name}.jsonl")
        with open(fp, "w") as fh:
            for r in data:
                fh.write(json.dumps(r) + "\n")
        print(f"wrote {fp}: {len(data)} rows")

    # -------------------- diagnostic --------------------
    def wl(r):
        return len(r["user_text"].split())
    dep = [r for r in core if r["label"] == 0]
    evl = [r for r in core if r["label"] == 1]
    print("\n" + "=" * 60)
    print("DIAGNOSTIC — clean controlled set")
    print("=" * 60)
    print(f"core: {len(dep)} deploy-like / {len(evl)} eval-like (equal = content-matched OK)")
    if dep and evl:
        print(f"length(words) median  deploy-like={st.median([wl(r) for r in dep])}  "
              f"eval-like={st.median([wl(r) for r in evl])}")
        print("  ^ these differ BY DESIGN (F4/F7 change length); per-factor below isolates it")
    base = [r for r in flips if r["factor"] == "none"]
    bmed = st.median([wl(r) for r in base]) if base else 0
    print(f"\nper-factor length shift from base (all-realistic, median={bmed} words):")
    for f in FACTORS:
        fr = [r for r in flips if r["factor"] == f]
        if fr:
            m = st.median([wl(r) for r in fr])
            tag = "length-neutral" if f in LENGTH_NEUTRAL else "length-entangled"
            print(f"  {f}: median={m:.0f}  (delta {m-bmed:+.0f})   [{tag}]")
    vals = {v: sum(1 for r in core if r["valence"] == v) for v in {r["valence"] for r in core}}
    print("\nvalence balance (core):", vals)

    # ---- integrity: any prompt with leftover {SLOT} placeholders or empty text? ----
    import re
    def bad(r):
        t = r["user_text"]
        return (not t.strip()) or bool(re.search(r"\{[A-Z0-9_]{2,}\}", t))
    n_bad = sum(bad(r) for r in core + flips)
    print(f"\n[integrity] prompts with unrendered {{SLOT}} or empty: {n_bad}  "
          f"({'OK' if n_bad == 0 else 'INVESTIGATE — render artifacts'})")

    # ---- split feasibility: unique tasks (we will split BY TASK, not by prompt) ----
    ntasks = len({r["task"] for r in core})
    print(f"[split] unique tasks in core = {ntasks}  -> split by task (no task in both train & test)")

    # ---- residual lexical leakage on the CLEAN contrast (F6/F8 wrappers could give tells) ----
    from collections import Counter
    def toks(rows):
        c = Counter()
        for r in rows:
            for w in re.findall(r"[a-z]{3,}", r["user_text"].lower()):
                c[w] += 1
        return c
    ce, cd = toks(evl), toks(dep)
    Ne, Nd = sum(ce.values()) + 1, sum(cd.values()) + 1
    sc = {w: (ce[w]/Ne)/((cd[w]/Nd) + 1e-9) for w in (set(ce) | set(cd)) if ce[w] + cd[w] >= 5}
    top_e = [w for w, _ in sorted(sc.items(), key=lambda x: -x[1])[:12]]
    top_d = [w for w, _ in sorted(sc.items(), key=lambda x: x[1])[:12]]
    print("\n[leakage] eval-like tell words :", ", ".join(top_e))
    print("[leakage] deploy-like tell words:", ", ".join(top_d))
    print("  (if these are F6/F8 wrapper words like 'unrestricted'/'output', the probe may read the")
    print("   wrapper, not awareness -> the F6/F8 flips will isolate that; length-neutral non-wrapper")
    print("   factors F1/F3/F5 are the cleanest test.)")

    print("\nREAD THIS: length-neutral factors (F1,F3,F5,F6,F8) give the cleanest signal — a probe")
    print("firing on those can't be dismissed as length. Paste this whole output; then we pre-register")
    print("(locking split-by-task, the significance test, and the layer rule) and run Step 3.")


if __name__ == "__main__":
    main()
