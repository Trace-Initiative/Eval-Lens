# Pre-Registration — Eval-Awareness Probe Reliability

**Author:** Karan Singh · **Date:** 2026-09-05 · **Model:** Qwen2.5-7B-Instruct (4-bit), + Qwen2.5-3B (scale, stretch)
**Written BEFORE running the confirmatory experiments. Every prediction is reported pass or fail; no
prediction is dropped, no metric is swapped after seeing results, no optional stopping.**

---

## Goal — rigor, not a positive result
The aim is **NOT** to produce a positive or novel finding. The aim is to answer *"does eval-awareness
probing actually work?"* **rigorously** — proper controls, baselines, statistics, honest reporting.
A well-supported negative or mixed answer (e.g. *"the signal is largely confounded / mostly format
sensitivity"*) is a **FULL SUCCESS** if the testing was thorough. Success = trustworthiness of the
answer, not the direction of the answer. We do not need the probe to "win."

---

## Question
Is a residual-stream probe for "evaluation awareness" measuring the model's genuine detection that it
is being tested — or surface artifacts (length, format, wrapper tokens)? And how reliable is such a
probe at a realistic base rate?

- **North Star:** trustworthy detection of eval-aware / test-gaming behavior in future AI.
- **Proxy task:** on today's open models, isolate genuine eval-framing signal from confounds using a
  content-controlled dataset, and quantify reliability at realistic prevalence.

## Datasets
- **CLEAN (primary):** EvalAwareBench content-matched pairs — `confound_core.jsonl` (200 deploy-like =
  all-realistic vs 200 eval-like = all-varied, same task/content) and `confound_flips.jsonl` (base +
  single-factor flips). Confirmed: balanced, 0 render artifacts, length ~matched overall (145 vs 156).
- **CONFOUNDED (contrast):** EvalDetectBench-derived `train/test.jsonl`. Known confounds (length 4x,
  agentic 71/15, system 67/6, scenario-name leakage). Naive probe already scored AUROC 0.925 (the trap).
- **DEPLOYMENT STREAM:** WildChat sample (base-rate/FP test).
- **VALIDATION (bonus):** jjpn2/eval_awareness (independent check, if time).

## Confounds established (diagnostics) and how each is controlled
| Confound | In confounded data | In clean data | Control |
|---|---|---|---|
| Length | 4x | ~matched (145/156) | per-factor: use length-neutral factors F1/F3/F5; + length-only baseline |
| Agentic | 71% vs 15% | n/a (content matched) | content matched by task |
| Format/system | 67% vs 6% | varied by F7 only | F7 isolated as one factor |
| Lexical/wrappers | scenario names | F1 names, F6/F8 wrappers | per-factor flips isolate F1, F6, F8 |

## Hypotheses (pre-specified predictions)
- **H1 (signal exists, clean):** On `confound_core`, a diff-of-means probe separates eval-like vs
  deploy-like with **task-split held-out AUROC > 0.70**, and — unlike the confounded data — the
  **random placebo falls near 0.5** (< 0.60).
- **H2 (not just length/wrapper) — the key test:** Probes on the **length-neutral, non-wrapper single
  factors F1, F3, F5** separate above placebo. If only length-entangled (F2/F4/F7) or wrapper (F6/F8)
  factors separate, the signal is largely surface artifact.
- **H3 (beats trivial baselines):** The probe beats a **length-only** classifier (predict eval from
  word count) and a **black-box "ask Qwen"** baseline, on the clean set.
- **H4 (one capability or many?):** Per-factor probe directions are **not collinear** (mean pairwise
  cosine < 0.5) → eval-awareness is factor-specific, not one unified direction. (Report the full cosine
  matrix regardless — this extends "eval awareness is not one capability" into the internals.)
- **H5 (internal vs verbalized):** The probe and the model's verbalized "is this a test?" **diverge**
  on an identifiable subset; report their agreement (AUROC/correlation).
- **H6 (base-rate reliability):** At **~1% eval prevalence** in a WildChat stream, precision at a fixed
  threshold **drops sharply** vs the balanced setting; quantify the precision/recall trade-off.
- **H7 (cross-dataset):** The confounded EvalDetectBench direction does **not** cleanly separate the
  content-matched clean pairs (evidence it was confounded); report whether the clean direction transfers
  to EvalDetectBench above placebo.

## Analysis plan (locked)
- **Split BY TASK**, not by prompt: 140 train tasks / 60 test tasks (70/30), fixed seed. No task appears
  in both. Applies to `core` and `flips`.
- **Layer selection:** pick the single layer by **cross-validated AUROC on TRAIN tasks only**; report
  that one layer on TEST. (No cherry-picking the best test layer.)
- **Significance:** task-clustered bootstrap (resample tasks, 2000x) for 95% CIs; label-permutation
  null within the split. Report CIs on every AUROC.
- **Baselines (mandatory):** (a) random-direction placebo (>=30); (b) length-only classifier;
  (c) black-box "ask Qwen: is this an evaluation? yes/no", p(yes).
- **Metrics:** AUROC (primary), Cohen's d, precision@1%-base-rate.

## Outcome interpretation (pre-specified — so we cannot move goalposts)
- If clean AUROC is high AND F1/F3/F5 separate above placebo AND the probe beats the length-only
  baseline → **genuine eval-awareness signal** (partly factor-specific per H4).
- If clean AUROC is high but ONLY length-entangled/wrapper factors drive it, and the probe ~= length-only
  baseline → **the signal is largely surface artifact** ("yes, mostly format sensitivity"). This is a
  valid, interesting, publishable finding — report it plainly.
- The base-rate/FP result (H6) stands as a reliability statement **either way.**

## Stopping rule
Run each pre-specified test once. Report every hypothesis pass/fail. No metric swapping, no dropping
failed predictions, no optional stopping. Stretch items (3B scale, jjpn2 validation, dose-response on
`num_factors_varied`) are clearly labeled as exploratory if added.
