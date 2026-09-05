# Eval-Lens

Reading whether a language model knows it is being tested, from its internal activations. This is a small, content-controlled audit of that question, done on one 7B model on a 6 GB laptop.

## What I found
- A high probe score is not proof. The all-factors probe reached AUROC 1.000, but a random direction reached 0.986, so the classes separate on the surface alone.
- There is a real signal, but only for specific length-neutral cues at mid layers (the stakes cue: 0.924 against a random 0.661).
- It is not one direction (average cue cosine 0.17).
- It is unreliable as a monitor: at a realistic 1% rate of evals, precision is only 24 to 42 percent.

So the honest answer is: it partly works, with caveats.

## Method
A difference-of-means probe on last-token residual-stream activations of Qwen2.5-7B-Instruct (4-bit). Pre-registered, split by task, with three baselines: a random direction, a length-only classifier, and simply asking the model. Details in `PREREGISTRATION.md`.

## Datasets
- EvalAwareBench (content-controlled): `aisa-group/EvalAwareBench`
- EvalDetectBench (realistic): `el7982/aware-bench`
- WildChat (deployment stream): `allenai/WildChat-1M`

## How to run
1. `pip install -r requirements.txt`
2. `python fetch_data.py` — download the datasets
3. `python build_prompts.py` and `python build_confound.py` — build the prompt sets
4. `python extract_probe.py`, `python extract_confound.py`, `python extract_wildchat.py` — extract activations (GPU)
5. `python analyze_confound.py` and `python analyze_extra.py` — the main analysis
6. `python causal_steer.py` and `python make_figures.py` — the causal test and the figures

Runs on a single 6 GB GPU in 4-bit. No model downloads beyond Hugging Face.
