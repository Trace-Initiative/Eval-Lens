# Eval-Lens

Reading whether a language model knows it is being tested, from its internal activations. This is a small, content-controlled audit of that question, done on one 7B model on a 6 GB laptop.

## Paper
This repository holds the first study behind the preprint:

Karan Singh (2026). *Content-Matched Is Not Lexically Matched: Text and Untrained-Network Baselines for Evaluation-Awareness Probing.* Preprint, Zenodo. https://doi.org/10.5281/zenodo.23193081

The preprint extends this study to 17 open models and adds two baselines. A bag-of-words classifier given the probe's labels and splits separates all eight cues of EvalAwareBench (median AUROC 0.89–1.00), and 13 untrained networks pass the per-cue probe test on every cue. Probe success on such benchmarks therefore cannot, by itself, show that a model represents being evaluated. Read the per-cue results below with that in mind.

## What I found (first study, Qwen2.5-7B-Instruct)
Numbers from the corrected, pre-registered analysis of 2026-09-19 (oriented AUROC; layers chosen on training data only).
- A high probe score is not proof. The all-factors probe reached AUROC 1.000, but 200 random directions reached a mean of 0.738 and a 95th percentile of 0.943 (maximum 1.000), so the classes separate on the surface alone.
- 6 of the 8 single cues pass the per-cue rule (CI above the random-direction 95th percentile, Holm-corrected p < 0.05, and above a length-only classifier). Example: the stakes cue reached 0.924 [0.886, 0.961] against a random-direction 95th percentile of 0.618. The preprint shows that word-only and untrained-network baselines pass the same test.
- It is not one direction: the average cosine between cue directions is 0.17, against 0.50 expected by chance with this construction.
- It is unreliable as a monitor: at a realistic 1% rate of evaluations, precision is 17.6 to 19.5 percent (95% CI 13.3 to 27.0).

Earlier versions of this README reported a random-direction score of 0.986 and a precision of 24 to 42 percent; those came from the analysis before the correction. The corrected analysis code is in the preprint's materials, not yet in this repository.

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

## Citation
If you use this work, please cite the preprint above (see `CITATION.cff`).

## License
MIT (see `LICENSE`).
