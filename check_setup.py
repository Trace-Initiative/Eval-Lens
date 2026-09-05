#!/usr/bin/env python3
"""
check_setup.py — MATS eval-awareness project readiness check.

Run this on the machine that will do the experiments (AryanPC, where the GPU + HF cache live):
    python3 check_setup.py

It reports, clearly:
  1. System + disk
  2. GPU / CUDA (can we load models at all)
  3. Python packages we need (transformers path, NOT llama.cpp)
  4. Which target models are cached AND usable for PROBING (safetensors, not GGUF)
  5. Every other HF model cached (in case useful)
  6. Any datasets already cached
  7. A final verdict: what you have vs. what to download

Nothing is downloaded or loaded into VRAM; this only inspects what already exists.
"""

import os
import sys
import shutil
import glob

# Models we actually need for the probing project (must be HF/safetensors, INSTRUCT variants).
TARGETS = [
    "Qwen/Qwen2.5-7B-Instruct",   # primary
    "Qwen/Qwen2.5-3B-Instruct",   # scale check (stretch)
]

# Packages the transformers/probing pipeline needs (GGUF/LM Studio does NOT count).
NEEDED_PKGS = [
    "torch", "transformers", "bitsandbytes", "accelerate",
    "sklearn", "numpy", "pandas", "datasets", "huggingface_hub",
]

# Dataset repo-id hints to look for in the datasets cache (best-effort substrings).
DATASET_HINTS = ["evaldetect", "eval-detect", "eval_aware", "evaluation-aware",
                 "wildchat", "lmsys", "devbunova"]


def line(c="-", n=70):
    print(c * n)


def human(nbytes):
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if abs(nbytes) < 1024.0:
            return f"{nbytes:6.1f} {unit}"
        nbytes /= 1024.0
    return f"{nbytes:.1f} PB"


def section(title):
    print()
    line("=")
    print(title)
    line("=")


# ----------------------------------------------------------------------------- 1
section("1. SYSTEM")
try:
    print("hostname :", os.uname().nodename)
except Exception:
    print("hostname : (unknown)")
print("user     :", os.environ.get("USER", "(unknown)"))
print("home     :", os.path.expanduser("~"))
print("python   :", sys.version.split()[0], "->", sys.executable)
try:
    du = shutil.disk_usage(os.path.expanduser("~"))
    print(f"disk free: {human(du.free)}  (of {human(du.total)})   [need ~25-30 GB free if downloading models]")
except Exception as e:
    print("disk     : could not read:", e)


# ----------------------------------------------------------------------------- 2
section("2. GPU / CUDA")
try:
    import torch
    print("torch    :", torch.__version__)
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {p.name}  |  {human(p.total_memory)} VRAM")
        print("  -> CUDA OK. 4-bit 7B needs ~5 GB VRAM; you're fine at 6 GB.")
    else:
        print("  !! CUDA NOT available. Probing will be very slow on CPU.")
        print("     Check the torch install matches your CUDA, or use the GPU machine.")
except Exception as e:
    print("torch not importable:", e)


# ----------------------------------------------------------------------------- 3
section("3. PYTHON PACKAGES (transformers path — what probing needs)")
missing = []
for pkg in NEEDED_PKGS:
    name = "scikit-learn" if pkg == "sklearn" else pkg
    try:
        mod = __import__(pkg)
        ver = getattr(mod, "__version__", "?")
        print(f"  [OK]      {name:16s} {ver}")
    except Exception:
        print(f"  [MISSING] {name:16s}  -> pip install {name}")
        missing.append(name)


# ----------------------------------------------------------------------------- 4 & 5
section("4/5. HUGGINGFACE MODEL CACHE  (safetensors = usable for probing; gguf = NOT)")

def scan_repo_files(repo_path):
    """Return (has_safetensors, has_gguf, has_config, total_bytes) for a cached repo dir."""
    has_st = has_gguf = has_cfg = False
    total = 0
    for root, _dirs, files in os.walk(repo_path):
        for f in files:
            fp = os.path.join(root, f)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
            fl = f.lower()
            if fl.endswith(".safetensors"):
                has_st = True
            elif fl.endswith(".gguf"):
                has_gguf = True
            if fl == "config.json":
                has_cfg = True
    return has_st, has_gguf, has_cfg, total


all_models = []  # (repo_id, has_st, has_gguf, has_cfg, size, path)
cache_ok = False
try:
    from huggingface_hub import scan_cache_dir
    info = scan_cache_dir()
    for repo in info.repos:
        if repo.repo_type != "model":
            continue
        has_st, has_gguf, has_cfg, total = scan_repo_files(str(repo.repo_path))
        all_models.append((repo.repo_id, has_st, has_gguf, has_cfg, total, str(repo.repo_path)))
    cache_ok = True
except Exception as e:
    # Fallback: manually walk the default hub dir.
    print("  (huggingface_hub.scan_cache_dir unavailable, walking cache manually:", e, ")")
    hub = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    if os.path.isdir(hub):
        for d in sorted(glob.glob(os.path.join(hub, "models--*"))):
            repo_id = os.path.basename(d)[len("models--"):].replace("--", "/")
            has_st, has_gguf, has_cfg, total = scan_repo_files(d)
            all_models.append((repo_id, has_st, has_gguf, has_cfg, total, d))
        cache_ok = True
    else:
        print("  !! No HF hub cache found at", hub)

def find(repo_id):
    for m in all_models:
        if m[0].lower() == repo_id.lower():
            return m
    return None

print("\n  >>> TARGET MODELS (the ones we need):")
need_download = []
for t in TARGETS:
    m = find(t)
    if m is None:
        print(f"    [MISSING]  {t}")
        need_download.append(t)
    else:
        _id, has_st, has_gguf, has_cfg, size, path = m
        usable = has_st and has_cfg
        tag = "USABLE for probing" if usable else ("GGUF-only, NOT usable for probing" if has_gguf else "incomplete")
        print(f"    [{'OK' if usable else 'CHECK'}]  {t}")
        print(f"            format: {'safetensors' if has_st else ''}{' + config' if has_cfg else ''}"
              f"{' | gguf' if has_gguf else ''}  ->  {tag}")
        print(f"            size: {human(size)}   path: {path}")
        if not usable:
            need_download.append(t)

print("\n  >>> ALL other cached HF models (for reference):")
if all_models:
    for _id, has_st, has_gguf, has_cfg, size, _path in sorted(all_models, key=lambda x: -x[4]):
        if _id in [t for t in TARGETS]:
            continue
        fmt = []
        if has_st:
            fmt.append("safetensors")
        if has_gguf:
            fmt.append("gguf")
        print(f"    {human(size)}  {_id:45s} [{', '.join(fmt) or 'other'}]")
else:
    print("    (none found)")


# ----------------------------------------------------------------------------- 6
section("6. DATASET CACHE (best-effort — is anything relevant already downloaded?)")
found_ds = []
try:
    from huggingface_hub import scan_cache_dir
    info = scan_cache_dir()
    for repo in info.repos:
        if repo.repo_type == "dataset":
            rid = repo.repo_id.lower()
            hit = any(h in rid for h in DATASET_HINTS)
            mark = "  <-- possibly relevant" if hit else ""
            print(f"    {human(repo.size_on_disk)}  {repo.repo_id}{mark}")
            if hit:
                found_ds.append(repo.repo_id)
except Exception:
    ds_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "datasets")
    if os.path.isdir(ds_dir):
        for d in sorted(os.listdir(ds_dir)):
            print("   ", d)
    else:
        print("    (no datasets cache found — we'll download the 3 small datasets)")


# ----------------------------------------------------------------------------- 7
section("7. VERDICT")
print("Packages to install :", ", ".join(missing) if missing else "none — all present")
print("Models to download  :", ", ".join(need_download) if need_download else "NONE — both target models are cached & usable")
print("Datasets            : EvalDetectBench, Devbunova 2x2, WildChat/LMSYS sample (<300 MB total)")
print("                      (found in cache: " + (", ".join(found_ds) if found_ds else "none — will download") + ")")
print()
print("If 'Models to download' is NONE and CUDA is OK, we are ready to build the experiment scripts.")
line("=")
