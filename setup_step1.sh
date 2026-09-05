#!/usr/bin/env bash
# =============================================================================
# STEP 1 — environment + data setup for the MATS eval-awareness project.
# Run on AryanPC (where the GPU + HF model cache live):
#     bash setup_step1.sh
#
# It: builds a clean venv, installs the transformers/probing stack (CUDA torch),
# verifies the GPU works, downloads + inspects the datasets, then re-runs the
# readiness check. Models are ALREADY cached (Qwen2.5-7B/3B) so nothing re-downloads.
# =============================================================================
set -uo pipefail

ENV_DIR="${HOME}/mats_env"
PROJ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "======================================================================"
echo "STEP 1: environment + data setup"
echo "  project dir: ${PROJ_DIR}"
echo "  venv dir   : ${ENV_DIR}"
echo "======================================================================"

# --- 1. venv --------------------------------------------------------------
if [ ! -d "${ENV_DIR}" ]; then
  echo "[1/6] creating venv at ${ENV_DIR}"
  python3 -m venv "${ENV_DIR}"
else
  echo "[1/6] venv already exists at ${ENV_DIR} (reusing)"
fi
# shellcheck disable=SC1091
source "${ENV_DIR}/bin/activate"
python -m pip install --upgrade pip wheel setuptools

# --- 2. torch (CUDA build) ------------------------------------------------
# RTX 4050 is Ada (sm_89) -> CUDA 12.x wheels. Fall back to default if the
# pinned index fails.
echo "[2/6] installing PyTorch (CUDA 12.1 build)"
pip install torch --index-url https://download.pytorch.org/whl/cu121 \
  || { echo "  cu121 wheel failed, trying default torch wheel"; pip install torch; }

# --- 3. the rest of the stack --------------------------------------------
echo "[3/6] installing transformers / probing stack"
pip install \
  "transformers>=4.44" accelerate bitsandbytes \
  scikit-learn numpy pandas datasets huggingface_hub matplotlib

# --- 4. verify CUDA -------------------------------------------------------
echo "[4/6] verifying torch + CUDA (this must say CUDA available: True)"
python - <<'PY'
import torch
print("  torch:", torch.__version__)
print("  CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"    GPU {i}: {p.name}  {p.total_memory/1e9:.1f} GB")
else:
    print("  !! CUDA NOT available. Fixes: check `nvidia-smi`; if driver is older,")
    print("     reinstall torch with cu118:  pip install torch --index-url https://download.pytorch.org/whl/cu118")
PY

# --- 5. fetch + inspect datasets -----------------------------------------
echo "[5/6] downloading + inspecting datasets"
python "${PROJ_DIR}/fetch_data.py"

# --- 6. re-run the readiness check ---------------------------------------
echo "[6/6] re-running readiness check inside the venv"
python "${PROJ_DIR}/check_setup.py" || true

echo
echo "======================================================================"
echo "STEP 1 DONE."
echo "  Activate this env in future with:  source ${ENV_DIR}/bin/activate"
echo "  If CUDA said True and datasets downloaded, paste the output back."
echo "======================================================================"
