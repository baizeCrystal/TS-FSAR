#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)";cd "$root"
conda_bin="${CONDA_EXE:-conda}";env_name="${TSFSAR_ENV_NAME:-tsfsar-dto-kd}"
command -v "$conda_bin" >/dev/null 2>&1 || { echo 'conda is required (install it in user space or use the platform image)' >&2; exit 1; }
if ! "$conda_bin" env list | awk '{print $1}' | grep -qx "$env_name"; then "$conda_bin" env create -f environment.yml -n "$env_name"; fi
# Install the pinned packages without forcing a CUDA build selected for another host.
source "$($conda_bin info --base)/etc/profile.d/conda.sh";conda activate "$env_name"
if [[ -n "${TSFSAR_TORCH_INDEX_URL:-}" ]]; then python -m pip install --upgrade --index-url "$TSFSAR_TORCH_INDEX_URL" torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2; fi
python -m pip install -r requirements-campus.txt
python - <<'PY'
import torch,transformers
print('installed torch',torch.__version__,'cuda build',torch.version.cuda)
print('installed transformers',transformers.__version__)
PY
echo "Environment ready: conda activate $env_name"
