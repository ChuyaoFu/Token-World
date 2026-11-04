#!/usr/bin/env bash
# Install the release environment without modifying the SDK or original project.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_environment.sh"
BASE_PYTHON="${PPU_PYTHON:-python3.12}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || { echo 'Missing --python value' >&2; exit 2; }
      BASE_PYTHON="$2"; shift 2 ;;
    -h|--help)
      cat <<'HELP'
Usage: bash scripts/setup_env.sh [--python /path/to/python3.12]

Create/update ./venv with system-site-packages for the PPU SDK's torch 2.6.0
and torchvision 0.21.0. Install TokenWorld with codec, eval and test dependencies
under configs/environment/ppu.txt constraints. Does not install the PPU SDK,
replace its PyTorch, download models/data or start training.

Requires the PPU SDK and its Python 3.12 environment on each execution node.
PPU_PYTHON overrides the base Python; PPU_SDK_SETUP overrides SDK envsetup.sh.
Run once from one node, not concurrently from every worker. Re-running is safe.
HELP
      exit 0 ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done
cd -- "$TOKENWORLD_ROOT"
load_ppu_sdk
export PYTHONNOUSERSITE=1
# Resolve the system interpreter even when called from an activated venv.
BASE_PYTHON="$("$BASE_PYTHON" -c 'import sys; print(sys._base_executable)')"
"$BASE_PYTHON" - <<'PY'
import sys
import importlib.metadata as metadata
assert sys.version_info[:2] == (3, 12), 'This PPU recipe requires Python 3.12'
for name, expected in [('torch', '2.6.0'), ('torchvision', '0.21.0')]:
    actual = metadata.version(name)
    assert actual == expected, f'SDK must supply {name}=={expected}; found {actual}'
import torch, torchvision
print(f'Base Python: {sys.executable}; SDK torch: {torch.__version__} ({torch.__file__})')
PY
if [[ -e "$TOKENWORLD_VENV/pyvenv.cfg" ]]; then
  "$BASE_PYTHON" - "$TOKENWORLD_VENV/pyvenv.cfg" <<'PY'
import sys
from pathlib import Path
cfg = dict(line.split(' = ', 1) for line in Path(sys.argv[1]).read_text().splitlines() if ' = ' in line)
assert cfg.get('include-system-site-packages') == 'true', 'Existing venv must expose the SDK system packages'
assert cfg.get('version', '').startswith('3.12.'), 'Existing venv must use Python 3.12'
PY
else
  "$BASE_PYTHON" -m venv --system-site-packages "$TOKENWORLD_VENV"
fi
PYTHON_BIN="$TOKENWORLD_VENV/bin/python"
# Exact constraints retain installed SDK builds instead of fetching CUDA wheels.
"$PYTHON_BIN" - "$BASE_PYTHON" <<'PY'
import subprocess, sys, torch
expected = subprocess.check_output([sys.argv[1], '-c', 'import torch; print(torch.__file__)'], text=True).strip()
assert torch.__file__ == expected, 'Existing venv shadows SDK torch; use a clean release venv'
PY
"$PYTHON_BIN" -m pip install -c "$TOKENWORLD_ROOT/configs/environment/ppu.txt" -e '.[codec,eval,test]'
# Check actual imports, including Qwen3-VL and the two training entrypoints.
"$PYTHON_BIN" - "$BASE_PYTHON" "$TOKENWORLD_VENV/environment.json" <<'PY'
import importlib.metadata as metadata
import json, os, platform, subprocess, sys
from pathlib import Path
import torch, torchvision, av
from transformers import Qwen3VLForConditionalGeneration
import tokenworld.codec.train
import tokenworld.dynamics.train
expected = subprocess.check_output([sys.argv[1], '-c', 'import torch; print(torch.__file__)'], text=True).strip()
assert torch.__file__ == expected, 'SDK torch was unexpectedly replaced'
report = {
    'python': platform.python_version(), 'executable': sys.executable,
    'base_python': sys.argv[1], 'ppu_sdk': os.environ.get('PPU_SDK'),
    'torch_file': torch.__file__, 'torch_git_version': torch.version.git_version,
    'cuda_compatibility_version': torch.version.cuda,
    'packages': {n: metadata.version(n) for n in sorted({d.metadata['Name'] for d in metadata.distributions() if d.metadata['Name']})},
    'verified': ['Qwen3VL import', 'S-VAE trainer import', 'world-model trainer import'],
    'training_validated': False,
}
Path(sys.argv[2]).write_text(json.dumps(report, indent=2) + '\n')
print(f'Environment ready: {sys.executable}')
print(f'Installed version inventory: {sys.argv[2]}')
PY
