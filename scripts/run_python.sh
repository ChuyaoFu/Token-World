#!/usr/bin/env bash
# Use the release venv and PPU SDK for cache export, tests, or Python diagnostics.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/_environment.sh"
PYTHON_BIN="${PYTHON_BIN:-$TOKENWORLD_VENV/bin/python}"
cd -- "$TOKENWORLD_ROOT"
load_training_environment
exec "$PYTHON_BIN" "$@"
