#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export TORCH_HOME="${TORCH_HOME:-$SCRIPT_DIR/../outputs/pretrained/torch}"
exec bash "$SCRIPT_DIR/run_python.sh" -m tokenworld.evaluation.render_replay "$@"
