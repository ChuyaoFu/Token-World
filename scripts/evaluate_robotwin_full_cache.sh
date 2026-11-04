#!/usr/bin/env bash
# Compatibility entry point for the full-KV ablation.
set -euo pipefail
export ROBOTWIN_CACHE_MODE=full
exec bash "$(dirname "$0")/evaluate_robotwin.sh" "$@"
