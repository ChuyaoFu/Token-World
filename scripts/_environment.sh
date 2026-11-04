#!/usr/bin/env bash
# Shared by the installer and runtime launchers; paths follow this checkout.
TOKENWORLD_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TOKENWORLD_VENV="$TOKENWORLD_ROOT/venv"

load_ppu_sdk() {
  local sdk_setup="${PPU_SDK_SETUP:-/usr/local/PPU_SDK/envsetup.sh}"
  [[ -r "$sdk_setup" ]] || {
    printf 'PPU SDK not found: %s (set PPU_SDK_SETUP if installed elsewhere)\n' "$sdk_setup" >&2
    return 1
  }
  # Vendor envsetup.sh reads optional, possibly unset library-path variables.
  local restore_nounset=0 sdk_status=0
  [[ $- == *u* ]] && restore_nounset=1
  set +u
  source "$sdk_setup" cuda >/dev/null || sdk_status=$?
  if (( restore_nounset )); then set -u; fi
  return "$sdk_status"
}

load_training_environment() {
  load_ppu_sdk || return $?
  export PYTHONNOUSERSITE=1
  export PATH="$TOKENWORLD_VENV/bin:$PATH"
  [[ -x "$PYTHON_BIN" ]] || command -v "$PYTHON_BIN" >/dev/null 2>&1 || {
    printf 'Python not found: %s\nRun: bash %s/scripts/setup_env.sh\n' "$PYTHON_BIN" "$TOKENWORLD_ROOT" >&2
    return 1
  }
}
