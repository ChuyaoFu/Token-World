#!/usr/bin/env bash
set -eo pipefail
base=${ROBOTWIN_SIM_RUNTIME:?Set ROBOTWIN_SIM_RUNTIME to your separately installed RoboTwin simulator runtime}
source "${PPU_SDK_SETUP:-/usr/local/PPU_SDK/envsetup.sh}" cuda >/dev/null
export LD_LIBRARY_PATH="$base/mesa-prefix/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export VK_ICD_FILENAMES="$base/mesa-prefix/usr/share/vulkan/icd.d/lvp_icd.json" VK_DRIVER_FILES="$base/mesa-prefix/usr/share/vulkan/icd.d/lvp_icd.json"
if [[ -n "${TOKENWORLD_RENDER_LIB:-}" ]]; then
    export LD_LIBRARY_PATH="$TOKENWORLD_RENDER_LIB:$LD_LIBRARY_PATH"
    export SAPIEN_VULKAN_LIBRARY_PATH="$TOKENWORLD_RENDER_LIB/libvulkan.so.1"
    export VK_ICD_FILENAMES="$TOKENWORLD_RENDER_LIB/cpu_icd.json" VK_DRIVER_FILES="$TOKENWORLD_RENDER_LIB/cpu_icd.json"
fi
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export XDG_RUNTIME_DIR="${TMPDIR:-/tmp}/tokenworld-sim-runtime-$UID" XDG_CACHE_HOME="${TMPDIR:-/tmp}/tokenworld-sim-cache-$UID"
mkdir -p "$XDG_RUNTIME_DIR" "$XDG_CACHE_HOME"
chmod 700 "$XDG_RUNTIME_DIR"
export LP_NUM_THREADS=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export ROBOTWIN_RENDER_SHADER=default ROBOTWIN_DISABLE_DENOISER=1 ROBOTWIN_DISABLE_POLICY_TOPP=1 ROBOTWIN_POLICY_INTERPOLATION_STEPS=5 ROBOTWIN_SKIP_POLICY_INTERMEDIATE_RENDER=1
exec "$base/sim-env/bin/python" "$@"
