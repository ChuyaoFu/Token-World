#!/usr/bin/env bash
# Stage the pinned CPU Vulkan runtime once per node, then test it on this image.
set -euo pipefail
helper=$(cd "$(dirname "$0")" && pwd)
release=$(cd "$helper/../.." && pwd)
digest=17642827ba84653db0bca9c707a4c36b42d2d9ee91d6a27e619df255a02d11c9
archive="$release/outputs/robotwin_sim_runtime/cpu-renderer-v1.tar.gz"
dest=${TOKENWORLD_RENDER_LIB:?renderer directory not set}
mkdir -p "$(dirname "$dest")"
exec 9>"$dest.lock"
flock -w 120 9
if [[ ! -f "$dest/$digest.ready" ]]; then
    printf '%s  %s\n' "$digest" "$archive" | sha256sum --check
    if [[ -e "$dest" ]]; then echo "Unrecognized renderer cache at $dest" >&2; exit 1; fi
    tmp=$(mktemp -d "$dest.build.XXXXXX")
    tar -xzf "$archive" -C "$tmp"
    python3 - "$dest" "$tmp/cpu_icd.json" <<'PY'
import json, sys
from pathlib import Path
target = Path(sys.argv[1]).resolve() / 'libvulkan_lvp.so'
Path(sys.argv[2]).write_text(json.dumps({'file_format_version':'1.0.0', 'ICD':{'library_path':str(target), 'api_version':'1.4.318'}}))
PY
    touch "$tmp/$digest.ready"
    mv "$tmp" "$dest"
fi
flock -u 9
echo "Testing SAPIEN CPU camera on $(hostname)"
if ! bash "$helper/sim_python.sh" "$helper/check_renderer.py"; then
    echo "Renderer test failed before asset preparation and policy launch. Vulkan loader diagnostics:" >&2
    VK_LOADER_DEBUG=error,warn bash "$helper/sim_python.sh" "$helper/check_renderer.py" || true
    LD_LIBRARY_PATH="$dest" ldd "$dest/libvulkan_lvp.so" >&2
    exit 1
fi
