#!/usr/bin/env bash
# Run the complete open_drawer pipeline from raw images/PKLs to O6 Zarr.
set -euo pipefail

# Physical GPU 6 is in an uncorrectable ECC state on server-04; expose the
# seven healthy GPUs by default while still allowing callers to override it.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,7}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${SCRIPT_DIR}/00_pipeline_config_open_drawer_object_obs_tcp_v1.yaml"

exec bash "${SCRIPT_DIR}/run_human2dex_object_pipeline.sh" --config "${CONFIG}" "$@"
