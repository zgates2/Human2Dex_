#!/usr/bin/env bash
# Run the complete lift_bottom pipeline from raw images/PKLs to O6 Zarr.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${SCRIPT_DIR}/00_pipeline_config_lift_bottom_object_obs_tcp_v1.yaml"

exec bash "${SCRIPT_DIR}/run_human2dex_object_pipeline.sh" --config "${CONFIG}" "$@"
