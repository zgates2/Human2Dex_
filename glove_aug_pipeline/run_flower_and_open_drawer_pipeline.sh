#!/usr/bin/env bash
# Run both requested tasks:
#   1. flower_in_vase with tulip flower task-object tracking
#   2. open_drawer without task-object prompt tracking
# Each task builds both O6 and Wuji Zarr outputs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

exec_task() {
  local name="$1"
  shift
  echo
  echo "########################################################################"
  echo "# ${name}"
  echo "########################################################################"
  bash "$@"
}

exec_task "flower_in_vase: tulip flower object pipeline" \
  "${SCRIPT_DIR}/run_flower_in_vase_object_pipeline.sh"

exec_task "open_drawer: no task-object prompt tracking pipeline" \
  "${SCRIPT_DIR}/run_open_drawer_no_object_pipeline.sh"

echo
echo "Both requested pipelines completed."
