#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default collection target: 30 Hz MVS RGB + external/ego L515 RGB/depth.
# Edit these defaults when changing the collection target.
CONFIG_PATH="${CONFIG_PATH:-${SCRIPT_DIR}/collect_l515_mvs.yaml}"
TASK_NAME="${TASK_NAME:-test_all_cameras_30hz_fixed_buffered}"
OUT_ROOT="${OUT_ROOT:-/home/zjc/Desktop/human2dex_L515/teleop/data}"
DURATION="${DURATION:-10}"

CONDA_SH="${CONDA_SH:-/home/zjc/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-l515_mvs310}"

source "${CONDA_SH}"
conda activate "${CONDA_ENV}"

cd "${SCRIPT_DIR}"
PYTHONNOUSERSITE=1 python collect_l515_mvs.py \
  --config "${CONFIG_PATH}" \
  --task-name "${TASK_NAME}" \
  --out-root "${OUT_ROOT}" \
  --duration "${DURATION}" \
  --sync-mode fixed_rate_buffered \
  --sync-delay-ms 35 \
  --sync-max-delta-ms 75 \
  --no-drop-repeated-frames \
  --l515-stagger-sec 0.5
