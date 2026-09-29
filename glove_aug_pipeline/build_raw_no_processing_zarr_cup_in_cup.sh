#!/usr/bin/env bash
# Convert raw cup_in_cup PKLs directly to trainable Zarr.
#
# No preprocessing, no wrist fusion, no TCP backfill, no skeleton rendering,
# no appearance augmentation, no object tracking, and no objectPocketObs.
# This script only repacks existing PKL/image fields into training Zarr.
set -euo pipefail

PY="${PYTHON_BIN:-/share/project/liyuanyuan/anaconda3/envs/sam3/bin/python}"
REPO_ROOT="/home/zjc/Desktop/human2dex"
DSL_ROOT="/home/zjc/Desktop/human2dex"
CONVERTER="${REPO_ROOT}/tools/convert_pkl_to_training_zarr.py"
INPUT="/share/project/liyuanyuan/data/dexglove_data/pkl_dataset/cup_in_cup/cup_in_cup"
O6_OUTPUT="/share/project/liyuanyuan/data/dexglove_data/dex_data/linker_o6/cup_in_cup_raw_no_processing"
WUJI_OUTPUT="/share/project/liyuanyuan/data/dexglove_data/dex_data/wuji/cup_in_cup_raw_no_processing"

TARGET="both"
OVERWRITE=0
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage:
  bash glove_aug_pipeline/build_raw_no_processing_zarr_cup_in_cup.sh [options]

Options:
  --target TARGET   o6 | wuji | both, default: both
  --overwrite       Allow replacing existing output directories
  --dry-run         Print/scan conversion commands without writing
  -h, --help        Show this help

Raw input:
  /share/project/liyuanyuan/data/dexglove_data/pkl_dataset/cup_in_cup/cup_in_cup

Field mapping, intentionally raw/no-processing:
  image:       rgbImage
  trajectory:  trajectoryPose
  linker_o6:   hand_command  (converter reads o6_command or handCommand)
  wuji:        wuji_command
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target)
      if [[ $# -lt 2 ]]; then
        echo "--target requires o6, wuji, or both" >&2
        exit 2
      fi
      TARGET="$2"
      shift 2
      ;;
    --overwrite) OVERWRITE=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

case "${TARGET}" in
  o6|wuji|both) ;;
  *) echo "--target must be o6, wuji, or both: ${TARGET}" >&2; exit 2 ;;
esac

should_run_target() {
  local target="$1"
  [[ "${TARGET}" == "both" || "${TARGET}" == "${target}" ]]
}

run_convert() {
  local target="$1"
  local output="$2"
  local gripper_source="$3"

  if ! should_run_target "${target}"; then
    return 0
  fi
  if [[ ! -d "${INPUT}" ]]; then
    echo "Missing input: ${INPUT}" >&2
    exit 1
  fi
  if [[ -e "${output}" && "${OVERWRITE}" != "1" ]]; then
    echo "Output already exists: ${output}" >&2
    echo "Use --overwrite only after confirming this output can be replaced." >&2
    exit 1
  fi

  local command=(
    "${PY}" "${CONVERTER}"
    --input "${INPUT}"
    --output "${output}"
    --data-scaling-laws-root "${DSL_ROOT}"
    --rgb-image-field rgbImage
    --trajectory-pose-source trajectoryPose
    --gripper-source "${gripper_source}"
    --output-format zip
    --workers 64
    --opencv-threads 1
    --blosc-threads 1
    --image-batch-size 32
    --max-inflight-tasks 128
    --image-compressor blosc
    --skip-missing-images
    --skip-empty-episodes
  )
  if [[ "${OVERWRITE}" == "1" ]]; then
    command+=(--overwrite)
  fi
  if [[ "${DRY_RUN}" == "1" ]]; then
    command+=(--dry-run)
  fi

  echo
  echo "========== Raw no-processing Zarr: cup_in_cup / ${target} =========="
  echo "input:  ${INPUT}"
  echo "output: ${output}"
  echo "trajectory: trajectoryPose"
  echo "gripper: ${gripper_source}"
  (cd "${REPO_ROOT}" && "${command[@]}")
}

run_convert "o6" "${O6_OUTPUT}" "hand_command"
run_convert "wuji" "${WUJI_OUTPUT}" "wuji_command"

echo
echo "cup_in_cup raw no-processing Zarr conversion completed."
