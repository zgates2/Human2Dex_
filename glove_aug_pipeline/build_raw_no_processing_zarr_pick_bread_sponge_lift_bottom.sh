#!/usr/bin/env bash
# Convert raw DexGlove PKL datasets directly to training Zarr.
#
# No preprocessing, no wrist fusion, no TCP backfill, no skeleton rendering,
# no appearance augmentation, no object tracking, and no objectPocketObs.
# This script only repacks existing PKL/image fields into trainable Zarr.
set -euo pipefail

PY="${PYTHON_BIN:-/share/project/liyuanyuan/anaconda3/envs/sam3/bin/python}"
REPO_ROOT="/home/zjc/Desktop/human2dex"
DSL_ROOT="/home/zjc/Desktop/human2dex"
CONVERTER="${REPO_ROOT}/tools/convert_pkl_to_training_zarr.py"
O6_ROOT="/share/project/liyuanyuan/data/dexglove_data/dex_data/linker_o6"
WUJI_ROOT="/share/project/liyuanyuan/data/dexglove_data/dex_data/wuji"

TARGET="both"
ONLY=""
OVERWRITE=0
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage:
  bash glove_aug_pipeline/build_raw_no_processing_zarr_pick_bread_sponge_lift_bottom.sh [options]

Options:
  --target TARGET   o6 | wuji | both, default: both
  --only TASK       bread | sponge | bottle, default: all three
  --overwrite       Allow replacing existing output directories
  --dry-run         Print/scan conversion commands without writing
  -h, --help        Show this help

Raw inputs:
  bread:
    /share/project/liyuanyuan/data/dexglove_data/pkl_dataset/pick_bread_new/pick_bread_new
  sponge:
    /share/project/liyuanyuan/data/dexglove_data/pkl_dataset/pick_sponge_new/pick_sponge_new
  bottle:
    /share/project/liyuanyuan/data/dexglove_data/pkl_dataset/lift_bottom/lift_bottom

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
    --only)
      if [[ $# -lt 2 ]]; then
        echo "--only requires bread, sponge, or bottle" >&2
        exit 2
      fi
      ONLY="$2"
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

should_run_task() {
  local task="$1"
  [[ -z "${ONLY}" || "${ONLY}" == "${task}" ]]
}

should_run_target() {
  local target="$1"
  [[ "${TARGET}" == "both" || "${TARGET}" == "${target}" ]]
}

run_convert() {
  local task="$1"
  local target="$2"
  local input="$3"
  local output_name="$4"
  local gripper_source="$5"
  local output_root

  if ! should_run_task "${task}"; then
    return 0
  fi
  if ! should_run_target "${target}"; then
    return 0
  fi
  if [[ ! -d "${input}" ]]; then
    echo "Missing input for ${task}: ${input}" >&2
    exit 1
  fi

  case "${target}" in
    o6) output_root="${O6_ROOT}/${output_name}" ;;
    wuji) output_root="${WUJI_ROOT}/${output_name}" ;;
    *) echo "Unknown target: ${target}" >&2; exit 2 ;;
  esac

  if [[ -e "${output_root}" && "${OVERWRITE}" != "1" ]]; then
    echo "Output already exists: ${output_root}" >&2
    echo "Use --overwrite only after confirming this output can be replaced." >&2
    exit 1
  fi

  local command=(
    "${PY}" "${CONVERTER}"
    --input "${input}"
    --output "${output_root}"
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
  echo "========== Raw no-processing Zarr: ${task} / ${target} =========="
  echo "input:  ${input}"
  echo "output: ${output_root}"
  echo "trajectory: trajectoryPose"
  echo "gripper: ${gripper_source}"
  (cd "${REPO_ROOT}" && "${command[@]}")
}

run_task() {
  local task="$1"
  local input="$2"
  local output_base="$3"

  run_convert "${task}" "o6" "${input}" "${output_base}" "hand_command"
  run_convert "${task}" "wuji" "${input}" "${output_base}" "wuji_command"
}

run_task \
  "bread" \
  "/share/project/liyuanyuan/data/dexglove_data/pkl_dataset/pick_bread_new/pick_bread_new" \
  "pick_bread_new_raw_no_processing"

run_task \
  "sponge" \
  "/share/project/liyuanyuan/data/dexglove_data/pkl_dataset/pick_sponge_new/pick_sponge_new" \
  "pick_sponge_new_raw_no_processing"

run_task \
  "bottle" \
  "/share/project/liyuanyuan/data/dexglove_data/pkl_dataset/lift_bottom/lift_bottom" \
  "lift_bottom_raw_no_processing"

echo
echo "Requested raw no-processing Zarr conversions completed."
