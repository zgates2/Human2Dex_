#!/usr/bin/env bash
# Build Wuji Zarr datasets from already validated object-pipeline QC segments.
# This script reuses existing stage-08 outputs and does not rerun SAM, wrist,
# grasp-pocket, object tracking, objectPocketObs, appearance augmentation, or QC.
set -euo pipefail

PY="${PYTHON_BIN:-/share/project/liyuanyuan/anaconda3/envs/sam3/bin/python}"
REPO_ROOT="/home/zjc/Desktop/human2dex"
DSL_ROOT="/home/zjc/Desktop/human2dex"
CONVERTER="${REPO_ROOT}/tools/convert_pkl_to_training_zarr.py"
WUJI_ROOT="/share/project/liyuanyuan/data/dexglove_data/dex_data/wuji"

OVERWRITE=0
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage:
  bash glove_aug_pipeline/build_wuji_zarr_reuse_existing_object_pipelines.sh [options]

Options:
  --overwrite   Allow replacing existing output directories
  --dry-run     Print conversion commands without writing
  -h, --help    Show this help

Builds four Wuji Zarr outputs:
  pick_sponge_new_sponge_object_pocket_single_view_skeleton_tcp_v1
  lift_bottom_plastic_water_bottle_object_pocket_single_view_skeleton_tcp_v1
  pick_bread_new_croissant_object_pocket_single_view_skeleton_tcp_v1
  cup_in_cup_green_red_cup_pocket_single_view_skeleton_tcp_v1
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --overwrite) OVERWRITE=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

run_convert() {
  local name="$1"
  local input="$2"
  local object_dim="$3"
  local output="${WUJI_ROOT}/${name}"

  if [[ ! -d "${input}" ]]; then
    echo "Missing input segments: ${input}" >&2
    exit 1
  fi
  if [[ -e "${output}" && "${OVERWRITE}" != "1" ]]; then
    echo "Output already exists: ${output}" >&2
    echo "Use --overwrite only after confirming this output can be replaced." >&2
    exit 1
  fi

  local command=(
    "${PY}" "${CONVERTER}"
    --input "${input}"
    --output "${output}"
    --data-scaling-laws-root "${DSL_ROOT}"
    --rgb-image-field rgbImage
    --object-pocket-obs-field objectPocketObs
    --object-pocket-obs-dim "${object_dim}"
    --trajectory-pose-source trajectoryPose_tcp
    --gripper-source fused_wuji_command
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
  echo "========== Wuji Zarr: ${name} =========="
  (cd "${REPO_ROOT}" && "${command[@]}")
}

run_convert \
  "pick_sponge_new_sponge_object_pocket_single_view_skeleton_tcp_v1" \
  "/share/project/liyuanyuan/data/dexglove_data/pkl_dataset/pick_sponge_new/pipeline_object_obs_v1/qc_segments_single_view_v1/common" \
  5

run_convert \
  "lift_bottom_plastic_water_bottle_object_pocket_single_view_skeleton_tcp_v1" \
  "/share/project/liyuanyuan/data/dexglove_data/pkl_dataset/lift_bottom/pipeline_object_obs_v1/qc_segments_single_view_v1/common" \
  5

run_convert \
  "pick_bread_new_croissant_object_pocket_single_view_skeleton_tcp_v1" \
  "/share/project/liyuanyuan/data/dexglove_data/pkl_dataset/pick_bread_new/pipeline_object_obs_v1/qc_segments_single_view_v1/common" \
  5

run_convert \
  "cup_in_cup_green_red_cup_pocket_single_view_skeleton_tcp_v1" \
  "/share/project/liyuanyuan/data/dexglove_data/pkl_dataset/cup_in_cup/pipeline_object_obs_v1/qc_segments_single_view_v1/common" \
  10

echo
echo "All requested Wuji Zarr conversions completed."
