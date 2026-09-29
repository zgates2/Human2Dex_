#!/usr/bin/env bash
# Build pts21-action Zarr datasets from already processed DexGlove PKLs.
#
# The gripper/action source is fused_pts21_mano -> 63D gripper vector, so
# action is trajectoryPose_tcp(6D) + fused_pts21_mano(63D).
set -euo pipefail

PY="${PYTHON_BIN:-/share/project/liyuanyuan/anaconda3/envs/sam3/bin/python}"
REPO_ROOT="/home/zjc/Desktop/human2dex"
DSL_ROOT="/home/zjc/Desktop/human2dex"
CONVERTER="${REPO_ROOT}/tools/convert_pkl_to_training_zarr.py"
PTS21_ROOT="/share/project/liyuanyuan/data/dexglove_data/dex_data/pts21"

OVERWRITE=0
DRY_RUN=0
ONLY=""

usage() {
  cat <<'EOF'
Usage:
  bash glove_aug_pipeline/build_pts21_zarr_reuse_existing_pipelines.sh [options]

Options:
  --only TASK    Run one task: sponge|cube|bread|cup|flower|egg_carton|bottle
  --overwrite    Allow replacing existing output directories
  --dry-run      Print conversion commands without writing
  -h, --help     Show this help

Builds pts21-action Zarr outputs from existing processed PKLs:
  sponge      pick_sponge_new object-pocket skeleton QC segments
  cube        pick_cube pipeline_object_obs_v2 object-pocket skeleton QC segments
  bread       pick_bread_new object-pocket skeleton QC segments
  cup         cup_in_cup object-pocket skeleton QC segments
  flower      flower_in_vase object-pocket skeleton QC segments
  egg_carton  Egg_Carton no-object skeleton appearance PKLs
  bottle      lift_bottom base_enriched PKLs; no skeleton and no objectPocketObs
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --only)
      if [[ $# -lt 2 ]]; then
        echo "--only requires a task name" >&2
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

should_run() {
  local task="$1"
  [[ -z "${ONLY}" || "${ONLY}" == "${task}" ]]
}

run_convert() {
  local task="$1"
  local name="$2"
  local input="$3"
  local object_dim="$4"
  local output="${PTS21_ROOT}/${name}"

  if ! should_run "${task}"; then
    return 0
  fi
  if [[ ! -d "${input}" ]]; then
    echo "Missing input for ${task}: ${input}" >&2
    exit 1
  fi
  if [[ -e "${output}" && "${OVERWRITE}" != "1" ]]; then
    echo "Output already exists for ${task}: ${output}" >&2
    echo "Use --overwrite only after confirming this output can be replaced." >&2
    exit 1
  fi

  local command=(
    "${PY}" "${CONVERTER}"
    --input "${input}"
    --output "${output}"
    --data-scaling-laws-root "${DSL_ROOT}"
    --rgb-image-field rgbImage
    --trajectory-pose-source trajectoryPose_tcp
    --gripper-source fused_pts21_mano
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
  if [[ "${object_dim}" != "0" ]]; then
    command+=(
      --object-pocket-obs-field objectPocketObs
      --object-pocket-obs-dim "${object_dim}"
    )
  fi
  if [[ "${OVERWRITE}" == "1" ]]; then
    command+=(--overwrite)
  fi
  if [[ "${DRY_RUN}" == "1" ]]; then
    command+=(--dry-run)
  fi

  echo
  echo "========== pts21 Zarr: ${task} -> ${name} =========="
  (cd "${REPO_ROOT}" && "${command[@]}")
}

run_convert \
  "sponge" \
  "pick_sponge_new_sponge_object_pocket_single_view_skeleton_tcp_pts21_v1" \
  "/share/project/liyuanyuan/data/dexglove_data/pkl_dataset/pick_sponge_new/pipeline_object_obs_v1/qc_segments_single_view_v1/common" \
  5

run_convert \
  "cube" \
  "pick_cube_object_pocket_single_view_skeleton_tcp_v2_pts21_v1" \
  "/share/project/liyuanyuan/data/dexglove_data/pkl_dataset/pick_cube/pipeline_object_obs_v2/qc_segments_single_view_v1/common" \
  10

run_convert \
  "bread" \
  "pick_bread_new_croissant_object_pocket_single_view_skeleton_tcp_pts21_v1" \
  "/share/project/liyuanyuan/data/dexglove_data/pkl_dataset/pick_bread_new/pipeline_object_obs_v1/qc_segments_single_view_v1/common" \
  5

run_convert \
  "cup" \
  "cup_in_cup_green_red_cup_pocket_single_view_skeleton_tcp_pts21_v1" \
  "/share/project/liyuanyuan/data/dexglove_data/pkl_dataset/cup_in_cup/pipeline_object_obs_v1/qc_segments_single_view_v1/common" \
  10

run_convert \
  "flower" \
  "flower_in_vase_tulip_flower_object_pocket_single_view_skeleton_tcp_pts21_v1" \
  "/share/project/liyuanyuan/data/dexglove_data/pkl_dataset/flower_in_vase/pipeline_object_obs_v1/qc_segments_single_view_v1/common" \
  5

run_convert \
  "egg_carton" \
  "Egg_Carton_no_object_single_view_skeleton_tcp_pts21_v1" \
  "/share/project/liyuanyuan/data/dexglove_data/pkl_dataset/Egg_Carton/pipeline_no_object_v1/appearance_single_view_skeleton_v1" \
  0

run_convert \
  "bottle" \
  "lift_bottom_no_object_single_view_tcp_no_skeleton_pts21_v1" \
  "/share/project/liyuanyuan/data/dexglove_data/pkl_dataset/lift_bottom/pipeline_object_obs_v1/base_enriched_v1" \
  0

echo
echo "Requested pts21 Zarr conversions completed."
