#!/usr/bin/env bash
# Fast path for canteen_water: reuse existing base_enriched and hand masks,
# regenerate no-skeleton appearance data, then build O6 and Wuji Zarr.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${SCRIPT_DIR}/00_pipeline_config_canteen_water_no_object_no_skeleton_tcp_v1.yaml"
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage:
  bash glove_aug_pipeline/run_canteen_water_no_object_no_skeleton_reuse_pipeline.sh [options]

Options:
  --dry-run       Print commands without writing
  -h, --help      Show this help

This script does not rerun raw preprocessing, wrist fusion, SAM hand masks,
object tracking, objectPocketObs, or object QC. It only runs:
  07 no-skeleton appearance regeneration
  09 O6 and Wuji Zarr conversion without objectPocketObs
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

PY="${PYTHON_BIN:-/share/project/liyuanyuan/anaconda3/envs/sam3/bin/python}"

config_value() {
  "${PY}" -c 'import functools,sys,yaml; d=yaml.safe_load(open(sys.argv[1])); v=functools.reduce(lambda x,k: x.get(k) if isinstance(x,dict) else None, sys.argv[2].split("."), d); print(v)' "$CONFIG" "$1"
}

run_stage_07() {
  local command=("${PY}" "${SCRIPT_DIR}/07_appearance_only_augment.py" --config "${CONFIG}")
  if (( DRY_RUN )); then command+=(--dry-run); fi
  echo
  echo "========== Stage 7: 07_appearance_only_augment.py (no skeleton) =========="
  "${command[@]}"
}

run_convert() {
  local target="$1"
  local repo_root data_scaling_laws_root input output source_image trajectory_source gripper_source
  repo_root="$(config_value runtime.repo_root)"
  data_scaling_laws_root="$(config_value runtime.data_scaling_laws_root)"
  input="$(config_value paths.appearance_root)"
  output="$(config_value paths.${target}_zarr_root)"
  source_image="$(config_value fields.source_image)"
  trajectory_source="$(config_value labels.trajectory_source)"
  gripper_source="$(config_value labels.${target}_action_field)"
  local command=(
    "${PY}" "${repo_root}/tools/convert_pkl_to_training_zarr.py"
    --input "${input}"
    --output "${output}"
    --data-scaling-laws-root "${data_scaling_laws_root}"
    --rgb-image-field "${source_image}"
    --trajectory-pose-source "${trajectory_source}"
    --gripper-source "${gripper_source}"
    --output-format "$(config_value zarr.output_format)"
    --workers "$(config_value parallel.zarr_workers)"
    --opencv-threads "$(config_value parallel.zarr_opencv_threads)"
    --blosc-threads "$(config_value parallel.zarr_blosc_threads)"
    --image-batch-size "$(config_value parallel.zarr_image_batch_size)"
    --max-inflight-tasks "$(config_value parallel.zarr_max_inflight_tasks)"
    --image-compressor "$(config_value zarr.image_compressor)"
    --skip-missing-images
    --skip-empty-episodes
  )
  if (( DRY_RUN )); then command+=(--dry-run); fi
  echo
  echo "========== Stage 9: convert_pkl_to_training_zarr.py (${target}, no objectPocketObs) =========="
  (cd "${repo_root}" && "${command[@]}")
}

run_stage_07
run_convert o6
run_convert wuji

echo
echo "canteen_water no-skeleton O6/Wuji pipeline completed."
