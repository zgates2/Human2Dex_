#!/usr/bin/env bash
# Fast path for canteen_water: reuse existing base_enriched and hand masks,
# ensure no-skeleton appearance data exists, trim the last 150 PKL messages from
# every episode, then build O6 and Wuji Zarr from the trimmed PKLs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${SCRIPT_DIR}/00_pipeline_config_canteen_water_no_object_no_skeleton_tail150_tcp_v1.yaml"
DRY_RUN=0
DROP_SHORT=0

usage() {
  cat <<'EOF'
Usage:
  bash glove_aug_pipeline/run_canteen_water_no_object_no_skeleton_tail150_reuse_pipeline.sh [options]

Options:
  --dry-run       Print commands without writing
  --drop-short    Drop episodes with <=150 frames instead of failing
  -h, --help      Show this help

This script does not rerun raw preprocessing, wrist fusion, SAM hand masks,
object tracking, objectPocketObs, or object QC. It only runs/reuses:
  07 no-skeleton appearance regeneration
  10 trim last 150 frames from every episode PKL into a new dataset root
  09 O6 and Wuji Zarr conversion without objectPocketObs
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --drop-short) DROP_SHORT=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

PY="${PYTHON_BIN:-/share/project/liyuanyuan/anaconda3/envs/sam3/bin/python}"

config_value() {
  "${PY}" -c 'import functools,sys,yaml; d=yaml.safe_load(open(sys.argv[1])); v=functools.reduce(lambda x,k: x.get(k) if isinstance(x,dict) else None, sys.argv[2].split("."), d); print(v)' "$CONFIG" "$1"
}

run_stage_07_if_needed() {
  local appearance_root
  appearance_root="$(config_value paths.appearance_root)"
  echo
  echo "========== Stage 7: no-skeleton appearance =========="
  if [[ -d "${appearance_root}" ]]; then
    echo "Reuse existing appearance_root: ${appearance_root}"
    return 0
  fi
  local command=("${PY}" "${SCRIPT_DIR}/07_appearance_only_augment.py" --config "${CONFIG}")
  if (( DRY_RUN )); then command+=(--dry-run); fi
  "${command[@]}"
}

run_trim_tail() {
  local command=(
    "${PY}" "${SCRIPT_DIR}/10_trim_episode_tail.py"
    --config "${CONFIG}"
  )
  if (( DROP_SHORT )); then command+=(--drop-short); fi
  if (( DRY_RUN )); then command+=(--dry-run); fi
  echo
  echo "========== Stage 10: trim last 150 frames from each episode PKL =========="
  "${command[@]}"
}

run_convert() {
  local target="$1"
  local repo_root data_scaling_laws_root input output source_image trajectory_source gripper_source
  repo_root="$(config_value runtime.repo_root)"
  data_scaling_laws_root="$(config_value runtime.data_scaling_laws_root)"
  input="$(config_value paths.trimmed_appearance_root)"
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
  echo "========== Stage 9: convert_pkl_to_training_zarr.py (${target}, trimmed tail150, no objectPocketObs) =========="
  (cd "${repo_root}" && "${command[@]}")
}

run_stage_07_if_needed
run_trim_tail
run_convert o6
run_convert wuji

echo
echo "canteen_water no-skeleton tail150 O6/Wuji pipeline completed."
