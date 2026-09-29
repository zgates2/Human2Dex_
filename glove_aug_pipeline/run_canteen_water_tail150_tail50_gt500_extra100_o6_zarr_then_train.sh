#!/usr/bin/env bash
# Second-pass canteen_water processing: trim already-tail150 no-skeleton PKLs,
# build linker_o6 zarr, then launch linker_o6 training with the generated zarr.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${SCRIPT_DIR}/00_pipeline_config_canteen_water_no_object_no_skeleton_tail150_tail50_gt500_extra100_o6_v1.yaml"
TRAIN_SCRIPT="/home/zjc/Desktop/human2dex/train_scripts/glove_linker_o6_canteen_water_no_object_tail150_tail50_gt500_extra100_20hz.sh"
DRY_RUN=0
DROP_SHORT=0
SKIP_TRAIN=0

usage() {
  cat <<'EOF'
Usage:
  bash glove_aug_pipeline/run_canteen_water_tail150_tail50_gt500_extra100_o6_zarr_then_train.sh [options]

Options:
  --dry-run       Print/scan commands without writing zarr or launching training
  --drop-short    Drop episodes that become empty after trimming instead of failing
  --skip-train    Generate trimmed PKLs and O6 zarr, but do not launch training
  -h, --help      Show this help

Input is the already processed dataset:
  /share/project/liyuanyuan/data/dexglove_data/pkl_dataset/canteen_water/pipeline_no_object_no_skeleton_tail150_v1/appearance_single_view_tail150_v1

Trimming rule, measured on that processed input:
  source_len <= 500: remove last 50 frames
  source_len >  500: remove last 150 frames (50 + extra 100)
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --drop-short) DROP_SHORT=1; shift ;;
    --skip-train) SKIP_TRAIN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

PY="${PYTHON_BIN:-/share/project/liyuanyuan/anaconda3/envs/sam3/bin/python}"

config_value() {
  "${PY}" -c 'import functools,sys,yaml; d=yaml.safe_load(open(sys.argv[1])); v=functools.reduce(lambda x,k: x.get(k) if isinstance(x,dict) else None, sys.argv[2].split("."), d); print(v)' "$CONFIG" "$1"
}

run_trim_tail() {
  local command=("${PY}" "${SCRIPT_DIR}/10_trim_episode_tail.py" --config "${CONFIG}")
  if (( DROP_SHORT )); then command+=(--drop-short); fi
  if (( DRY_RUN )); then command+=(--dry-run); fi
  echo
  echo "========== Stage 10: trim tail50, plus extra100 when processed source episode >500 frames =========="
  "${command[@]}"
}

run_convert_o6() {
  local repo_root data_scaling_laws_root input output source_image trajectory_source gripper_source
  repo_root="$(config_value runtime.repo_root)"
  data_scaling_laws_root="$(config_value runtime.data_scaling_laws_root)"
  input="$(config_value paths.trimmed_appearance_root)"
  output="$(config_value paths.o6_zarr_root)"
  source_image="$(config_value fields.source_image)"
  trajectory_source="$(config_value labels.trajectory_source)"
  gripper_source="$(config_value labels.o6_action_field)"
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
  echo "========== Stage 9: build linker_o6 zarr only =========="
  if (( DRY_RUN )) && [[ ! -d "${input}" ]]; then
    echo "DRY_RUN=1: trimmed input does not exist because Stage 10 did not write it; skip converter execution."
    printf 'Would run:'
    printf ' %q' "${command[@]}"
    printf '\n'
    return 0
  fi
  (cd "${repo_root}" && "${command[@]}")
}

run_train() {
  local zarr_root dataset_zip
  zarr_root="$(config_value paths.o6_zarr_root)"
  dataset_zip="${zarr_root}/dataset.zarr.zip"
  echo
  echo "========== Launch linker_o6 training =========="
  echo "dataset: ${dataset_zip}"
  echo "train_script: ${TRAIN_SCRIPT}"
  if (( DRY_RUN )); then
    echo "DRY_RUN=1: skip training launch"
    return 0
  fi
  if (( SKIP_TRAIN )); then
    echo "--skip-train set: skip training launch"
    return 0
  fi
  if [[ ! -f "${dataset_zip}" ]]; then
    echo "Missing generated dataset: ${dataset_zip}" >&2
    exit 1
  fi
  DATASET_PATH="${dataset_zip}" bash "${TRAIN_SCRIPT}"
}

run_trim_tail
run_convert_o6
run_train
