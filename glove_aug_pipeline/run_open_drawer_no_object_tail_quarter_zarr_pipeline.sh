#!/usr/bin/env bash
# Reuse processed open_drawer no-object data, trim the last floor(N/4) frames
# of every episode into a new PKL tree, then build linker_o6 and Wuji zarr.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${SCRIPT_DIR}/00_pipeline_config_open_drawer_no_object_tail_quarter_wuji_o6_v1.yaml"
DRY_RUN=0
DROP_SHORT=0
OVERWRITE_ZARR=0

usage() {
  cat <<'EOF'
Usage:
  bash glove_aug_pipeline/run_open_drawer_no_object_tail_quarter_zarr_pipeline.sh [options]

Options:
  --dry-run          Print/scan commands without writing zarr
  --drop-short       Drop episodes that become empty after trimming instead of failing
  --overwrite-zarr   Pass --overwrite to zarr conversion outputs
  -h, --help         Show this help

Input:
  /share/project/liyuanyuan/data/dexglove_data/pkl_dataset/open_drawer/pipeline_no_object_v1/appearance_single_view_skeleton_v1

Trim rule:
  remove the last floor(N/4) frames from each episode PKL, and hard-link only
  retained rgbImage files into a new output tree.

Outputs:
  /share/project/liyuanyuan/data/dexglove_data/dex_data/linker_o6/open_drawer_no_object_single_view_skeleton_tcp_tail_quarter_v1/dataset.zarr.zip
  /share/project/liyuanyuan/data/dexglove_data/dex_data/wuji/open_drawer_no_object_single_view_skeleton_tcp_tail_quarter_v1/dataset.zarr.zip
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --drop-short) DROP_SHORT=1; shift ;;
    --overwrite-zarr) OVERWRITE_ZARR=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

PY="${PYTHON_BIN:-/share/project/liyuanyuan/anaconda3/envs/sam3/bin/python}"

config_value() {
  "${PY}" -c 'import functools,sys,yaml; d=yaml.safe_load(open(sys.argv[1])); v=functools.reduce(lambda x,k: x.get(k) if isinstance(x,dict) else None, sys.argv[2].split("."), d); print(v)' "$CONFIG" "$1"
}

run_trim_tail_quarter() {
  local input output manifest_dir
  input="$(config_value paths.appearance_root)"
  output="$(config_value paths.trimmed_appearance_root)"
  manifest_dir="${output}/.human2dex_pipeline"

  echo
  echo "========== Stage 10: trim last floor(N/4) frames from each episode PKL =========="
  if [[ -d "${output}" ]]; then
    if [[ -d "${manifest_dir}" ]]; then
      echo "Reuse completed trimmed PKL root: ${output}"
      return 0
    fi
    echo "Trimmed output exists but has no completion manifest; possible partial output: ${output}" >&2
    echo "Move it aside manually before rerunning." >&2
    exit 1
  fi

  local command=("${PY}" "${SCRIPT_DIR}/10_trim_episode_tail.py" --config "${CONFIG}")
  if (( DROP_SHORT )); then command+=(--drop-short); fi
  if (( DRY_RUN )); then command+=(--dry-run); fi
  echo "source: ${input}"
  echo "output: ${output}"
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
  if (( OVERWRITE_ZARR )); then command+=(--overwrite); fi
  if (( DRY_RUN )); then command+=(--dry-run); fi

  echo
  echo "========== Stage 9: convert_pkl_to_training_zarr.py (${target}, no objectPocketObs) =========="
  if (( DRY_RUN )) && [[ ! -d "${input}" ]]; then
    echo "DRY_RUN=1: trimmed input does not exist because trim stage did not write it; print converter command only."
    printf 'Would run:'
    printf ' %q' "${command[@]}"
    printf '
'
    return 0
  fi
  (cd "${repo_root}" && "${command[@]}")
}

run_trim_tail_quarter
run_convert o6
run_convert wuji

echo
echo "open_drawer tail-quarter linker_o6/Wuji zarr pipeline completed."
