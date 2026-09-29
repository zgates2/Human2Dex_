#!/usr/bin/env bash
# Run Egg_Carton from raw images/PKLs to O6 Zarr, skipping task-object SAM3 tracking.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${SCRIPT_DIR}/00_pipeline_config_egg_carton_no_object_tcp_v1.yaml"
FROM=1
TO=9
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage:
  bash glove_aug_pipeline/run_egg_carton_no_object_pipeline.sh [options]

Options:
  --config PATH   Task YAML
  --from N        First stage, 1..9
  --to N          Last stage, 1..9
  --dry-run       Print commands without writing

Stages:
  01 writable base from immutable source
  01b optional trajectoryPose_palm/trajectoryPose_tcp backfill
  02 wrist/PICO fusion
  03 grasp-pocket prediction
  04 SAM3 human-hand masks for appearance only
  07 hand-mask-guided appearance augmentation
  09 O6 Zarr conversion without objectPocketObs

Stages 05 and 06 are intentionally skipped: no prompt-based task-object
tracking and no objectPocketObs generation are run.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --from) FROM="$2"; shift 2 ;;
    --to) TO="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if (( FROM < 1 || TO > 9 || FROM > TO )); then
  echo "Invalid stage range: --from ${FROM} --to ${TO}" >&2
  exit 2
fi

PY="${PYTHON_BIN:-/share/project/liyuanyuan/anaconda3/envs/sam3/bin/python}"

config_value() {
  "${PY}" -c 'import functools,sys,yaml; d=yaml.safe_load(open(sys.argv[1])); v=functools.reduce(lambda x,k: x.get(k) if isinstance(x,dict) else None, sys.argv[2].split("."), d); print(v)' "$CONFIG" "$1"
}

config_bool() {
  [[ "$(config_value "$1")" == "True" ]] && echo 1 || echo 0
}

run_stage() {
  local stage="$1"
  local script="$2"
  shift 2
  if (( stage < FROM || stage > TO )); then return 0; fi
  local command=("${PY}" "${SCRIPT_DIR}/${script}" --config "${CONFIG}")
  if (( DRY_RUN )); then command+=(--dry-run); fi
  echo
  echo "========== Stage ${stage}: ${script} =========="
  "${command[@]}" "$@"
}

run_convert_o6() {
  local stage=9
  if (( stage < FROM || stage > TO )); then return 0; fi
  local repo_root data_scaling_laws_root input output source_image trajectory_source gripper_source
  repo_root="$(config_value runtime.repo_root)"
  data_scaling_laws_root="$(config_value runtime.data_scaling_laws_root)"
  input="$(config_value paths.appearance_root)"
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
  echo "========== Stage 9: convert_pkl_to_training_zarr.py (O6, no objectPocketObs) =========="
  (cd "${repo_root}" && "${command[@]}")
}

run_stage 1 01_prepare_base_enriched.py
if (( FROM <= 1 && TO >= 1 )); then
  if [[ "$(config_bool trajectory.backfill_tcp)" == "1" ]]; then
    run_stage 1 01b_backfill_trajectory_refs.py
  fi
fi

if (( FROM <= 2 && TO >= 2 )); then
  if [[ "$(config_bool reuse.wrist_fusion)" == "1" ]]; then
    echo "[02] reuse.wrist_fusion=true: fields already copied by stage 01"
  else
    run_stage 2 02_backfill_wrist_fusion.py
  fi
fi

if (( FROM <= 3 && TO >= 3 )); then
  if [[ "$(config_bool reuse.grasp_pocket)" == "1" ]]; then
    echo "[03] reuse.grasp_pocket=true: fields already copied by stage 01"
  else
    run_stage 3 03_backfill_grasp_pocket.py
  fi
fi

if (( FROM <= 4 && TO >= 4 )); then
  if [[ "$(config_bool reuse.hand_masks)" == "1" ]]; then
    echo "[04] reuse.hand_masks=true: existing HAND masks reused for appearance only"
  else
    run_stage 4 04_generate_sam3_masks.py
  fi
fi

if (( FROM <= 6 && TO >= 5 )); then
  echo "[05-06] skipped: no prompt-based task-object tracking requested"
fi

run_stage 7 07_appearance_only_augment.py
run_convert_o6

echo "Egg_Carton no-object pipeline stages ${FROM}..${TO} completed."
