#!/usr/bin/env bash
# Complete single-view task-object pipeline. The old G/L runner is unchanged.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${SCRIPT_DIR}/00_pipeline_config_pick_cube_object_obs_tcp_v1.yaml"
FROM=1
TO=9
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage:
  bash glove_aug_pipeline/run_human2dex_object_pipeline.sh [options]

Options:
  --config PATH   Task YAML
  --from N        First stage, 1..9
  --to N          Last stage, 1..9
  --dry-run       Print commands without writing

Stages:
  01 writable base from immutable source
  01b optional trajectoryPose_palm/trajectoryPose_tcp backfill
  02 wrist/PICO fusion (config may reuse)
  03 grasp-pocket prediction (config may reuse)
  04 SAM3 human-hand masks for appearance only (config may reuse)
  05 SAM3 configured task-object tracking (8 GPU)
  06 causal deployment-rate objectPocketObs generation (CPU)
  07 hand-mask-guided appearance augmentation (CPU)
  08 single-view/object-observation/action QC and segmentation (CPU)
  09 O6/Wuji Zarr conversion with objectPocketObs (CPU)
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

config_bool() {
  "${PY}" -c 'import functools,sys,yaml; d=yaml.safe_load(open(sys.argv[1])); v=functools.reduce(lambda x,k: x.get(k) if isinstance(x,dict) else None, sys.argv[2].split("."), d); print("1" if v is True else "0")' "$CONFIG" "$1"
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

run_stage 5 05_track_task_objects.py
run_stage 6 06_build_object_pocket_obs.py
run_stage 7 07_appearance_only_augment.py
run_stage 8 08_qc_single_view.py
run_stage 9 09_build_single_view_zarr.py --target o6

echo "Single-view object pipeline stages ${FROM}..${TO} completed."
