#!/usr/bin/env bash
# Run the numbered Human2Dex derived-data pipeline.
# Every numbered script can also be launched independently with --config.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${SCRIPT_DIR}/00_pipeline_config.yaml"
FROM=1
TO=8
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage:
  bash glove_aug_pipeline/run_human2dex_pipeline.sh [options]

Options:
  --config PATH   Pipeline YAML (default: 00_pipeline_config.yaml)
  --from N        First stage, 1..8 (default: 1)
  --to N          Last stage, 1..8 (default: 8)
  --dry-run       Print every command without writing data

Stages:
  01 prepare writable base from immutable raw data
  02 wrist inference + PICO fusion + retargeting
  03 direct-RGB grasp-pocket backfill (8 GPU shards)
  04 SAM3 hand masks (8 GPU)
  05 appearance-only mask-guided augmentation (CPU parallel)
  06 G/L fisheye rendering (8 GPU)
  07 strict QC and target-specific contiguous segments (CPU parallel)
  08 O6/Wuji independent Zarr conversion (CPU parallel)

GPU stages 02, 03, 04 and 06 are intentionally sequential by default: each is
configured to occupy all eight GPUs.  This avoids accidental 16-process GPU
oversubscription.  CPU-only 05, 07 and 08 already use their configured worker
pools.
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

if (( FROM < 1 || TO > 8 || FROM > TO )); then
  echo "Invalid stage range: --from ${FROM} --to ${TO}" >&2
  exit 2
fi

PY="${PYTHON_BIN:-/share/project/liyuanyuan/anaconda3/envs/sam3/bin/python}"
run_stage() {
  local stage="$1"
  shift
  if (( stage < FROM || stage > TO )); then return 0; fi
  local script
  script=$(printf "%02d_%s.py" "$stage" "$1")
  shift
  local command=("${PY}" "${SCRIPT_DIR}/${script}" --config "${CONFIG}")
  if (( DRY_RUN )); then command+=(--dry-run); fi
  echo
  echo "========== Stage ${stage}: ${script} =========="
  "${command[@]}" "$@"
}

run_stage 1 prepare_base_enriched
run_stage 2 backfill_wrist_fusion
run_stage 3 backfill_grasp_pocket
run_stage 4 generate_sam3_masks
run_stage 5 appearance_only_augment
run_stage 6 materialize_gl_views
run_stage 7 qc_and_segment
run_stage 8 build_training_zarr

echo "Pipeline stages ${FROM}..${TO} completed."
