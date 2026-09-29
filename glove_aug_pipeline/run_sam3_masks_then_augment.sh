#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/share/project/liyuanyuan/anaconda3/envs/sam3/bin/python}"
MASK_CONFIG="${MASK_CONFIG:-${SCRIPT_DIR}/01_generate_sam3_masks.yaml}"
AUG_CONFIG="${AUG_CONFIG:-${SCRIPT_DIR}/02_augment_dataset.yaml}"

#对应的路径需要根据实际情况修改
INPUT_ROOT="${INPUT_ROOT:-/share/project/liyuanyuan/data/dexglove_data/pkl_dataset/flower_in_vase/flower_in_vase}"
MASK_OUTPUT="${MASK_OUTPUT:-/share/project/liyuanyuan/data/dexglove_data/pkl_dataset/flower_in_vase/flower_in_vase_sam3_masks}"
AUG_OUTPUT="${AUG_OUTPUT:-/share/project/liyuanyuan/data/dexglove_data/pkl_dataset/flower_in_vase/flower_in_vase}"
QC_OUTPUT="${QC_OUTPUT:-/share/project/liyuanyuan/data/dexglove_data/pkl_dataset/flower_in_vase/flower_in_vase_qc}"

IMAGE_SUBDIRS="${IMAGE_SUBDIRS:-images}"
NUM_GPUS="${NUM_GPUS:-8}"
MASK_BATCH_SIZE="${MASK_BATCH_SIZE:-8}"
MASK_PREFETCH="${MASK_PREFETCH:-8}"
AUG_WORKERS="${AUG_WORKERS:-8}"
AUG_IO_WORKERS="${AUG_IO_WORKERS:-2}"
OVERWRITE="${OVERWRITE:-0}"

LIMIT_EPISODES="${LIMIT_EPISODES:-}"
LIMIT_FRAMES="${LIMIT_FRAMES:-}"
EPISODE="${EPISODE:-}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

usage() {
  cat <<'EOF'
Run SAM3 mask generation first, then run hand augmentation.

Default:
  INPUT_ROOT=/share/project/liyuanyuan/data/dexglove_data/pkl_dataset/pick_6_pro
  MASK_OUTPUT=/share/project/liyuanyuan/data/dexglove_data/pkl_dataset/pick_6_sam3_masks
  AUG_OUTPUT=/share/project/liyuanyuan/data/dexglove_data/pkl_dataset/pick_6_pro

Common overrides:
  INPUT_ROOT=/path/to/raw_dataset \
  MASK_OUTPUT=/path/to/sam3_masks \
  AUG_OUTPUT=/path/to/aug_dataset \
  IMAGE_SUBDIRS=images \
  LIMIT_EPISODES=3 \
  LIMIT_FRAMES=20 \
  bash glove_aug_pipeline/run_sam3_masks_then_augment.sh

Notes:
  IMAGE_SUBDIRS is comma-separated.
  OVERWRITE=1 appends --overwrite to both stages.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

add_image_subdirs() {
  local -n out_array=$1
  local raw="${2:-}"
  local item
  IFS=',' read -ra parts <<<"${raw}"
  for item in "${parts[@]}"; do
    item="${item#"${item%%[![:space:]]*}"}"
    item="${item%"${item##*[![:space:]]}"}"
    if [[ -n "${item}" ]]; then
      out_array+=(--image-subdir "${item}")
    fi
  done
}

cd "${REPO_ROOT}"

mask_cmd=(
  "${PYTHON_BIN}"
  "${SCRIPT_DIR}/01_generate_sam3_masks.py"
  --config "${MASK_CONFIG}"
  --input "${INPUT_ROOT}"
  --output "${MASK_OUTPUT}"
  --num-gpus "${NUM_GPUS}"
  --batch-size "${MASK_BATCH_SIZE}"
  --prefetch "${MASK_PREFETCH}"
)
add_image_subdirs mask_cmd "${IMAGE_SUBDIRS}"

aug_cmd=(
  "${PYTHON_BIN}"
  "${SCRIPT_DIR}/02_augment_dataset.py"
  --config "${AUG_CONFIG}"
  --input "${INPUT_ROOT}"
  --output "${AUG_OUTPUT}"
  --mask-root "${MASK_OUTPUT%/}/masks"
  --qc-output "${QC_OUTPUT}"
  --workers "${AUG_WORKERS}"
  --io-workers "${AUG_IO_WORKERS}"
)
add_image_subdirs aug_cmd "${IMAGE_SUBDIRS}"

if [[ -n "${LIMIT_EPISODES}" ]]; then
  mask_cmd+=(--limit-episodes "${LIMIT_EPISODES}")
  aug_cmd+=(--limit-episodes "${LIMIT_EPISODES}")
fi

if [[ -n "${LIMIT_FRAMES}" ]]; then
  mask_cmd+=(--limit-frames "${LIMIT_FRAMES}")
  aug_cmd+=(--limit-frames "${LIMIT_FRAMES}")
fi

if [[ -n "${EPISODE}" ]]; then
  mask_cmd+=(--episode "${EPISODE}")
  aug_cmd+=(--episode "${EPISODE}")
fi

if [[ "${OVERWRITE}" == "1" || "${OVERWRITE}" == "true" || "${OVERWRITE}" == "yes" ]]; then
  mask_cmd+=(--overwrite)
  aug_cmd+=(--overwrite)
fi

printf '\n[1/2] Generate SAM3 masks\n'
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES}"
printf '%q ' "${mask_cmd[@]}"
printf '\n'
"${mask_cmd[@]}"

printf '\n[2/2] Augment dataset\n'
printf '%q ' "${aug_cmd[@]}"
printf '\n'
"${aug_cmd[@]}"

printf '\nDone.\n'
printf 'Mask output: %s\n' "${MASK_OUTPUT}"
printf 'Aug output:  %s\n' "${AUG_OUTPUT}"
printf 'QC output:   %s\n' "${QC_OUTPUT}"