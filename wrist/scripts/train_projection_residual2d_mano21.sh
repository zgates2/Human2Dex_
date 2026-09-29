#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/home/zjc/Desktop/human2dex}
PY=${PY:-/share/project/liyuanyuan/anaconda3/envs/sam3/bin/python}
ANN=${ANN:-/share/project/liyuanyuan/data/dexglove_data/pkl_dataset/wrist_2D/annotations/visible_hand_points_mano21_v1.json}
STAGE1=${STAGE1:-${REPO}/wrist/outputs/test_3/runs/stage2_mano_onestage_v1/checkpoints/best.pt}
INIT_PROJ=${INIT_PROJ:-${REPO}/wrist/outputs/stage2_projection_head_v2/proj_head_mano21_v2/checkpoints/best.pt}
OUT=${OUT:-${REPO}/wrist/outputs/stage2_projection_head_residual2d}
RUN_NAME=${RUN_NAME:-proj_head_mano21_residual2d_v1}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

cd "${REPO}"

echo "Repo:       ${REPO}"
echo "Python:     ${PY}"
echo "Ann:        ${ANN}"
echo "Stage1:     ${STAGE1}"
echo "Init proj:  ${INIT_PROJ}"
echo "Out:        ${OUT}/${RUN_NAME}"
echo "CUDA:       ${CUDA_VISIBLE_DEVICES}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" "${PY}" \
  wrist/scripts/train_stage2_projection_head.py \
  --annotations "${ANN}" \
  --stage1-checkpoint "${STAGE1}" \
  --init-projection-head-checkpoint "${INIT_PROJ}" \
  --labels mano21 \
  --min-visible "${MIN_VISIBLE:-8}" \
  --split-by "${SPLIT_BY:-random}" \
  --val-ratio "${VAL_RATIO:-0.2}" \
  --projection-head-type residual2d \
  --residual-scale "${RESIDUAL_SCALE:-32}" \
  --residual-reg-weight "${RESIDUAL_REG_WEIGHT:-1e-4}" \
  --epochs "${EPOCHS:-500}" \
  --batch-size "${BATCH_SIZE:-32}" \
  --num-workers "${NUM_WORKERS:-8}" \
  --lr "${LR:-1e-3}" \
  --weight-decay "${WEIGHT_DECAY:-1e-4}" \
  --loss-beta "${LOSS_BETA:-4.0}" \
  --output-dir "${OUT}" \
  --run-name "${RUN_NAME}" \
  --num-overlays "${NUM_OVERLAYS:-32}"
