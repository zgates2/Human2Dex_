#!/usr/bin/env bash
set -euo pipefail

# pick_bread_head_2: 原始 PKL -> Pico-only/Wuji + ego camera Zarr -> Wuji DP training.
# 不执行 SAM3、pocket、G/L、外观增强、鱼眼矫正或其他离线语义处理。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DSL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATA_ROOT="${DATA_ROOT:-/share/project/liyuanyuan/data/dexglove_data/pkl_dataset/pick_bread_head_2}"
DATASET_DIR="${DATASET_DIR:-/share/project/liyuanyuan/data/dexglove_data/dex_data/wuji/pick_bread_head_2_pico_only_ego_v1}"
DATASET_PATH="${DATASET_DIR}/dataset.zarr.zip"
CONVERTER_ROOT="${CONVERTER_ROOT:-/home/zjc/Desktop/human2dex}"
CONVERTER_PY="${CONVERTER_PY:-/share/project/liyuanyuan/anaconda3/envs/sam3/bin/python}"
ACCELERATE_BIN="${ACCELERATE_BIN:-/share/project/lsq/miniconda3/envs/rdp/bin/accelerate}"

RGB_FIELD="l515.ego.rgbImage"
GRIPPER_SOURCE="wuji_command"
POSE_SOURCE="${TRAJECTORY_POSE_SOURCE:-trajectoryPose}"
CONVERT_WORKERS="${CONVERT_WORKERS:-16}"
MAX_INFLIGHT_TASKS="${MAX_INFLIGHT_TASKS:-32}"
NUM_GPUS="${NUM_GPUS:-8}"
GEOMETRY_CROP_RATIO="${GEOMETRY_CROP_RATIO:-0.95}"
N_ACTION_STEPS="${N_ACTION_STEPS:-16}"
OBS_DOWN_SAMPLE_STEPS="${OBS_DOWN_SAMPLE_STEPS:-3}"
ACTION_DOWN_SAMPLE_STEPS="${ACTION_DOWN_SAMPLE_STEPS:-3}"
ACTION_HORIZON="${ACTION_HORIZON:-16}"
NUM_EPOCHS="${NUM_EPOCHS:-500}"
BATCH_SIZE="${BATCH_SIZE:-32}"
USE_RATIO="${USE_RATIO:-1.0}"
VISION_MODEL="${VISION_MODEL:-vit_base_patch14_reg4_dinov2.lvd142m}"

TASK_NAME="wuji_pick_bread_head_2_pico_only_ego"
RUN_TAG="pick_bread_head_2_pico_only_ego_wuji"

export WANDB_MODE="${WANDB_MODE:-offline}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

mkdir -p "${DATASET_DIR%/*}"
if [[ ! -f "${DATASET_PATH}" || "${FORCE_REBUILD:-0}" == "1" ]]; then
  echo "== Convert raw PKL -> Zarr (Wuji / ego camera) =="
  "${CONVERTER_PY}" "${CONVERTER_ROOT}/tools/convert_pkl_to_training_zarr.py" \
    --input "${DATA_ROOT}" \
    --output "${DATASET_DIR}" \
    --overwrite \
    --rgb-image-field "${RGB_FIELD}" \
    --gripper-source "${GRIPPER_SOURCE}" \
    --trajectory-pose-source "${POSE_SOURCE}" \
    --data-scaling-laws-root "${DSL_ROOT}" \
    --workers "${CONVERT_WORKERS}" \
    --max-inflight-tasks "${MAX_INFLIGHT_TASKS}" \
    --opencv-threads 1 \
    --blosc-threads 1 \
    --image-batch-size 16 \
    --image-compressor blosc
else
  echo "== Reuse existing Zarr: ${DATASET_PATH} =="
fi

[[ -f "${DATASET_PATH}" ]] || { echo "missing dataset: ${DATASET_PATH}" >&2; exit 2; }

logging_time="$(date "+%d-%H.%M.%S")"
now_date="$(date "+%Y.%m.%d")"
run_dir="${RUN_DIR:-data/outputs/${now_date}/${logging_time}_${RUN_TAG}}"

echo "dataset: ${DATASET_PATH}"
echo "camera field: ${RGB_FIELD}"
echo "Pico hand source: wuji_command (20D)"
echo "trajectory pose source: ${POSE_SOURCE}"
echo "run_dir: ${run_dir}"

cd "${SCRIPT_DIR}"
"${ACCELERATE_BIN}" launch --num_processes "${NUM_GPUS}" ../train.py \
  --config-name=train_diffusion_unet_timm_umi_workspace_wuji \
  task.name="${TASK_NAME}" \
  multi_run.run_dir="${run_dir}" \
  multi_run.wandb_name_base="${logging_time}_${RUN_TAG}" \
  hydra.run.dir="${run_dir}" \
  hydra.sweep.dir="${run_dir}" \
  task.dataset_path="${DATASET_PATH}" \
  +task.dataset.val_grouping=manifest_episode \
  task.dataset.val_ratio="${VAL_RATIO:-0.05}" \
  n_action_steps="${N_ACTION_STEPS}" \
  task.obs_down_sample_steps="${OBS_DOWN_SAMPLE_STEPS}" \
  task.action_down_sample_steps="${ACTION_DOWN_SAMPLE_STEPS}" \
  task.action_horizon="${ACTION_HORIZON}" \
  policy.obs_encoder.transforms.0.ratio="${GEOMETRY_CROP_RATIO}" \
  training.num_epochs="${NUM_EPOCHS}" \
  dataloader.batch_size="${BATCH_SIZE}" \
  logging.mode="${WANDB_MODE}" \
  logging.name="${logging_time}_${RUN_TAG}" \
  policy.obs_encoder.model_name="${VISION_MODEL}" \
  task.dataset.use_ratio="${USE_RATIO}"
