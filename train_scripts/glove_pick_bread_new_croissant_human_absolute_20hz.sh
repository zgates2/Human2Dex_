#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

task_name="pick_bread_new_croissant_human_absolute_v1"
n_action_steps="${N_ACTION_STEPS:-16}"
obs_down_sample_steps="${OBS_DOWN_SAMPLE_STEPS:-3}"
action_down_sample_steps="${ACTION_DOWN_SAMPLE_STEPS:-3}"
action_horizon="${ACTION_HORIZON:-16}"
dataset_path="${DATASET_PATH:-/share/project/liyuanyuan/data/dexglove_data/dex_data/linker_o6/pick_bread_new_croissant_human_absolute_v1/dataset.zarr.zip}"
accelerate_bin="${ACCELERATE_BIN:-/share/project/lsq/miniconda3/envs/rdp/bin/accelerate}"

export WANDB_MODE="${WANDB_MODE:-offline}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

logging_time=$(date "+%d-%H.%M.%S")
now_seconds="${logging_time: -8}"
now_date=$(date "+%Y.%m.%d")
run_dir="${RUN_DIR:-data/outputs/${now_date}/${now_seconds}_${task_name}}"

echo "dataset: ${dataset_path}"
echo "run_dir: ${run_dir}"
echo "experiment: Human-Absolute"
echo "explicit observation: objectAbsoluteObs (5D, image-center reference)"
echo "auxiliary target: objectPocketObs (training only, never encoded)"

"${accelerate_bin}" launch --num_processes "${NUM_GPUS:-8}" ../train.py \
  --config-name=train_diffusion_unet_timm_umi_workspace_linker_o6_blue_cube_pocket \
  task=linker_o6_bread_human_absolute \
  task.name=linker_o6_bread_human_absolute \
  multi_run.run_dir="${run_dir}" \
  multi_run.wandb_name_base="${logging_time}" \
  hydra.run.dir="${run_dir}" \
  hydra.sweep.dir="${run_dir}" \
  task.dataset_path="${dataset_path}" \
  n_action_steps="${n_action_steps}" \
  task.obs_down_sample_steps="${obs_down_sample_steps}" \
  task.action_down_sample_steps="${action_down_sample_steps}" \
  task.action_horizon="${action_horizon}" \
  policy.object_pocket_aux_loss.enabled="${OBJECT_POCKET_AUX_ENABLED:-true}" \
  policy.object_pocket_aux_loss.obs_key=objectPocketObs \
  policy.object_pocket_aux_loss.weight="${OBJECT_POCKET_AUX_WEIGHT:-0.05}" \
  policy.object_pocket_aux_loss.min_confidence="${OBJECT_POCKET_AUX_MIN_CONFIDENCE:-0.3}" \
  training.seed="${TRAIN_SEED:-42}" \
  training.num_epochs="${NUM_EPOCHS:-300}" \
  dataloader.batch_size="${BATCH_SIZE:-32}" \
  logging.mode="${WANDB_MODE}" \
  logging.name="${logging_time}_${task_name}" \
  policy.obs_encoder.model_name="${VISION_MODEL:-vit_base_patch14_reg4_dinov2.lvd142m}" \
  task.dataset.use_ratio="${USE_RATIO:-1.0}"
