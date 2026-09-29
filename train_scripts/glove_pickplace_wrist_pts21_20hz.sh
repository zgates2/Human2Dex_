#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"


task_name="pick_cube_wrist_pts21_tcp"
n_action_steps=16
obs_down_sample_steps=3   #降采样 
action_down_sample_steps=3
action_horizon=16
dataset_path="${DATASET_PATH:-/share/project/liyuanyuan/data/dexglove_data/dex_data/pts21/pick_cube_tcp/dataset.zarr.zip}"

export WANDB_MODE=offline #设置wandb为离线模式，避免在服务器上使用wandb时出现问题
export HF_ENDPOINT=https://hf-mirror.com

logging_time=$(date "+%d-%H.%M.%S")
now_seconds="${logging_time: -8}"
now_date=$(date "+%Y.%m.%d")
run_dir="data/outputs/${now_date}/${now_seconds}_${task_name}"
echo "${run_dir}"
echo "${dataset_path}"

export HF_HOME=/root/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/root/.cache/huggingface/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

accelerate launch --num_processes 8 ../train.py \
--config-name=train_diffusion_unet_timm_umi_workspace_wrist_pts21 \
multi_run.run_dir=${run_dir} multi_run.wandb_name_base=${logging_time} hydra.run.dir=${run_dir} hydra.sweep.dir=${run_dir} \
task.dataset_path=${dataset_path} \
n_action_steps=${n_action_steps} \
task.obs_down_sample_steps=${obs_down_sample_steps} \
task.action_down_sample_steps=${action_down_sample_steps} \
task.action_horizon=${action_horizon} \
'task.shape_meta.obs.robot0_gripper_width.shape=[63]' \
'task.shape_meta.action.shape=[72]' \
training.num_epochs=200 \
dataloader.batch_size=32 \
logging.mode=offline \
logging.name="${logging_time}_${task_name}" \
policy.obs_encoder.model_name='vit_base_patch14_reg4_dinov2.lvd142m' \
task.dataset.use_ratio=1.0
