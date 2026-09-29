#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"


task_name="linker_pick_cube_human2dex_gl_skeleton_tcp_v2"
n_action_steps=16
obs_down_sample_steps=3      #降采样
action_down_sample_steps=3
action_horizon=16
# dataset_path="${DATASET_PATH:-/share/project/liyuanyuan/data/dexglove_data/dex_data/linker_o6/pick_3/dataset.zarr.zip}"
dataset_path="${DATASET_PATH:-/share/project/liyuanyuan/data/dexglove_data/dex_data/linker_o6/pick_cube_human2dex_gl_skeleton_tcp_v2/dataset.zarr.zip}"
dual_view_mode="${DUAL_VIEW_MODE:-gl}"         # gl | g_only          G + L patch 融合 | 仅主视角

export WANDB_MODE=offline
export HF_ENDPOINT=https://hf-mirror.com

logging_time=$(date "+%d-%H.%M.%S")
now_seconds="${logging_time: -8}"
now_date=$(date "+%Y.%m.%d")
# run_dir="data/outputs/${now_date}/${now_seconds}"
run_dir="data/outputs/${now_date}/${now_seconds}_${task_name}"
echo ${run_dir}
echo "${dataset_path}"

export HF_HOME=/root/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/root/.cache/huggingface/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

accelerate launch --num_processes 8 ../train.py \
--config-name=train_diffusion_unet_timm_umi_workspace_linker_o6_gl \
multi_run.run_dir=${run_dir} multi_run.wandb_name_base=${logging_time} hydra.run.dir=${run_dir} hydra.sweep.dir=${run_dir} \
task.dataset_path=${dataset_path} \
n_action_steps=${n_action_steps} \
task.obs_down_sample_steps=${obs_down_sample_steps} \
task.action_down_sample_steps=${action_down_sample_steps} \
task.action_horizon=${action_horizon} \
training.num_epochs=500 \
dataloader.batch_size=32 \
logging.mode=offline \
logging.name="${logging_time}_${task_name}" \
policy.obs_encoder.model_name='vit_base_patch14_reg4_dinov2.lvd142m' \
policy.obs_encoder.dual_view_mode=${dual_view_mode} \
task.dataset.use_ratio=1.0