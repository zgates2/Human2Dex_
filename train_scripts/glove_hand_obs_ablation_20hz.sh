#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

mode="${HAND_OBS_MODE:-clean}"
dataset_path="${DATASET_PATH:-/share/project/liyuanyuan/data/dexglove_data/dex_data/linker_o6/pick_6_mix_5_wrist/dataset.zarr.zip}"
# 17_1A100 exposes one A100. Historical scripts launched eight workers onto
# that single visible GPU; new paper runs default to one process.
num_processes="${NUM_PROCESSES:-1}"
num_epochs="${NUM_EPOCHS:-200}"
batch_size="${BATCH_SIZE:-32}"
dry_run="${DRY_RUN:-0}"
accelerate_bin="${ACCELERATE_BIN:-/share/project/lsq/miniconda3/envs/rdp/bin/accelerate}"

if [[ ! -x "${accelerate_bin}" ]]; then
  echo "Accelerate executable not found: ${accelerate_bin}" >&2
  exit 2
fi

n_action_steps=16
obs_down_sample_steps=3
action_down_sample_steps=3
action_horizon=16

ignore_hand=false
time_shift=0
augment_enabled=false
dropout_prob=0.0
bias_std=0.0
noise_std=0.0

case "${mode}" in
  clean)
    ;;
  no_hand)
    ignore_hand=true
    ;;
  noise)
    augment_enabled=true
    bias_std="${HAND_BIAS_STD:-0.05}"
    noise_std="${HAND_NOISE_STD:-0.02}"
    ;;
  noise_dropout)
    augment_enabled=true
    dropout_prob="${HAND_DROPOUT_PROB:-0.10}"
    bias_std="${HAND_BIAS_STD:-0.05}"
    noise_std="${HAND_NOISE_STD:-0.02}"
    ;;
  stale1)
    time_shift=-1
    ;;
  stale2)
    time_shift=-2
    ;;
  stale4)
    time_shift=-4
    ;;
  *)
    echo "Unsupported HAND_OBS_MODE=${mode}" >&2
    echo "Expected: clean, no_hand, noise, noise_dropout, stale1, stale2, stale4" >&2
    exit 2
    ;;
esac

export WANDB_MODE=offline
export HF_HOME=/root/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/root/.cache/huggingface/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

logging_time="$(date '+%d-%H.%M.%S')"
now_date="$(date '+%Y.%m.%d')"
task_name="linker_hand_obs_${mode}_sample3"
run_dir="data/outputs/${now_date}/${logging_time}_${task_name}"

cmd=(
  "${accelerate_bin}" launch --num_processes "${num_processes}" ../train.py
  --config-name=train_diffusion_unet_timm_umi_workspace_linker_o6
  "multi_run.run_dir=${run_dir}"
  "multi_run.wandb_name_base=${logging_time}"
  "hydra.run.dir=${run_dir}"
  "hydra.sweep.dir=${run_dir}"
  "task.dataset_path=${dataset_path}"
  "task.ignore_proprioception=true"
  "task.ignore_hand_observation=${ignore_hand}"
  "task.hand_obs_time_shift_steps=${time_shift}"
  "task.dataset.val_grouping=manifest_episode"
  "task.dataset.seed=42"
  "n_action_steps=${n_action_steps}"
  "task.obs_down_sample_steps=${obs_down_sample_steps}"
  "task.action_down_sample_steps=${action_down_sample_steps}"
  "task.action_horizon=${action_horizon}"
  "training.num_epochs=${num_epochs}"
  "dataloader.batch_size=${batch_size}"
  "checkpoint.only_save_recent=true"
  "logging.mode=offline"
  "logging.name=${logging_time}_${task_name}"
  "policy.obs_encoder.model_name=vit_base_patch14_reg4_dinov2.lvd142m"
  "policy.lowdim_obs_augmentation.enabled=${augment_enabled}"
  "policy.lowdim_obs_augmentation.dropout_prob=${dropout_prob}"
  "policy.lowdim_obs_augmentation.per_dimension_bias_std=${bias_std}"
  "policy.lowdim_obs_augmentation.element_noise_std=${noise_std}"
  "task.dataset.use_ratio=1.0"
)

printf 'HAND_OBS_MODE=%s\nDATASET_PATH=%s\nRUN_DIR=%s\n' \
  "${mode}" "${dataset_path}" "${run_dir}"
printf 'COMMAND:'
printf ' %q' "${cmd[@]}"
printf '\n'

if [[ "${dry_run}" == "1" ]]; then
  exit 0
fi

"${cmd[@]}"
