#!/usr/bin/env bash

# [1/3] SAM3 mask
# [2/3] wrist MANO21 + projection + fusion on original PKLs
# [3/3] augmentation + prediction sync + skeleton render

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-/share/project/liyuanyuan/anaconda3/envs/sam3/bin/python}"

# 配置文件路径。
# 常规情况下不需要改；如果你为别的数据集单独复制了 YAML，再通过环境变量覆盖。
MASK_CONFIG="${MASK_CONFIG:-${SCRIPT_DIR}/01_generate_sam3_masks.yaml}"
AUG_CONFIG="${AUG_CONFIG:-${SCRIPT_DIR}/02_augment_pick_6_pro_camera_mount.yaml}"

# 数据根目录和输入/输出路径。
# 常改：INPUT_ROOT，用于切换原始 PKL 数据集。
# 常改：MASK_OUTPUT/AUG_OUTPUT/QC_OUTPUT，用于把不同实验版本输出到独立目录。
# 注意：AUG_OUTPUT 不要设成 INPUT_ROOT，避免增强图片写回原始数据。
DATA_ROOT="${DATA_ROOT:-/share/project/liyuanyuan/data/dexglove_data/pkl_dataset}"
INPUT_ROOT="${INPUT_ROOT:-/share/project/liyuanyuan/data/dexglove_data/pkl_dataset/pick_cube/pick_cube}"
MASK_OUTPUT="${MASK_OUTPUT:-${DATA_ROOT}/pick_cube/pick_cube_sam3_masks}"
AUG_OUTPUT="${AUG_OUTPUT:-${DATA_ROOT}/pick_cube/pick_cube_cam_mount_aug_v2}"
QC_OUTPUT="${QC_OUTPUT:-${DATA_ROOT}/pick_cube/pick_cube_cam_mount_aug_qc_v2}"

# 要处理的图像流。默认只处理每个 episode 下的 images/。
# 多路相机用逗号分隔，例如：IMAGE_SUBDIRS=images,l515/ego/rgb
IMAGE_SUBDIRS="${IMAGE_SUBDIRS:-images}"

# SAM3 mask 生成使用的 GPU 和 batch 参数。
# 常改：CUDA_VISIBLE_DEVICES / NUM_GPUS，用于指定可见 GPU。
# 显存不足：先把 MASK_BATCH_SIZE 从 8 降到 4、2、1。
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NUM_GPUS="${NUM_GPUS:-8}"
MASK_BATCH_SIZE="${MASK_BATCH_SIZE:-32}"
MASK_PREFETCH="${MASK_PREFETCH:-8}"

# Wrist 推理 + fusion + 2D projection 参数。
# 该阶段只在原始数据上跑 GPU wrist 模型；增强数据只同步字段和骨架，不再重复推理。
STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT:-${REPO_ROOT}/wrist/outputs/test_3/runs/stage2_mano_onestage_v1/checkpoints/best.pt}"
PROJECTION_HEAD_CHECKPOINT="${PROJECTION_HEAD_CHECKPOINT:-/home/zjc/Desktop/human2dex/wrist/outputs/stage2_projection_head_residual2d/proj_head_mano21_residual2d_grouped_20260730_200049/checkpoints/best.pt}"
WRIST_DEVICES="${WRIST_DEVICES:-auto}"
WRIST_BATCH_SIZE="${WRIST_BATCH_SIZE:-128}"
WRIST_NUM_WORKERS="${WRIST_NUM_WORKERS:-8}"
FUSION_WORKERS="${FUSION_WORKERS:-16}"
WRIST_LOG_EVERY="${WRIST_LOG_EVERY:-5}"
WRIST_OVERWRITE="${WRIST_OVERWRITE:-1}" 

# 增强数据生成参数。
# 常改：AUG_VARIANTS，表示每个原始 episode 生成几个增强 episode。
# 常改：AUG_WORKERS，CPU 并行进程数；磁盘压力大或调试时设 1。
# 少改：AUG_EPISODE_SUFFIX，控制输出 episode 后缀。
AUG_VARIANTS="${AUG_VARIANTS:-3}"
AUG_EPISODE_SUFFIX="${AUG_EPISODE_SUFFIX:-_cam_mount_aug}"
AUG_WORKERS="${AUG_WORKERS:-8}"
AUG_IO_WORKERS="${AUG_IO_WORKERS:-2}"
AUG_MP_START_METHOD="${AUG_MP_START_METHOD:-fork}"

# 调试/小样本限制。
# LIMIT_EPISODES=1 LIMIT_FRAMES=80 适合先做 QC。
# EPISODE=episode_0001 用于只处理指定 episode。
# QC_FRAMES 控制每个 episode 保存多少张 overlay 质检图。
LIMIT_EPISODES="${LIMIT_EPISODES:-}"
LIMIT_FRAMES="${LIMIT_FRAMES:-}"
EPISODE="${EPISODE:-}"
QC_FRAMES="${QC_FRAMES:-8}"

# 覆盖和跳过开关。
# MASK_OVERWRITE=1：覆盖已有 mask，适合调 mask 参数后重跑。
# AUG_OVERWRITE=1：覆盖已有增强图片，适合调相机扰动/颜色增强后重跑。
# SKIP_MASK=1：跳过 mask 生成，直接复用 MASK_OUTPUT/masks。
# SKIP_WRIST=1：跳过原始数据 wrist/fusion/projection，复用已有 wrist_* / fused_* 字段。
# SKIP_AUG=1：只生成 mask / wrist 字段，不生成增强数据。
# DRY_RUN=1：只打印将要执行的命令，不真正运行。
MASK_OVERWRITE="${MASK_OVERWRITE:-0}"
AUG_OVERWRITE="${AUG_OVERWRITE:-0}"
SKIP_MASK="${SKIP_MASK:-0}"
SKIP_WRIST="${SKIP_WRIST:-0}"
SKIP_AUG="${SKIP_AUG:-0}"
SYNC_PREDICTIONS="${SYNC_PREDICTIONS:-1}"
RENDER_WRIST_SKELETON="${RENDER_WRIST_SKELETON:-1}"
DRY_RUN="${DRY_RUN:-0}"

usage() {
  cat <<'EOF'
Run the full DexUMI wrist-view visual augmentation pipeline:
  1) generate SAM3 human-hand masks;
  2) run wrist RGB -> MANO21 + projection head + PICO fusion on original PKLs;
  3) run mask-guided color/material augmentation plus camera-mount perturbation,
     sync wrist/fused fields into augmented PKLs, and render transformed skeletons.

Default target:
  INPUT_ROOT=/share/project/liyuanyuan/data/dexglove_data/pkl_dataset/pick_6_pro
  MASK_OUTPUT=/share/project/liyuanyuan/data/dexglove_data/pkl_dataset/pick_6_pro_sam3_masks
  AUG_OUTPUT=/share/project/liyuanyuan/data/dexglove_data/pkl_dataset/pick_6_pro_cam_mount_aug
  QC_OUTPUT=/share/project/liyuanyuan/data/dexglove_data/pkl_dataset/pick_6_pro_cam_mount_aug_qc

Quick QC run:
  LIMIT_EPISODES=1 LIMIT_FRAMES=80 AUG_WORKERS=1 MASK_OVERWRITE=1 AUG_OVERWRITE=1 \
  bash glove_aug_pipeline/run_pick_6_pro_camera_mount_pipeline.sh

Full run:
  bash glove_aug_pipeline/run_pick_6_pro_camera_mount_pipeline.sh

Useful overrides:
  INPUT_ROOT=/path/to/pkl_dataset      # 原始 PKL/image 数据集根目录；切换任务时最常改。
  MASK_OUTPUT=/path/to/sam3_masks      # SAM3 mask 输出目录；脚本会读取其 masks/ 子目录。
  AUG_OUTPUT=/path/to/aug_dataset      # 增强后的数据集输出目录；不要和 INPUT_ROOT 相同。
  QC_OUTPUT=/path/to/aug_qc            # 增强 QC 输出目录，里面有 overlay 和 stats.json。
  IMAGE_SUBDIRS=images                 # 处理的图像流；多路相机用逗号分隔。
  EPISODE=episode_0001                 # 只处理一个 episode，适合定位问题。
  LIMIT_EPISODES=3                     # 只处理前 N 个 episode，适合小样本 QC。
  LIMIT_FRAMES=100                     # 每个 episode 只处理前 N 帧，适合快速看效果。
  QC_FRAMES=8                          # 每个 episode 保存多少张 QC overlay。
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 # 指定使用哪些 GPU。
  NUM_GPUS=8                           # SAM3 多进程使用 GPU 数量，应和可见 GPU 匹配。
  WRIST_DEVICES=auto                    # wrist 推理使用的 GPU；auto=所有可见 GPU。
  STAGE1_CHECKPOINT=/path/best.pt       # wrist MANO21 权重。
  PROJECTION_HEAD_CHECKPOINT=/path/best.pt # 2D projection head 权重。
  WRIST_BATCH_SIZE=128                  # wrist 推理 batch。
  FUSION_WORKERS=16                     # CPU fusion/retarget worker 数。
  MASK_BATCH_SIZE=8                    # SAM3 batch；OOM 时降到 4/2/1。
  AUG_VARIANTS=3                       # 每个原始 episode 生成几个增强 episode。
  AUG_WORKERS=8                        # 增强阶段 CPU worker；调试或磁盘慢时设 1。
  MASK_OVERWRITE=1                     # 覆盖已有 mask。
  AUG_OVERWRITE=1                      # 覆盖已有增强图片。
  SKIP_MASK=1                          # 跳过 mask 生成，复用已有 MASK_OUTPUT/masks。
  SKIP_WRIST=1                         # 跳过 wrist/fusion/projection，复用已有字段。
  SKIP_AUG=1                           # 只生成 mask，不做增强。
  SYNC_PREDICTIONS=1                   # 增强 PKL 同步原始 wrist_* / fused_*。
  RENDER_WRIST_SKELETON=1              # 在最终增强 RGB 上画骨架。
  DRY_RUN=1                            # 只打印命令，不实际执行。

Notes:
  - IMAGE_SUBDIRS is comma-separated.
  - Original pts21_mano / o6_command / wuji_command are preserved.
  - Generated wrist_* / fused_* fields are additive and refreshable.
  - Camera-mount perturbation is configured in 02_augment_pick_6_pro_camera_mount.yaml.
EOF
}

truthy() {
  case "${1:-0}" in
    1|true|TRUE|yes|YES|y|Y|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

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

append_common_limits() {
  local -n out_array=$1
  if [[ -n "${LIMIT_EPISODES}" ]]; then
    out_array+=(--limit-episodes "${LIMIT_EPISODES}")
  fi
  if [[ -n "${LIMIT_FRAMES}" ]]; then
    out_array+=(--limit-frames "${LIMIT_FRAMES}")
  fi
  if [[ -n "${EPISODE}" ]]; then
    out_array+=(--episode "${EPISODE}")
  fi
}

append_wrist_limits() {
  local -n out_array=$1
  if [[ -n "${LIMIT_EPISODES}" ]]; then
    out_array+=(--limit-pkls "${LIMIT_EPISODES}")
  fi
  if [[ -n "${LIMIT_FRAMES}" ]]; then
    out_array+=(--limit-frames "${LIMIT_FRAMES}")
  fi
  if [[ -n "${EPISODE}" ]]; then
    echo "Warning: EPISODE is not supported by wrist fusion stage; augmentation/mask stages still use it." >&2
  fi
}

print_cmd() {
  printf '%q ' "$@"
  printf '\n'
}

run_cmd() {
  print_cmd "$@"
  if ! truthy "${DRY_RUN}"; then
    "$@"
  fi
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

cd "${REPO_ROOT}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python not found or not executable: ${PYTHON_BIN}" >&2
  exit 2
fi
if [[ ! -f "${MASK_CONFIG}" ]]; then
  echo "Mask config not found: ${MASK_CONFIG}" >&2
  exit 2
fi
if [[ ! -f "${AUG_CONFIG}" ]]; then
  echo "Aug config not found: ${AUG_CONFIG}" >&2
  exit 2
fi
if [[ ! -d "${INPUT_ROOT}" ]]; then
  echo "Input dataset root not found: ${INPUT_ROOT}" >&2
  exit 2
fi
if ! truthy "${SKIP_WRIST}"; then
  if [[ ! -f "${STAGE1_CHECKPOINT}" ]]; then
    echo "Stage1 checkpoint not found: ${STAGE1_CHECKPOINT}" >&2
    exit 2
  fi
  if [[ ! -f "${PROJECTION_HEAD_CHECKPOINT}" ]]; then
    echo "Projection head checkpoint not found: ${PROJECTION_HEAD_CHECKPOINT}" >&2
    exit 2
  fi
fi

export CUDA_VISIBLE_DEVICES

mask_cmd=(
  "${PYTHON_BIN}"
  "${SCRIPT_DIR}/01_generate_sam3_masks.py"
  --config "${MASK_CONFIG}"
  --input "${INPUT_ROOT}"
  --output "${MASK_OUTPUT}"
  --num-gpus "${NUM_GPUS}"
  --batch-size "${MASK_BATCH_SIZE}"
  --prefetch "${MASK_PREFETCH}"
  --qc-frames "${QC_FRAMES}"
)
add_image_subdirs mask_cmd "${IMAGE_SUBDIRS}"
append_common_limits mask_cmd
if truthy "${MASK_OVERWRITE}"; then
  mask_cmd+=(--overwrite)
fi

wrist_cmd=(
  "${PYTHON_BIN}"
  "${REPO_ROOT}/tools/add_wrist_fusion_predictions_to_pkl.py"
  --data-root "${INPUT_ROOT}"
  --checkpoint "${STAGE1_CHECKPOINT}"
  --projection-head-checkpoint "${PROJECTION_HEAD_CHECKPOINT}"
  --model-type stage2
  --devices "${WRIST_DEVICES}"
  --batch-size "${WRIST_BATCH_SIZE}"
  --num-workers "${WRIST_NUM_WORKERS}"
  --fusion-workers "${FUSION_WORKERS}"
  --log-every "${WRIST_LOG_EVERY}"
)
append_wrist_limits wrist_cmd

aug_cmd=(
  "${PYTHON_BIN}"
  "${SCRIPT_DIR}/02_augment_dataset.py"
  --config "${AUG_CONFIG}"
  --input "${INPUT_ROOT}"
  --output "${AUG_OUTPUT}"
  --mask-root "${MASK_OUTPUT%/}/masks"
  --qc-output "${QC_OUTPUT}"
  --variants "${AUG_VARIANTS}"
  --episode-suffix "${AUG_EPISODE_SUFFIX}"
  --workers "${AUG_WORKERS}"
  --io-workers "${AUG_IO_WORKERS}"
  --mp-start-method "${AUG_MP_START_METHOD}"
  --qc-frames "${QC_FRAMES}"
  --camera-mount-aug-enabled
)
add_image_subdirs aug_cmd "${IMAGE_SUBDIRS}"
append_common_limits aug_cmd
if truthy "${SYNC_PREDICTIONS}"; then
  aug_cmd+=(--sync-predictions-from "${INPUT_ROOT}")
fi
if truthy "${RENDER_WRIST_SKELETON}"; then
  aug_cmd+=(--render-wrist-skeleton)
fi
if truthy "${AUG_OVERWRITE}"; then
  aug_cmd+=(--overwrite)
fi

echo "Repo:        ${REPO_ROOT}"
echo "Input:       ${INPUT_ROOT}"
echo "Mask output: ${MASK_OUTPUT}"
echo "Aug output:  ${AUG_OUTPUT}"
echo "QC output:   ${QC_OUTPUT}"
echo "Images:      ${IMAGE_SUBDIRS}"
echo "CUDA:        ${CUDA_VISIBLE_DEVICES}"
echo "Stage1 ckpt: ${STAGE1_CHECKPOINT}"
echo "Proj ckpt:   ${PROJECTION_HEAD_CHECKPOINT}"
if truthy "${DRY_RUN}"; then
  echo "Mode:        DRY_RUN"
fi

if ! truthy "${SKIP_MASK}"; then
  echo
  echo "[1/3] Generate SAM3 masks"
  run_cmd "${mask_cmd[@]}"
else
  echo
  echo "[1/3] Generate SAM3 masks: skipped"
fi

if ! truthy "${SKIP_WRIST}"; then
  echo
  echo "[2/3] Run wrist MANO21 + projection + fusion on original PKLs"
  run_cmd "${wrist_cmd[@]}"
else
  echo
  echo "[2/3] Run wrist MANO21 + projection + fusion: skipped"
fi

if ! truthy "${SKIP_AUG}"; then
  echo
  echo "[3/3] Generate augmented dataset + sync fields + render skeleton"
  run_cmd "${aug_cmd[@]}"
else
  echo
  echo "[3/3] Generate augmented dataset: skipped"
fi

echo
echo "Done."
echo "Mask overlays: ${MASK_OUTPUT%/}/overlays"
echo "Mask stats:    ${MASK_OUTPUT%/}/stats"
echo "Aug QC:        ${QC_OUTPUT}"
