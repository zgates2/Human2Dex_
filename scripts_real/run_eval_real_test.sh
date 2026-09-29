#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
# CKPT_PATH="/home/zjc/Desktop/human2dex/ckpt/wipeboard/epoch0100.ckpt"      # 300条擦白板数据
CKPT_PATH="/mnt/nvme4t/OmniUMI/ckpt/2026.05.19/latest.ckpt"         # 120条擦白板数据
OUTPUT_DIR="${OUTPUT_DIR:-data_local/eval_test}"
ROBOT_CONFIG="${ROBOT_CONFIG:-example/eval_robots_config.yaml}"
FREQUENCY="${FREQUENCY:-20}"
MAX_TIMESTEPS="${MAX_TIMESTEPS:-3}"
HEADLESS="${HEADLESS:-1}"
DRY_RUN_ACTIONS="${DRY_RUN_ACTIONS:-0}"
INIT_JOINTS="${INIT_JOINTS:-1}"
PDB_BEFORE_JOINT_INIT="${PDB_BEFORE_JOINT_INIT:-0}"
PDB_BEFORE_PREDICT_ACTION="${PDB_BEFORE_PREDICT_ACTION:-0}"
PDB_BEFORE_POLICY_ACTION="${PDB_BEFORE_POLICY_ACTION:-0}"
DEBUG_ACTION_START_DELAY="${DEBUG_ACTION_START_DELAY:-0.3}"
DEBUG_SAVE_INFERENCE="${DEBUG_SAVE_INFERENCE:-0}"
DEBUG_SAVE_DIR="${DEBUG_SAVE_DIR:-${OUTPUT_DIR}/debug_inference}"
DEBUG_SAVE_EVERY="${DEBUG_SAVE_EVERY:-1}"
DEBUG_SAVE_PLOTS="${DEBUG_SAVE_PLOTS:-0}"
GRIPPER_ACTION_LEAD_STEPS="${GRIPPER_ACTION_LEAD_STEPS:-4}"
ROBOT_ACTION_STITCH_STEPS="${ROBOT_ACTION_STITCH_STEPS:-4}"
PREDICT_EVERY_STEP="${PREDICT_EVERY_STEP:-0}"
KEEP_ROBOT_QUEUE="${KEEP_ROBOT_QUEUE:-0}"
FRAME_LATENCY_S="${FRAME_LATENCY_S:-}"
TEMPORAL_AGG="${TEMPORAL_AGG:-0}"
ENSEMBLE_STEPS="${ENSEMBLE_STEPS:-8}"
HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

EXTRA_ARGS=()
if [[ "${HEADLESS}" != "0" ]]; then
  EXTRA_ARGS+=(--headless)
fi
if [[ "${DRY_RUN_ACTIONS}" != "0" ]]; then
  EXTRA_ARGS+=(--dry-run-actions)
fi
if [[ "${INIT_JOINTS}" != "0" ]]; then
  EXTRA_ARGS+=(-j)
fi
if [[ "${PDB_BEFORE_JOINT_INIT}" != "0" ]]; then
  EXTRA_ARGS+=(--pdb-before-joint-init)
fi
if [[ "${PDB_BEFORE_PREDICT_ACTION}" != "0" ]]; then
  EXTRA_ARGS+=(--pdb-before-predict-action)
fi
if [[ "${PDB_BEFORE_POLICY_ACTION}" != "0" ]]; then
  EXTRA_ARGS+=(--pdb-before-policy-action --debug-action-start-delay "${DEBUG_ACTION_START_DELAY}")
fi
if [[ "${DEBUG_SAVE_INFERENCE}" != "0" ]]; then
  EXTRA_ARGS+=(--debug-save-inference --debug-save-dir "${DEBUG_SAVE_DIR}" --debug-save-every "${DEBUG_SAVE_EVERY}")
fi
if [[ "${DEBUG_SAVE_PLOTS}" != "0" ]]; then
  EXTRA_ARGS+=(--debug-save-plots)
fi
EXTRA_ARGS+=(--gripper-action-lead-steps "${GRIPPER_ACTION_LEAD_STEPS}")
EXTRA_ARGS+=(--robot-action-stitch-steps "${ROBOT_ACTION_STITCH_STEPS}")
if [[ "${PREDICT_EVERY_STEP}" != "0" ]]; then
  EXTRA_ARGS+=(--predict-every-step)
fi
if [[ "${KEEP_ROBOT_QUEUE}" != "0" ]]; then
  EXTRA_ARGS+=(--keep-robot-queue)
fi
if [[ -n "${FRAME_LATENCY_S}" ]]; then
  EXTRA_ARGS+=(--frame-latency-s "${FRAME_LATENCY_S}")
fi
if [[ "${TEMPORAL_AGG}" != "0" ]]; then
  EXTRA_ARGS+=(--temporal_agg)
fi
EXTRA_ARGS+=(--ensemble_steps "${ENSEMBLE_STEPS}")
if [[ "${HF_HUB_OFFLINE}" != "0" ]]; then
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
fi

mkdir -p "$(dirname "${OUTPUT_DIR}")"

echo "Debug real-run defaults: FREQUENCY=${FREQUENCY}, GRIPPER_ACTION_LEAD_STEPS=${GRIPPER_ACTION_LEAD_STEPS}, ROBOT_ACTION_STITCH_STEPS=${ROBOT_ACTION_STITCH_STEPS}, PREDICT_EVERY_STEP=${PREDICT_EVERY_STEP}, KEEP_ROBOT_QUEUE=${KEEP_ROBOT_QUEUE}, FRAME_LATENCY_S=${FRAME_LATENCY_S:-auto}, TEMPORAL_AGG=${TEMPORAL_AGG}, ENSEMBLE_STEPS=${ENSEMBLE_STEPS}, HF_HUB_OFFLINE=${HF_HUB_OFFLINE}, DRY_RUN_ACTIONS=${DRY_RUN_ACTIONS}, INIT_JOINTS=${INIT_JOINTS}, PDB_BEFORE_JOINT_INIT=${PDB_BEFORE_JOINT_INIT}, PDB_BEFORE_PREDICT_ACTION=${PDB_BEFORE_PREDICT_ACTION}, PDB_BEFORE_POLICY_ACTION=${PDB_BEFORE_POLICY_ACTION}, DEBUG_SAVE_INFERENCE=${DEBUG_SAVE_INFERENCE}, DEBUG_SAVE_PLOTS=${DEBUG_SAVE_PLOTS}, MAX_TIMESTEPS=${MAX_TIMESTEPS}"

exec "${PYTHON_BIN}" eval_real.py \
  -i "${CKPT_PATH}" \
  -o "${OUTPUT_DIR}" \
  -rc "${ROBOT_CONFIG}" \
  -f "${FREQUENCY}" \
  -mt "${MAX_TIMESTEPS}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
