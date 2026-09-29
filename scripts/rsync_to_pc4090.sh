#!/usr/bin/env bash
# Sync Work/Data-Scaling-Laws -> pc-4090:/home/zjc/Desktop/human2dex
#
# Excludes git history, datasets, checkpoints, large media, and editor caches.
# Run from anywhere; SOURCE is repo-relative.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE="$(cd "${SCRIPT_DIR}/.." && pwd)/"
DEST_HOST="${UMI_DEPLOY_HOST:-pc-4090}"
DEST_PATH="${UMI_DEPLOY_PATH:-/home/zjc/Desktop/human2dex}"
DEST="${DEST_HOST}:${DEST_PATH}/"

echo "source : ${SOURCE}"
echo "dest   : ${DEST}"
echo

# Use the system ssh, not whatever zsh function might be in scope.
RSYNC_SSH="${RSYNC_SSH:-/usr/bin/ssh}"

# macOS ships rsync 2.6.9 which lacks --info=progress2; -P covers --partial + --progress.
exec /usr/bin/rsync \
  -avzhP \
  --delete \
  -e "${RSYNC_SSH}" \
  --exclude='.git/' \
  --exclude='data/' \
  --exclude='data_local/' \
  --exclude='outputs/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='*.pyo' \
  --exclude='*.egg-info/' \
  --exclude='.pytest_cache/' \
  --exclude='*.ckpt' \
  --exclude='*.pth' \
  --exclude='*.mp4' \
  --exclude='*.zarr/' \
  --exclude='.vscode/' \
  --exclude='.idea/' \
  --exclude='.DS_Store' \
  --exclude='umi/real_world/_mvs_cpp/build/' \
  --exclude='umi/real_world/_mvs_cpp/*.so' \
  "${SOURCE}" "${DEST}"
