#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BUILD_DIR="${SCRIPT_DIR}/build"
MVS_SDK_ROOT_VALUE="${MVS_SDK_ROOT:-/opt/MVS}"

cmake -S "${SCRIPT_DIR}" -B "${BUILD_DIR}" -DMVS_SDK_ROOT="${MVS_SDK_ROOT_VALUE}"
cmake --build "${BUILD_DIR}" --config Release
