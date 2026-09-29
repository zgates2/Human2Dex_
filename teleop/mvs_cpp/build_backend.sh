#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BUILD_DIR="${SCRIPT_DIR}/build"
MVS_SDK_ROOT_VALUE="${MVS_SDK_ROOT:-/opt/MVS}"
CMAKE_BUILD_TYPE_VALUE="${CMAKE_BUILD_TYPE:-Release}"
PYTHON_EXECUTABLE_VALUE="${MVS_PYTHON_EXECUTABLE:-$(command -v python3)}"

if [[ -z "${PYTHON_EXECUTABLE_VALUE}" ]]; then
  echo "[build] python3 not found; set MVS_PYTHON_EXECUTABLE explicitly" >&2
  exit 1
fi

if [[ -f "${BUILD_DIR}/CMakeCache.txt" ]]; then
  CACHED_SOURCE_DIR=$(sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' "${BUILD_DIR}/CMakeCache.txt" | tail -n 1)
  if [[ -n "${CACHED_SOURCE_DIR}" && "${CACHED_SOURCE_DIR}" != "${SCRIPT_DIR}" ]]; then
    echo "[build] Stale CMake cache detected:"
    echo "        cache source = ${CACHED_SOURCE_DIR}"
    echo "        current source = ${SCRIPT_DIR}"
    echo "[build] Removing ${BUILD_DIR} and reconfiguring..."
    rm -rf "${BUILD_DIR}"
  fi
  if [[ -f "${BUILD_DIR}/CMakeCache.txt" ]]; then
    CACHED_BUILD_TYPE=$(sed -n 's/^CMAKE_BUILD_TYPE:STRING=//p' "${BUILD_DIR}/CMakeCache.txt" | tail -n 1)
    if [[ "${CACHED_BUILD_TYPE}" != "${CMAKE_BUILD_TYPE_VALUE}" ]]; then
      echo "[build] CMAKE_BUILD_TYPE changed: ${CACHED_BUILD_TYPE:-<empty>} -> ${CMAKE_BUILD_TYPE_VALUE}"
      echo "[build] Removing ${BUILD_DIR} and reconfiguring..."
      rm -rf "${BUILD_DIR}"
    fi
  fi
fi

cmake \
  -U "_Python_*" \
  -U "FIND_PACKAGE_MESSAGE_DETAILS_Python" \
  -S "${SCRIPT_DIR}" \
  -B "${BUILD_DIR}" \
  -DMVS_SDK_ROOT="${MVS_SDK_ROOT_VALUE}" \
  -DPython_EXECUTABLE="${PYTHON_EXECUTABLE_VALUE}" \
  -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE_VALUE}"
cmake --build "${BUILD_DIR}" --config Release
