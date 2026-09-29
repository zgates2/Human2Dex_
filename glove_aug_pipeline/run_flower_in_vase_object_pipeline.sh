#!/usr/bin/env bash
# Run flower_in_vase from raw images/PKLs to O6 and Wuji Zarr.
set -euo pipefail

# Physical GPU 6 is in an uncorrectable ECC state on server-04; expose the
# seven healthy GPUs by default while still allowing callers to override it.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,7}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${SCRIPT_DIR}/00_pipeline_config_flower_in_vase_object_obs_tcp_v1.yaml"
FROM=1
TO=9
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage:
  bash glove_aug_pipeline/run_flower_in_vase_object_pipeline.sh [options]

Options:
  --from N        First stage, 1..9
  --to N          Last stage, 1..9
  --dry-run       Print stage commands without running model/data processing
  -h, --help      Show this help

Stage 09 builds both requested targets: O6 and Wuji.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --from) FROM="$2"; shift 2 ;;
    --to) TO="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if (( FROM < 1 || TO > 9 || FROM > TO )); then
  echo "Invalid stage range: --from ${FROM} --to ${TO}" >&2
  exit 2
fi

if (( FROM <= 8 )); then
  RUNNER_ARGS=(--config "${CONFIG}" --from "${FROM}")
  if (( TO < 8 )); then
    RUNNER_ARGS+=(--to "${TO}")
  else
    RUNNER_ARGS+=(--to 8)
  fi
  if (( DRY_RUN )); then
    RUNNER_ARGS+=(--dry-run)
  fi
  bash "${SCRIPT_DIR}/run_human2dex_object_pipeline.sh" "${RUNNER_ARGS[@]}"
fi

if (( FROM <= 9 && TO >= 9 )); then
  PY="${PYTHON_BIN:-/share/project/liyuanyuan/anaconda3/envs/sam3/bin/python}"
  COMMAND=(
    "${PY}"
    "${SCRIPT_DIR}/09_build_single_view_zarr.py"
    --config "${CONFIG}"
    --target both
  )
  if (( DRY_RUN )); then
    COMMAND+=(--dry-run)
  fi
  echo
  echo "========== Stage 9: 09_build_single_view_zarr.py (O6 + Wuji) =========="
  "${COMMAND[@]}"
fi

echo "flower_in_vase object pipeline stages ${FROM}..${TO} completed."
