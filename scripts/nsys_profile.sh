#!/usr/bin/env bash
set -euo pipefail

# STAGE=1 MODE=full CUDA_VISIBLE_DEVICES=0 GPUS=1 bash scripts/nsys_profile.sh
# STAGE=2 MODE=light CUDA_VISIBLE_DEVICES=0 GPUS=1 bash scripts/nsys_profile.sh

# Configurable envs
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"   # 选择使用的GPU
PORT="${PORT:-28651}"
GPUS="${GPUS:-1}"               # dist_train的nproc_per_node
STAGE="${STAGE:-1}"             # 1 或 2
MODE="${MODE:-full}"            # full 或 light
OUT_DIR="${OUT_DIR:-$ROOT_DIR/nsys_reports}"
# 采样窗口控制（可选）：延迟/持续时间（秒）。空则不设置
DELAY="${DELAY:-}"               # e.g. 15  → 预热15秒后开始采样
DURATION="${DURATION:-}"         # e.g. 30  → 采样30秒
# 可选：超时（硬性终止训练进程），仅当你希望程序自动结束时设置
TIMEOUT_SEC="${TIMEOUT_SEC:-}"   # e.g. 70  → 70秒后终止被测进程
mkdir -p "$OUT_DIR"

# Pick config by stage
case "$STAGE" in
  1) CONFIG="projects/configs/sparsedrive_small_stage1.py"; OUT_PREFIX="stage1";;
  2) CONFIG="projects/configs/sparsedrive_small_stage2.py"; OUT_PREFIX="stage2";;
  *) echo "Invalid STAGE=$STAGE (use 1 or 2)"; exit 1;;
esac

# Trace options by mode
case "$MODE" in
  full)  TRACE_OPTS="--trace=cuda,nvtx,cublas,cudnn,osrt --sample=cpu --cpuctxsw=none";;
  light) TRACE_OPTS="--trace=cuda,nvtx --sample=none --cpuctxsw=none";;
  *) echo "Invalid MODE=$MODE (use full or light)"; exit 1;;
esac

ts="$(date +%Y%m%d_%H%M%S)"
OUT_BASENAME="${OUT_PREFIX}_${MODE}_${ts}"
OUT_PATH="$OUT_DIR/$OUT_BASENAME"

NSYS_WINDOW_OPTS=""
[ -n "$DELAY" ] && NSYS_WINDOW_OPTS+=" --delay=$DELAY"
[ -n "$DURATION" ] && NSYS_WINDOW_OPTS+=" --duration=$DURATION"

echo "[nsys] Profiling: STAGE=$STAGE MODE=$MODE GPUS=$GPUS CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES DELAY=${DELAY:-none} DURATION=${DURATION:-none} TIMEOUT=${TIMEOUT_SEC:-none}"

TARGET_CMD="cd \"$ROOT_DIR\" && PORT=$PORT bash ./tools/dist_train.sh $CONFIG $GPUS --deterministic"
if [ -n "$TIMEOUT_SEC" ]; then
  TARGET_WRAPPER="timeout --preserve-status ${TIMEOUT_SEC}s bash -lc '$TARGET_CMD'"
else
  TARGET_WRAPPER="bash -lc '$TARGET_CMD'"
fi

nsys profile \
  --output="$OUT_PATH" \
  $TRACE_OPTS \
  $NSYS_WINDOW_OPTS \
  --force-overwrite=true \
  bash -lc "$TARGET_WRAPPER"

# Generate HTML stats (summary + CUDA API + KERNEL)
REPORT_FILE=""
if [ -f "${OUT_PATH}.qdrep" ]; then
  REPORT_FILE="${OUT_PATH}.qdrep"
elif [ -f "${OUT_PATH}.nsys-rep" ]; then
  REPORT_FILE="${OUT_PATH}.nsys-rep"
fi

if [ -n "$REPORT_FILE" ]; then
  nsys stats --report summary,CUDA_API,CUDA_KERNEL \
    --format html --output "${OUT_PATH}_stats" "$REPORT_FILE" || true
  echo "Report: $REPORT_FILE"
  echo "HTML stats: ${OUT_PATH}_stats.html"
else
  echo "No report file found at ${OUT_PATH}.qdrep or ${OUT_PATH}.nsys-rep"
fi
