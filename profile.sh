#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
BUILD="$ROOT/build"
PERF_DIR="$ROOT/results/perf"
mkdir -p "$PERF_DIR"

INSTANCE="${1:-medium}"
KERNEL="${2:-5}"
SCALE="${3:-0.5}"

echo "=== Profiling con Nsight Systems ==="
cd "$BUILD"
nsys profile --output="$PERF_DIR/nsys_${INSTANCE}_cuda" --force-overwrite=true \
    ./cuda --instance="$INSTANCE" --kernel-size="$KERNEL" --scale="$SCALE" || echo "[nsys] fallo, continuando..."

echo ""
echo "=== Profiling con Nsight Compute ==="
ncu --set full --target-processes all --export "$PERF_DIR/ncu_${INSTANCE}_cuda" \
    ./cuda --instance="$INSTANCE" --kernel-size="$KERNEL" --scale="$SCALE" || echo "[ncu] fallo, continuando..."

if [ -f "$ROOT/results/resultados_full.csv" ]; then
    cp "$ROOT/results/resultados_full.csv" "$ROOT/results/resultados_full_pre_profile.csv"
fi
rm -f "$ROOT/results/resultados.csv"

cd "$ROOT"
echo ""
echo "=== Perfiles generados en $PERF_DIR ==="
ls -lh "$PERF_DIR"