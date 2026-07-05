#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
BUILD="$ROOT/build"
SRC="$ROOT/src"
RES="$ROOT/results"

INSTANCES="small medium large no-divisible"
KERNELS="5 9"
SCALES="0.5 1.75"

rm -f "$RES/resultados.csv"

cd "$SRC"

echo "=== CPU secuencial ==="
for I in $INSTANCES; do
  for K in $KERNELS; do
    for S in $SCALES; do
      echo "--- CPU $I k=$K s=$S ---"
      "$BUILD/secuencial" --instance="$I" --kernel-size="$K" --scale="$S"
    done
  done
done

echo ""
echo "=== CUDA clasico ==="
for I in $INSTANCES; do
  for K in $KERNELS; do
    for S in $SCALES; do
      echo "--- CUDA $I k=$K s=$S ---"
      "$BUILD/cuda" --instance="$I" --kernel-size="$K" --scale="$S"
    done
  done
done

echo ""
echo "=== CUDA Tile ==="
for I in $INSTANCES; do
  for K in $KERNELS; do
    for S in $SCALES; do
      echo "--- Tile $I k=$K s=$S ---"
      "$BUILD/tile" --instance="$I" --kernel-size="$K" --scale="$S"
    done
  done
done

echo ""
echo "=== cuTile Python ==="
for I in $INSTANCES; do
  for K in $KERNELS; do
    for S in $SCALES; do
      echo "--- cuTile $I k=$K s=$S ---"
      python3 "$SRC/cutile_pipeline.py" --instance="$I" --kernel-size="$K" --scale="$S"
    done
  done
done

echo ""
echo "=== Backup del CSV final ==="
cp "$RES/resultados.csv" "$RES/resultados_full.csv"
echo "CSV completo en $RES/resultados_full.csv"
wc -l < "$RES/resultados.csv" | xargs echo "Filas: "