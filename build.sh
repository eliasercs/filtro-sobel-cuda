#!/usr/bin/env bash
set -euo pipefail

BUILD_DIR="$(cd "$(dirname "$0")" && pwd)/build"
SRC_DIR="$(cd "$(dirname "$0")" && pwd)/src"

mkdir -p "$BUILD_DIR"

echo "=== Compilando version secuencial (CPU) ==="
nvcc -std=c++17 -arch=native \
    "$SRC_DIR/image.cpp" "$SRC_DIR/main.cpp" -o "$BUILD_DIR/secuencial"

echo "=== Compilando version CUDA clasica (sin Tile) ==="
nvcc -std=c++17 -arch=native \
    "$SRC_DIR/image.cpp" "$SRC_DIR/cuda_kernels.cu" "$SRC_DIR/main_cuda.cpp" -o "$BUILD_DIR/cuda"

echo "=== Compilando version CUDA Tile C++ ==="
nvcc -std=c++20 -arch=native -enable-tile \
    "$SRC_DIR/image.cpp" "$SRC_DIR/tile_kernels.cu" "$SRC_DIR/main_tile.cpp" -o "$BUILD_DIR/tile"

echo ""
echo "Compilacion completa:"
echo "  $BUILD_DIR/secuencial"
echo "  $BUILD_DIR/cuda"
echo "  $BUILD_DIR/tile"