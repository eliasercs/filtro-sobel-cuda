@echo off
setlocal

set "CCBIN=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64\cl.exe"

if not exist build mkdir build

echo === Compilando version secuencial (CPU) ===
nvcc -std=c++17 -arch=native -allow-unsupported-compiler -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH -ccbin "%CCBIN%" src\image.cpp src\main.cpp -o build\secuencial.exe
if errorlevel 1 goto :error

echo === Compilando version CUDA clasica (sin Tile) ===
nvcc -std=c++17 -arch=native -allow-unsupported-compiler -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH -ccbin "%CCBIN%" src\image.cpp src\cuda_kernels.cu src\main_cuda.cpp -o build\cuda.exe
if errorlevel 1 goto :error

echo === Compilando version CUDA Tile C++ ===
nvcc -std=c++20 -arch=native -enable-tile -allow-unsupported-compiler -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH -ccbin "%CCBIN%" src\image.cpp src\tile_kernels.cu src\main_tile.cpp -o build\tile.exe
if errorlevel 1 (
  echo [WARNING] No se pudo compilar la version CUDA Tile C++ ^(requiere CUDA 13.x con -enable-tile^). Se omite esta version.
)

echo.
echo Compilacion completa:
echo   build\secuencial.exe
echo   build\cuda.exe
echo   build\tile.exe
exit /b 0

:error
echo.
echo ERROR durante la compilacion.
exit /b 1
