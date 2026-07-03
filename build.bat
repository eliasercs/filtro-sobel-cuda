@echo off
setlocal

set "CCBIN=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64\cl.exe"

if not exist build mkdir build

echo === Compilando version secuencial (CPU) ===
nvcc -std=c++17 -arch=native -ccbin "%CCBIN%" src\image.cpp src\main.cpp -o build\secuencial.exe
if errorlevel 1 goto :error

echo === Compilando version CUDA clasica (sin Tile) ===
nvcc -std=c++17 -arch=native -ccbin "%CCBIN%" src\image.cpp src\cuda_kernels.cu src\main_cuda.cpp -o build\cuda.exe
if errorlevel 1 goto :error

echo.
echo Compilacion completa:
echo   build\secuencial.exe
echo   build\cuda.exe
exit /b 0

:error
echo.
echo ERROR durante la compilacion.
exit /b 1
