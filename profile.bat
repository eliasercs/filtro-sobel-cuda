@echo off
setlocal

set "NSYS=C:\Program Files\NVIDIA\Nsight Systems 2026.1.3\target-windows-x64\nsys.exe"
set "NCU=C:\Program Files\NVIDIA Corporation\Nsight Compute 2026.2.0\host\target-windows-x64\ncu.exe"

set "PERF_DIR=..\results\perf"
if not exist %PERF_DIR% mkdir %PERF_DIR%

set "INSTANCE=medium"
set "KERNEL=5"
set "SCALE=0.5"

if not "%~1"=="" set "INSTANCE=%~1"
if not "%~2"=="" set "KERNEL=%~2"
if not "%~3"=="" set "SCALE=%~3"

echo === Profiling con Nsight Systems ===
"%NSYS%" profile ^
  --output=%PERF_DIR%\nsys_%INSTANCE%_cuda ^
  --force-overwrite=true ^
  ..\build\cuda.exe --instance=%INSTANCE% --kernel-size=%KERNEL% --scale=%SCALE%
if errorlevel 1 echo [nsys] falló, continuando...

echo.
echo === Profiling con Nsight Compute ===
"%NCU%" --set full --target-processes all ^
  --output %PERF_DIR%\ncu_%INSTANCE%_cuda ^
  ..\build\cuda.exe --instance=%INSTANCE% --kernel-size=%KERNEL% --scale=%SCALE%
if errorlevel 1 echo [ncu] falló, continuando...

echo.
echo === Perfiles generados en %PERF_DIR% ===
dir %PERF_DIR%

endlocal
