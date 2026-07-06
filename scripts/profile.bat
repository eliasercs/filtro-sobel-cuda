@echo off
setlocal

set "NSYS_EXE=C:\Program Files\NVIDIA Corporation\Nsight Systems 2026.1.3\target-windows-x64\nsys.exe"
set "NCU_EXE=C:\Program Files\NVIDIA Corporation\Nsight Compute 2026.2.0\ncu.bat"

set "ROOT=%~dp0.."
set "BUILD=%ROOT%\build"
set "PERF_DIR=%ROOT%\results\perf"
if not exist "%PERF_DIR%" mkdir "%PERF_DIR%"

set "INSTANCE=medium"
set "KERNEL=5"
set "SCALE=0.5"

if not "%~1"=="" set "INSTANCE=%~1"
if not "%~2"=="" set "KERNEL=%~2"
if not "%~3"=="" set "SCALE=%~3"

echo === Profiling con Nsight Systems ===
cd /d "%BUILD%"
"%NSYS_EXE%" profile --output="%PERF_DIR%\nsys_%INSTANCE%_cuda" --force-overwrite=true "cuda.exe" --instance=%INSTANCE% --kernel-size=%KERNEL% --scale=%SCALE%
if errorlevel 1 echo [nsys] fallo, continuando...

echo.
echo === Profiling con Nsight Compute ===
"%NCU_EXE%" --set full --target-processes all --export "%PERF_DIR%\ncu_%INSTANCE%_cuda" "cuda.exe" --instance=%INSTANCE% --kernel-size=%KERNEL% --scale=%SCALE%
if errorlevel 1 echo [ncu] fallo, continuando...

if exist "%ROOT%results\resultados_full.csv" (
    copy /Y "%ROOT%results\resultados_full.csv" "%ROOT%results\resultados_full_pre_profile.csv" >nul
)
if exist "%ROOT%results\resultados.csv" del /q "%ROOT%results\resultados.csv"

cd /d "%ROOT%"
echo.
echo === Perfiles generados en %PERF_DIR% ===
dir "%PERF_DIR%"

endlocal
