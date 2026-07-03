@echo off
setlocal

set "ROOT=%~dp0"
set "OUT=%ROOT%filtro-sobel-cuda-release.zip"

if exist "%OUT%" del /q "%OUT%"

set "TARGET=%TEMP%\filtro-sobel-cuda-release"
if exist "%TARGET%" rmdir /s /q "%TARGET%"
mkdir "%TARGET%"

robocopy "%ROOT%AGENTS.md" "%TARGET%\" /NFL /NDL /NJH /NJS >nul
robocopy "%ROOT%TRACKING.md" "%TARGET%\" /NFL /NDL /NJH /NJS >nul
robocopy "%ROOT%README.md" "%TARGET%\" /NFL /NDL /NJH /NJS >nul
robocopy "%ROOT%Makefile" "%TARGET%\" /NFL /NDL /NJH /NJS >nul
robocopy "%ROOT%build.bat" "%TARGET%\" /NFL /NDL /NJH /NJS >nul
robocopy "%ROOT%run_all.bat" "%TARGET%\" /NFL /NDL /NJH /NJS >nul
robocopy "%ROOT%profile.bat" "%TARGET%\" /NFL /NDL /NJH /NJS >nul
robocopy "%ROOT%Actividad_4_INFO1195_2026_Actualizado.pdf" "%TARGET%\" /NFL /NDL /NJH /NJS >nul

robocopy "%ROOT%src" "%TARGET%\src" /E /NFL /NDL /NJH /NJS /XF "secuencial.exe" "cuda.exe" "tile.exe" "*.o" "*.obj" >nul
robocopy "%ROOT%data" "%TARGET%\data" /E /NFL /NDL /NJH /NJS >nul
robocopy "%ROOT%docs" "%TARGET%\docs" /E /NFL /NDL /NJH /NJS >nul
robocopy "%ROOT%scripts" "%TARGET%\scripts" /E /NFL /NDL /NJH /NJS >nul

if not exist "%ROOT%results" mkdir "%ROOT%results"
robocopy "%ROOT%results" "%TARGET%\results" /E /NFL /NDL /NJH /NJS /XD "perf" >nul
robocopy "%ROOT%results\perf" "%TARGET%\results\perf" /E /NFL /NDL /NJH /NJS /XF "*.lock" >nul

powershell -NoProfile -Command "Compress-Archive -Path '%TARGET%' -DestinationPath '%OUT%' -Force"
if errorlevel 1 goto :error

rmdir /s /q "%TARGET%"

echo.
echo Paquete generado: %OUT%
dir "%OUT%"

endlocal
exit /b 0

:error
echo Error al crear el paquete.
endlocal
exit /b 1
