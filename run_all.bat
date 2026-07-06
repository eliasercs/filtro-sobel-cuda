@echo off
setlocal

set "INSTANCES=small medium large no-divisible"
set "KERNELS=5 9"
set "SCALES=0.5 1.75"

set "ROOT=%~dp0"
set "BUILD=%ROOT%build"
set "SRC=%ROOT%src"
set "RES=%ROOT%results"

if exist "%RES%\resultados.csv" del /q "%RES%\resultados.csv"

cd /d "%SRC%"

echo === CPU secuencial ===
for %%I in (%INSTANCES%) do (
  for %%K in (%KERNELS%) do (
    for %%S in (%SCALES%) do (
      echo --- CPU ^&^& %%I k=%%K s=%%S ---
      "%BUILD%\secuencial.exe" --instance=%%I --kernel-size=%%K --scale=%%S
    )
  )
)

echo.
echo === CUDA clasico ===
for %%I in (%INSTANCES%) do (
  for %%K in (%KERNELS%) do (
    for %%S in (%SCALES%) do (
      echo --- CUDA ^&^& %%I k=%%K s=%%S ---
      "%BUILD%\cuda.exe" --instance=%%I --kernel-size=%%K --scale=%%S
    )
  )
)

echo.
echo === CUDA clasico (gaussiano separable) ===
for %%I in (%INSTANCES%) do (
  for %%K in (%KERNELS%) do (
    for %%S in (%SCALES%) do (
      echo --- CUDA Separable ^&^& %%I k=%%K s=%%S ---
      "%BUILD%\cuda.exe" --instance=%%I --kernel-size=%%K --scale=%%S --separable
    )
  )
)

echo.
echo === CUDA Tile ===
for %%I in (%INSTANCES%) do (
  for %%K in (%KERNELS%) do (
    for %%S in (%SCALES%) do (
      echo --- Tile ^&^& %%I k=%%K s=%%S ---
      "%BUILD%\tile.exe" --instance=%%I --kernel-size=%%K --scale=%%S
    )
  )
)

echo.
echo === cuTile Python ===
for %%I in (%INSTANCES%) do (
  for %%K in (%KERNELS%) do (
    for %%S in (%SCALES%) do (
      echo --- cuTile ^&^& %%I k=%%K s=%%S ---
      py -3.10 "%SRC%\cutile_pipeline.py" --instance=%%I --kernel-size=%%K --scale=%%S
    )
  )
)

echo.
echo === Backup del CSV final ===
copy /Y "%RES%\resultados.csv" "%RES%\resultados_full.csv" >nul
echo CSV completo en %RES%\resultados_full.csv
echo Filas: 
find /c /v "" < "%RES%\resultados.csv"

endlocal
