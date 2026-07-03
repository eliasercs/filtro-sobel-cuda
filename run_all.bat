@echo off
setlocal

set "INSTANCES=small medium large no-divisible"
set "KERNELS=5 9"
set "SCALES=0.5 1.75"

if exist ..\results\resultados.csv del /q ..\results\resultados.csv

echo === CPU secuencial ===
for %%I in (%INSTANCES%) do (
  for %%K in (%KERNELS%) do (
    for %%S in (%SCALES%) do (
      echo --- CPU ^&^& %%I k=%%K s=%%S ---
      ..\build\secuencial.exe --instance=%%I --kernel-size=%%K --scale=%%S
    )
  )
)

echo.
echo === CUDA clasico ===
for %%I in (%INSTANCES%) do (
  for %%K in (%KERNELS%) do (
    for %%S in (%SCALES%) do (
      echo --- CUDA ^&^& %%I k=%%K s=%%S ---
      ..\build\cuda.exe --instance=%%I --kernel-size=%%K --scale=%%S
    )
  )
)

echo.
echo === CUDA Tile ===
for %%I in (%INSTANCES%) do (
  for %%K in (%KERNELS%) do (
    for %%S in (%SCALES%) do (
      echo --- Tile ^&^& %%I k=%%K s=%%S ---
      ..\build\tile.exe --instance=%%I --kernel-size=%%K --scale=%%S
    )
  )
)

echo.
echo === cuTile Python ===
for %%I in (%INSTANCES%) do (
  for %%K in (%KERNELS%) do (
    for %%S in (%SCALES%) do (
      echo --- cuTile ^&^& %%I k=%%K s=%%S ---
      py -3.10 ..\src\cutile_pipeline.py --instance=%%I --kernel-size=%%K --scale=%%S
    )
  )
)

echo.
echo === Backup del CSV final ===
copy /Y ..\results\resultados.csv ..\results\resultados_full.csv >nul
echo CSV completo en ..\results\resultados_full.csv

endlocal
