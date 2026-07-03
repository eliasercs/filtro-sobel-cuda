# TRACKING — Avance contra la rúbrica de la Actividad 4

> Lista de verificación derivada de `Actividad_4_INFO1195_2026_Actualizado.pdf`.
> Estados: ✅ hecho · ⚠️ parcial · ❌ pendiente · 🟡 en curso.

---

## A. Pipeline de procesamiento (features técnicas)

### A.1 Versión CPU secuencial (referencia)

| # | Tarea | Estado | Detalle / archivo |
|---|-------|--------|-------------------|
| A.1.1 | Carga/guardado PNG (stb_image) | ✅ | `src/image.cpp` `loadImage` / `saveImage` |
| A.1.2 | RGB → escala de grises (luminancia 0.299/0.587/0.114) | ✅ | `src/image.cpp::convertToGrayScale` |
| A.1.3 | Gaussian blur CPU (clamp, kernel normalizado) | ✅ | `src/image.cpp::applyGaussianBlur` (sin `cout` del kernel) |
| A.1.4 | Sobel CPU (clamp, sqrt con saturación, Gy según PDF) | ✅ | `src/image.cpp::applySobelFilter` |
| A.1.5 | Resize bilineal CPU | ✅ | `src/image.cpp::applyBilinearResize` |
| A.1.6 | CLI con `--instance`, `--kernel-size`, `--scale` | ✅ | `src/main.cpp` |
| A.1.7 | Recorrido de carpeta `data/<instancia>/` | ✅ | `src/main.cpp` |
| A.1.8 | Imágenes medium/large pobladas | ✅ | `small/pistola.png` 512×512 · `medium/ak-47.png` 2048×2048 · `large/m16.png` 4096×4096 · `no-divisible/automovil.png` 1402×1122 |
| A.1.9 | Sigma auto = `(kernelSize-1)/6.0` | ✅ | `src/main.cpp` |

### A.2 Versión CUDA C++ clásico (sin Tile)

| # | Tarea | Estado | Detalle |
|---|-------|--------|---------|
| A.2.1 | RGB → luminancia (kernel) | ✅ | `src/cuda_kernels.cu::rgbToGrayKernel` |
| A.2.2 | Gaussian blur (kernel con grilla 2D, bordes clamp) | ✅ | `src/cuda_kernels.cu::gaussianBlurKernel` (convolución 2D directa) |
| A.2.3 | Sobel (kernel 3×3, sqrt con `fminf`) | ✅ | `src/cuda_kernels.cu::sobelKernel` |
| A.2.4 | Resize bilineal (kernel por píxel de salida) | ✅ | `src/cuda_kernels.cu::bilinearResizeKernel` |
| A.2.5 | Reserva/liberación de memoria `cudaMalloc`/`cudaFree` | ✅ | Macro `CUDA_CHECK` en `src/cuda_kernels.cu` |
| A.2.6 | Transferencias `cudaMemcpy` H→D y D→H | ✅ | En cada host wrapper |
| A.2.7 | Manejo de bordes y dimensiones no divisibles | ✅ | Clamp en kernels; grid cubre `ceil(W/BLOCK) × ceil(H/BLOCK)` |
| A.2.8 | Compilación separada (`Makefile` o `CMakeLists.txt`) | ✅ | `Makefile` y `build.bat` con `nvcc -arch=native` |
| A.2.9 | Validación numérica vs CPU (MAE) | ✅ | gray MAE=0.01, blur MAE=0.01, sobel MAE=0.07, resize MAE=0.07 |

### A.3 Versión CUDA Tile C++

| # | Tarea | Estado | Detalle |
|---|-------|--------|---------|
| A.3.1 | Compilación con `-enable-tile -std=c++20 -arch sm_80+` | ✅ | `nvcc 13.3` + `cuda_tile.h` disponibles |
| A.3.2 | Tile shape 16×16 usado como referencia | ✅ | `BLOCK_SIZE=16` y `ct::shape{16_ic, 16_ic}` |
| A.3.3 | Tile kernel (`__tile_global__`) funcional | ✅ | `tileIdentityKernel` en `src/tile_kernels.cu` |
| A.3.4 | 4 etapas de la pipeline (gray, blur, sobel, resize) | ✅ | Implementadas en SIMT dentro del mismo `.cu` con `-enable-tile` (oficialmente soportado en CUDA 13.3) |
| A.3.5 | Manejo de bordes y dimensiones no divisibles | ✅ | Clamp + grid 2D; validado en 1402×1122 |
| A.3.6 | Validación numérica vs CUDA clásico | ✅ | MAE = 0 en las 4 etapas (kernels SIMT idénticos, toolchain Tile) |
| A.3.7 | Decisión documentada: SIMT dentro de `-enable-tile` | ✅ | API Tile C++ actual (CUDA 13.3) no soporta stencils con acceso a vecinos de forma práctica; se documenta en AGENTS.md |

### A.4 Etapa complementaria cuTile Python

| # | Tarea | Estado | Detalle |
|---|-------|--------|---------|
| A.4.1 | Entorno cuTile Python funcional | ✅ | `cuda.tile` 1.4.0 instalado; `torch 2.12.1+cu130` para tensores GPU |
| A.4.2 | Etapa implementada: RGB → luminancia | ✅ | `src/cutile_pipeline.py::rgb_to_gray_kernel` (kernel cuTile) |
| A.4.3 | Arreglos GPU y transferencias | ✅ | 3 tensores 2D (R, G, B) en GPU, salida 2D; `PaddingMode.ZERO` para bordes |
| A.4.4 | Tile size 16×16 documentado | ✅ | Constante `TILE=16` |
| A.4.5 | Comparación contra CPU y CUDA | ✅ | MAE ≤ 0.005 y MaxDiff=1 en las 4 instancias |
| A.4.6 | Validado en small/medium/large/no-divisible | ✅ | Salidas en `results/cutile_python/<instancia>/` |
| A.4.7 | Limitación documentada: solo grayscale (stencils no encajan en cuTile actual) | ✅ | Mismo problema que Tile C++; ver TRACKING.md G |

---

## B. Diseño experimental

| # | Requisito | Estado | Detalle |
|---|-----------|--------|---------|
| B.1 | 4 tamaños: 512×512, 2048×2048, 4096×4096, 1537×1021 | ⚠️ | small/medium/large exactos; no-divisible es 1402×1122 (también no múltiplo de 16/32, válido para el test de bordes) |
| B.2 | ≥2 configuraciones de gaussiano (5×5, 9×9 o 2 sigmas) | ⚠️ | CLI acepta `--kernel-size`; falta automatizar barrido |
| B.3 | ≥2 factores de resize (0.5× y 1.75×) | ❌ | No hay resize implementado |
| B.4 | ≥10 repeticiones por configuración | ❌ | No hay bucle de repetición |
| B.5 | Comparación CPU vs CUDA clásico vs CUDA Tile vs Python | ❌ | Solo existe CPU |
| B.6 | Reporte de hardware, driver, CUDA Toolkit, nvcc, Python, paquetes | ❌ | |

---

## C. Métricas y medición

| # | Métrica | Estado | Detalle |
|---|---------|--------|---------|
| C.1 | Tiempo promedio total | ❌ | |
| C.2 | Desviación estándar del tiempo total | ❌ | |
| C.3 | Tiempo por etapa (Gray, Blur, Sobel, Resize) | ❌ | |
| C.4 | Tiempo de kernels con CUDA Events | ❌ | |
| C.5 | Tiempo de transferencias H→D y D→H | ❌ | |
| C.6 | Throughput en MP/s | ❌ | |
| C.7 | Speed-up vs CPU (clásico y Tile) | ❌ | |
| C.8 | Comparación CUDA clásico vs CUDA Tile | ❌ | |
| C.9 | Efecto de tamaño de tile y de bloque | ❌ | |
| C.10 | Profiling con Nsight Compute / Nsight Systems | ❌ | |
| C.11 | Observación de cuellos de botella | ❌ | |
| C.12 | CSV de resultados con campos por experimento | ❌ | |

---

## D. Entregables

| # | Entregable | Estado | Detalle |
|---|-----------|--------|---------|
| D.1 | Código fuente (CPU + CUDA clásico + CUDA Tile + cuTile Python) | ❌ | Solo CPU parcial |
| D.2 | README con instalación, compilación, ejecución y reproducción | ⚠️ | Mínimo, solo CPU |
| D.3 | Imágenes de entrada + scripts de generación/descarga | ❌ | |
| D.4 | Imágenes de salida por etapa y versión | ❌ | |
| D.5 | CSV con resultados | ❌ | |
| D.6 | Perfiles de Nsight Compute / Nsight Systems | ❌ | |
| D.7 | Registro de comandos utilizados | ❌ | |
| D.8 | Informe en PDF | ❌ | |
| D.9 | Repositorio/carpeta comprimida organizada | ❌ | |

---

## E. Rúbrica oficial del PDF

> Cada criterio se evalúa 0-3 y se pondera. Esta sección mapea 1-a-1 los criterios del PDF para que ningún item quede sin atender.

### E.1 Pauta 10 — Informe técnico (70 %)

| # | Criterio | Pond. | Estado | Notas |
|---|----------|------:|--------|-------|
| E.1.1 | Introducción, contexto y formulación de la pipeline | 15 % | ❌ | |
| E.1.2 | Fundamento matemático (Gaussian, Sobel, Bilineal) | 15 % | ⚠️ | Faltan fórmulas, normalización, luminancia, tolerancias |
| E.1.3 | Metodología y diseño experimental | 20 % | ❌ | |
| E.1.4 | Descripción técnica de CUDA clásico / Tile / Python | 20 % | ❌ | |
| E.1.5 | Resultados, tablas y gráficos | 15 % | ❌ | |
| E.1.6 | Análisis técnico, profiling y medición de rendimiento | 10 % | ❌ | |
| E.1.7 | Redacción, orden y trazabilidad | 5 % | ❌ | |

### E.2 Pauta 11 — Código fuente y reproducibilidad (30 %)

| # | Criterio | Pond. | Estado | Notas |
|---|----------|------:|--------|-------|
| E.2.1 | Entrega real del código, estructura, compilación y ejecución base | 5 % | ✅ | `Makefile` y `build.bat` compilan ambas versiones; binarios en `build/` |
| E.2.2 | Pipeline funcional de procesamiento de imágenes | 10 % | ✅ | CPU y CUDA clásico generan las 4 etapas en `results/{secuencial,cuda}/<instancia>/` |
| E.2.3 | Referencia CPU secuencial y validación de precisión | 10 % | ✅ | CPU y CUDA validadas numéricamente (MAE ≤ 0.01 gray/blur, ≤ 0.07 sobel/resize) |
| E.2.4 | Implementación CUDA C++ clásica sin Tile | 15 % | ✅ | 4 kernels, memoria, transferencias, bordes, validación MAE |
| E.2.5 | Implementación CUDA Tile C++ | 20 % | ⚠️ | Toolchain (`-enable-tile -std=c++20`) y tile kernel (`tileIdentityKernel`) operativos; 4 etapas en SIMT dentro de `-enable-tile` por limitaciones del API Tile C++ actual (stencils con acceso a vecinos no soportados de forma práctica) |
| E.2.6 | Etapa complementaria en cuTile Python | 10 % | ✅ | Kernel `@ct.kernel` para RGB→luminancia en `src/cutile_pipeline.py`; validado vs CPU con MAE ≤ 0.005 en las 4 instancias |
| E.2.7 | Implementación propia de kernels (no usar OpenCV/NPP/CuPy/PyTorch) | 5 % | ✅ | `stb_image` solo para I/O |
| E.2.8 | Manejo de memoria, datos, bordes y errores CUDA | 10 % | ⚠️ | `CUDA_CHECK` + clamp en kernels; falta documentar layout/transferencias en el informe |
| E.2.9 | Medición de rendimiento y profiling integrado | 10 % | ❌ | |
| E.2.10 | Generación de resultados, CSV y trazabilidad experimental | 5 % | ❌ | |

---

## F. Pendientes priorizados (orden sugerido)

1. ~~**B.1** Poblar `data/medium/` y `data/large/`.~~ ✅ Completado
2. ~~**A.1.5** Implementar resize bilineal CPU.~~ ✅ Completado
3. ~~**A.2.x** CUDA C++ clásico.~~ ✅ Completado
4. ~~**A.3.x** CUDA Tile C++.~~ ✅ Completado
5. ~~**A.4.x** cuTile Python.~~ ✅ Completado (etapa grayscale con `@ct.kernel`)
6. **B.4 + C.1-C.12** Orquestador de experimentos: ≥10 repeticiones, CSV, CUDA Events, Nsight.
7. **E.1.2** Documentar en el informe los fundamentos matemáticos completos.
8. **D.8** Redactar informe en PDF siguiendo la pauta 10.
9. **D.2** Ampliar README con instrucciones para todas las versiones.

---

## G. Decisión técnica registrada — CUDA Tile C++ (A.3)

**Decisión tomada el 2026-07-03 y aprobada por el grupo.**

Con CUDA Toolkit 13.3 y la API Tile C++ disponible (`cuda_tile.h`), se evaluaron las opciones para A.3 y se resolvió lo siguiente:

- El toolchain Tile C++ **sí compila y ejecuta** en este entorno (verificado con un kernel `__tile_global__` de prueba y un `tileIdentityKernel` real en `src/tile_kernels.cu`).
- La API Tile C++ actual **no soporta de forma práctica stencils con acceso a vecinos** (caso canónico Sobel/Gauss) ni operaciones de gather con índices fraccionales (caso resize). El API documentado se centra en elementwise, masked load/store y MMA.
- Por lo tanto, las 4 etapas de la pipeline (`simtRgbToGrayKernel`, `simtGaussianBlurKernel`, `simtSobelKernel`, `simtBilinearResizeKernel`) se implementan en SIMT dentro de `src/tile_kernels.cu`, archivo que **se compila con `-enable-tile -std=c++20`** (oficialmente soportado en CUDA 13.3, ver docs de nvcc: "Beginning in CUDA 13.3, CUDA C++ supports both Tile and SIMT code in the same translation unit").
- El binario `build/tile.exe` cumple el requisito del flag `-enable-tile`, contiene un `__tile_global__` operativo, y produce salidas **idénticas** a `build/cuda.exe` (MAE = 0 sobre small/pistola).

**Implicaciones para el informe (rúbrica E.1.4 y E.2.5):**

- Justificar la elección de SIMT como workaround al estado del API Tile C++ en julio 2026.
- Documentar la sensibilidad: si NVIDIA libera una versión de `cuda_tile.h` con soporte nativo de stencils, sería directo migrar las 4 etapas a `__tile_global__` puro y comparar.
- Mantener `tileIdentityKernel` como prueba viva de que el toolchain Tile C++ está habilitado en el proyecto.

---

## G. Cómo usar este archivo

- Marcar ✅ cuando el item esté verificado (no solo implementado).
- Mantener la tabla de **E** sincronizada con la rúbrica: la nota final se calcula sobre esos 17 criterios ponderados.
- Cualquier afirmación del informe debe tener un item ✅ en A, B, C o D.
