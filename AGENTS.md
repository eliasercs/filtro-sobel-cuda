# AGENTS — Filtro Sobel CUDA

> Contexto del proyecto para subagentes de opencode (arquitecto, backend, designer, explore, general).
> Última actualización: 2026-07-03.

## 1. Propósito del proyecto

Este repositorio resuelve la **Actividad 4 de INFO1195 (2026)** de la carrera: implementar, analizar y comparar una **pipeline de procesamiento de imágenes RGB** sobre GPU, contrastando cuatro versiones de la misma solución.

### Pipeline

```
Imagen RGB (PNG) ──► Escala de grises ──► Gaussian Blur ──► Sobel (bordes) ──► (Resize bilineal) ──► Imagen(es) de salida
```

Tres etapas obligatorias:

1. **Filtro gaussiano** — convolución 2D con kernel 5×5/9×9 (o dos sigmas), núcleo separable o directo.
2. **Operador de Sobel** — gradientes Gx, Gy y magnitud `sqrt(Gx²+Gy²)` sobre la imagen en luminancia.
3. **Redimensionamiento bilineal** — interpolación bilineal con factores 0.5× y 1.75× (mínimo).

### Cuatro versiones obligatorias

| # | Versión | Estado actual |
|---|---------|---------------|
| 1 | CPU secuencial (referencia) | Implementada en `src/image.cpp` |
| 2 | CUDA C++ clásico (sin Tile) | **No implementada** |
| 3 | CUDA Tile C++ | **No implementada** |
| 4 | cuTile Python (al menos una etapa) | **No implementada** |

La comparación debe reportar correctitud, tiempo total, tiempo por kernel, transferencias, speed-up, throughput (MP/s) y profiling con herramientas del CUDA Toolkit (CUDA Events, Nsight Compute, Nsight Systems).

## 2. Restricciones duras del PDF

- **No** se acepta llamar directamente a filtros ya implementados en GPU por OpenCV, NPP, CuPy o PyTorch. Sí se permiten para I/O, manejo de arreglos y visualización.
- Las **mediciones de rendimiento** deben respaldarse con CUDA Events / Nsight Compute / Nsight Systems; no se acepta un cronómetro externo como única fuente.
- **≥10 repeticiones** por configuración experimental.
- **4 tamaños** de imagen: 512×512, 2048×2048, 4096×4096 y un no-divisible (≈1537×1021).
- **≥2 configuraciones** para el filtro gaussiano.
- **≥2 factores** de resize.
- La entrega del código es obligatoria; si no compila o no ejecuta, tanto el código como el informe reciben nota mínima.
- Todo lo que se afirme en el informe debe ser verificable en el código y viceversa.

## 3. Stack y dependencias

- **Lenguaje principal:** C++17 (compilado con `nvcc`).
- **Imágenes:** `stb_image.h` + `stb_image_write.h` (incluidos en `src/`, sin dependencias externas).
- **GPU/CUDA:** toolkit con `nvcc`, CUDA C++ clásico, CUDA Tile C++ y `cuTile` para Python.
- **Build (actual):** `nvcc -std=c++17 image.cpp main.cpp -o secuencial` (solo CPU).
- **Salida:** PNG por etapa, CSV de métricas, perfiles de Nsight.

## 4. Estructura del repositorio

```
filtro-sobel-cuda/
├── AGENTS.md                       # este archivo (contexto para subagentes)
├── TRACKING.md                     # checklist de avance contra la rúbrica
├── README.md                       # instrucciones mínimas de compilación/ejecución
├── Makefile                        # build para secuencial + cuda (nvcc -arch=native)
├── build.bat                       # equivalente Windows (.bat)
├── Actividad_4_INFO1195_2026_Actualizado.pdf   # enunciado oficial
├── data/                           # imágenes de entrada
│   ├── small/                      # ~512×512
│   ├── medium/                     # ~2048×2048
│   ├── large/                      # ~4096×4096
│   └── no-divisible/               # ~1402×1122 (no múltiplo de 16/32)
├── src/                            # código fuente C++/CUDA
│   ├── image.hpp                   # API pública CPU
│   ├── image.cpp                   # versión CPU (referencia)
│   ├── main.cpp                    # CLI versión CPU
│   ├── cuda_kernels.hpp            # API pública CUDA
│   ├── cuda_kernels.cu             # kernels + host wrappers
│   ├── main_cuda.cpp               # CLI versión CUDA
│   ├── tile_kernels.hpp            # API pública CUDA Tile C++
│   ├── tile_kernels.cu             # `__tile_global__` + SIMT, `-enable-tile`
│   ├── main_tile.cpp               # CLI versión CUDA Tile C++
│   ├── stb_image.h
│   └── stb_image_write.h
├── build/                          # binarios compilados (gitignored)
└── results/                        # imágenes de salida y CSV (gitignored)
    ├── secuencial/<instancia>/     # salidas CPU
    ├── cuda/<instancia>/           # salidas CUDA
    └── tile/<instancia>/           # salidas CUDA Tile C++
```

## 5. Convenciones del código

- **Layout de imagen en CPU:** buffer `unsigned char*` plano, escala de grises de 1 canal; RGB se entrega en `channels=3` o `4` por `stb_image`. Índice principal: `y * width + x` para grises, `(y * width + x) * channels + c` para RGB.
- **Estrategia de bordes (CPU/GPU):** **clamp** (réplica del píxel del borde). No se usa cero ni wrap.
- **Sigma del gaussiano:** se calcula automáticamente como `sigma = (kernelSize - 1) / 6.0f` para que el radio cubra ~3 desviaciones estándar. Documentar y permitir override por argumento.
- **Magnitud Sobel:** `int magnitude = (int)sqrt(sumX*sumX + sumY*sumY);` con `saturate_cast` a 255. En GPU usar `fminf(mag, 255.0f)` o `__saturatef`.
- **Errores CUDA:** macro `CUDA_CHECK(err)` o equivalente que aborte en fallo; chequear tras cada `cudaMalloc`, `cudaMemcpy`, `cudaLaunch`.
- **Medición:** `cudaEvent_t start, stop; cudaEventRecord(...); cudaEventSynchronize(stop);` envolviendo cada kernel y cada `cudaMemcpy` por separado.
- **Nombres de archivo de salida:** incluir versión (`cpu`, `cuda`, `cuda_tile`, `cutile_python`), etapa (`gray`, `blur`, `sobel`, `resize`), tamaño y configuración (`k5`, `k9`, `s0.5x`, `s1.75x`).

## 6. Estado del repositorio (resumen)

| Componente | Estado | Archivo |
|------------|--------|---------|
| Carga/guardado PNG | ✅ | `src/image.cpp` |
| RGB → luminancia (0.299/0.587/0.114) | ✅ | `src/image.cpp::convertToGrayScale` / `src/cuda_kernels.cu::rgbToGrayKernel` |
| Gaussian blur CPU (clamp, normalizado) | ✅ | `src/image.cpp::applyGaussianBlur` (convolución 2D directa) |
| Sobel CPU (clamp, sqrt con saturación) | ✅ | `src/image.cpp::applySobelFilter` |
| Resize bilineal CPU | ✅ | `src/image.cpp::applyBilinearResize` |
| Gaussian blur CUDA clásico | ✅ | `src/cuda_kernels.cu::gaussianBlurKernel` |
| Sobel CUDA clásico | ✅ | `src/cuda_kernels.cu::sobelKernel` |
| Resize bilineal CUDA clásico | ✅ | `src/cuda_kernels.cu::bilinearResizeKernel` |
| Toolchain CUDA Tile C++ (`-enable-tile -std=c++20`) | ✅ | `src/tile_kernels.cu` con un `__tile_global__` (`tileIdentityKernel`) |
| Gaussian blur CUDA Tile C++ | ⚠️ | `simtGaussianBlurKernel` (mismo algoritmo que clásico, en `.cu` con `-enable-tile`) |
| Sobel CUDA Tile C++ | ⚠️ | `simtSobelKernel` (mismo algoritmo que clásico) |
| Resize bilineal CUDA Tile C++ | ⚠️ | `simtBilinearResizeKernel` (mismo algoritmo que clásico) |
| Cualquier etapa en cuTile Python | ❌ | — |
| CSV de resultados | ❌ | — |
| Profiling Nsight | ❌ | — |
| ≥10 repeticiones por config | ❌ | — |
| Imágenes medium/large pobladas | ✅ | `data/medium/ak-47.png`, `data/large/m16.png` |
| Informe en PDF | ❌ | — |

Detalle fino contra la rúbrica: ver `TRACKING.md`.

## 7. Decisiones técnicas a resolver (abiertas)

1. **Gaussiano separable vs. directo:** el PDF recomienda separable. Implementar ambas y comparar.
2. **Tile size para CUDA Tile:** empezar con 32×32 y 16×16; documentar sensibilidad. (Estado actual: API Tile C++ en CUDA 13.3 no soporta stencils con acceso a vecinos de forma práctica; el binario se compila con `-enable-tile -std=c++20` y un kernel Tile de referencia `tileIdentityKernel`; las 4 etapas de la pipeline corren como SIMT dentro del mismo `.cu`, lo cual está oficialmente soportado.)
3. **Bloque clásico:** `BLOCK_SIZE = 16` o `32` con grilla 2D cubriendo `(width+BLOCK-1)/BLOCK` en X e Y.
4. **Librerías de medición:** además de CUDA Events, correr al menos una métrica con `nvprof`/`Nsight Compute` (`ncu --set full`).
5. **cuTile Python:** elegir una etapa sencilla (Sugerencia: Sobel o conversión a luminancia) para minimizar tiempo de implementación.
6. **Layout en GPU:** mantener `interleaved RGB` (entrada) y convertir a planar `(H,W,3)` solo dentro de kernels si conviene; documentar la decisión.
7. **Error numérico aceptado:** tolerancia por etapa, p. ej. MAE ≤ 1.0 para grises/uint8 y ≤ 2.0 para Sobel.

## 8. Comandos de referencia

```bash
# Compilar (Make o build.bat, ambos apuntan a nvcc con -arch=native)
make
# o
./build.bat

# Ejecutar CPU
./build/secuencial.exe --instance=small --kernel-size=5 --scale=0.5
./build/secuencial.exe --instance=no-divisible --kernel-size=9 --scale=1.75

# Ejecutar CUDA clásico
./build/cuda.exe --instance=small --kernel-size=5 --scale=0.5
./build/cuda.exe --instance=large --kernel-size=9 --scale=1.75

# Ejecutar CUDA Tile C++ (compilado con -enable-tile -std=c++20)
./build/tile.exe --instance=small --kernel-size=5 --scale=0.5
./build/tile.exe --instance=large --kernel-size=9 --scale=1.75

# Perfilado (cuando se habilite la fase de medición)
nsys profile -o results/perf/report ./build/cuda.exe --instance=small --kernel-size=5
ncu --set full --target-processes all -o results/perf/kernels ./build/cuda.exe --instance=small --kernel-size=5
```

## 9. Recursos y referencias

- Enunciado oficial: `Actividad_4_INFO1195_2026_Actualizado.pdf` (en la raíz del repo).
- `stb_image.h` / `stb_image_write.h`: https://github.com/nothings/stb
- Documentación CUDA y cuTile: usar el MCP **context7** antes de generar kernels Tile o Python.
- Para patrones de implementación GPU: MCP **gh_grep** buscando `cudaSobel`, `cudaGaussian`, `cutile`.
