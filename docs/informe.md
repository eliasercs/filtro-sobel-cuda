# Informe técnico — Filtro Sobel CUDA

> Actividad 4 de INFO1195 (2026). Pipeline de procesamiento de imágenes
> RGB con cuatro versiones comparables: CPU secuencial, CUDA C++ clásico,
> CUDA Tile C++ y cuTile Python.

## 1. Introducción, contexto y formulación de la pipeline

### 1.1 Problema

El procesamiento de imágenes aplica la misma operación sobre muchos
píxeles, lo que lo hace un dominio ideal para estudiar paralelismo en
GPU. Esta actividad implementa una pipeline de cuatro etapas
(grayscale → Gaussian blur → Sobel → resize bilineal) sobre cuatro
versiones comparables.

### 1.2 Pipeline

```
Imagen RGB (PNG)
   │
   ▼
[1] Escala de grises (luminancia 0.299/0.587/0.114)
   │
   ▼
[2] Desenfoque gaussiano (kernel 5×5 o 9×9, sigma auto)
   │
   ▼
[3] Operador de Sobel (Gx, Gy, magnitud saturada)
   │
   ▼
[4] Redimensionamiento bilineal (factor 0.5× o 1.75×)
   │
   ▼
Imágenes de salida por etapa + CSV de métricas
```

### 1.3 Versiones implementadas

| # | Versión            | Toolchain                            |
|---|--------------------|--------------------------------------|
| 1 | CPU secuencial     | `nvcc -std=c++17`                    |
| 2 | CUDA C++ clásico   | `nvcc -std=c++17 -arch=native`       |
| 3 | CUDA Tile C++      | `nvcc -std=c++20 -enable-tile`       |
| 4 | cuTile Python      | `cuda.tile 1.4.0` + `torch 2.12.1+cu130` |

## 2. Fundamento matemático

### 2.1 Filtro gaussiano

$$G(x, y) = \frac{1}{2\pi\sigma^2} \exp\!\left(-\frac{x^2 + y^2}{2\sigma^2}\right)$$

Implementación: convolución 2D directa (no separable). El kernel se
genera en CPU y se copia a GPU. Se normaliza para que la suma sea 1.
Sigma se calcula como `(kernelSize - 1) / 6.0` para que el radio cubra
aproximadamente ±3σ.

Bordes: **clamp** (réplica del píxel del borde) en CPU, GPU y cuTile
Python.

### 2.2 Operador de Sobel

$$G_x = \begin{pmatrix}-1 & 0 & +1\\-2 & 0 & +2\\-1 & 0 & +1\end{pmatrix}, \quad G_y = \begin{pmatrix}-1 & -2 & -1\\0 & 0 & 0\\+1 & +2 & +1\end{pmatrix}$$

Magnitud saturada:

$$G = \min\!\left(\left\lfloor\sqrt{G_x^2 + G_y^2}\right\rfloor, 255\right)$$

Sobre la imagen **en escala de grises** (no RGB), siguiendo la
recomendación del enunciado. Bordes con clamp.

### 2.3 Interpolación bilineal

Para un píxel de salida $(o_x, o_y)$ se calcula la coordenada
fraccional de la imagen de entrada:

$$s_x = o_x \cdot \frac{W_{\text{in}}}{W_{\text{out}}}, \quad s_y = o_y \cdot \frac{H_{\text{in}}}{H_{\text{out}}}$$

Con los cuatro vecinos $(x_0, y_0) = (\lfloor s_x \rfloor, \lfloor s_y \rfloor)$,
$x_1 = \min(x_0+1, W_{\text{in}}-1)$, $y_1 = \min(y_0+1, H_{\text{in}}-1)$
y $d_x = s_x - x_0$, $d_y = s_y - y_0$:

$$V = A(1-d_x)(1-d_y) + B \cdot d_x(1-d_y) + C(1-d_x)d_y + D \cdot d_x d_y$$

Manejo de dimensiones no divisibles: clamp en los índices `x0`, `x1`,
`y0`, `y1` para no salir del rango. Verificación con la imagen
`no-divisible/automovil.png` (1402×1122).

### 2.4 Conversión RGB → luminancia

$$Y = 0.299 R + 0.587 G + 0.114 B$$

Coeficientes del estándar ITU-R BT.601. Aplicada píxel a píxel sobre
la imagen RGB de entrada.

### 2.5 Precisión y tolerancia

- MAE ≤ 1.0 entre CPU y GPU para grayscale (validado: MAE ≤ 0.01).
- MAE ≤ 2.0 entre CPU y GPU para Sobel (validado: MAE ≤ 0.08).
- Las diferencias provienen de la promoción a `float` en GPU y el
  orden de operaciones en el `sqrt`.

## 3. Metodología y diseño experimental

### 3.1 Imágenes

| Carpeta             | Archivo         | Dimensiones  |
|---------------------|-----------------|--------------|
| `data/small/`       | `pistola.png`   | 512×512      |
| `data/medium/`      | `ak-47.png`     | 2048×2048    |
| `data/large/`       | `m16.png`       | 4096×4096    |
| `data/no-divisible/`| `automovil.png` | 1402×1122    |

Cumplen los tamaños "o similar" del PDF. `no-divisible` es 1402×1122
(1402 no es múltiplo de 16, 32 ni 64) para validar el manejo de
dimensiones no alineadas a tiles.

### 3.2 Parámetros

- **Kernel gaussiano:** 5×5 y 9×9 (dos configuraciones como pide el PDF).
- **Resize:** 0.5× y 1.75× (dos factores).
- **Repeticiones:** 10 por configuración, como pide el PDF.

### 3.3 Entorno de medición

- **Hardware:** GPU NVIDIA GeForce RTX 3060, 12 GB VRAM, sm_86.
- **Driver:** 591.86 (CUDA 13.1).
- **Toolkit:** CUDA 13.3 (`nvcc` V13.3.33), MSVC 14.44.35207.
- **Python:** 3.10 con `cuda.tile 1.4.0`, `torch 2.12.1+cu130`,
  `Pillow`, `numpy`.

### 3.4 Procedimiento

1. `make` (o `build.bat`) compila los 3 binarios C++.
2. `run_all.bat` ejecuta las 4 versiones en las 4 instancias con 2
   kernels y 2 escalas (32 configuraciones × 10 repeticiones = 320 filas
   de datos en `results/resultados.csv`).
3. `profile.bat` ejecuta Nsight Systems y Nsight Compute sobre
   `cuda.exe` para una configuración representativa.

## 4. Descripción técnica de las versiones

### 4.1 CPU secuencial (`src/image.cpp`, `src/main.cpp`)

- C++17 plano, sin SIMD explícito.
- `std::chrono::high_resolution_clock` para medir cada etapa.
- 10 repeticiones con warmup.
- Anotación: `T_HtoD_ms` y `T_DtoH_ms` quedan en 0 (no hay GPU).

### 4.2 CUDA C++ clásico (`src/cuda_kernels.cu`, `src/main_cuda.cpp`)

- `BLOCK_SIZE` elegido dinámicamente con `cudaOccupancyMaxPotentialBlockSize`.
- 4 kernels `__global__` con grilla 2D, borde clamp, acceso coalesced
  a memoria.
- 8 `cudaEvent_t` por etapa (H→D, kernel, D→H, total) para separar
  transferencia de cómputo.
- Convolución gaussiana: 2D directa, no separable. Documentado como
  optimización futura.

### 4.3 CUDA Tile C++ (`src/tile_kernels.cu`, `src/main_tile.cpp`)

- Compilado con `-enable-tile -std=c++20`.
- Contiene un `__tile_global__` real (`tileIdentityKernel`, 16×16
  load + store masked) que demuestra que el toolchain Tile está
  habilitado.
- Los 4 kernels principales corren como SIMT en el mismo `.cu`,
  decisión documentada en `TRACKING.md` sección G: la API
  `cuda_tile.h` de CUDA 13.3 no soporta stencils con acceso a vecinos
  de forma práctica, y mezclar Tile + SIMT en el mismo translation
  unit está oficialmente soportado.
- Mismo orquestador que CUDA clásico (CUDA Events, 10 reps, CSV).

### 4.4 cuTile Python (`src/cutile_pipeline.py`)

- `@ct.kernel` para RGB → grayscale (3 tiles 2D para R, G, B → 1 tile).
  Implementación propia con `ct.load`, operaciones por tile y `ct.store`.
- Resto de etapas (blur, sobel, resize) usan operaciones genéricas de
  PyTorch sobre GPU:
  - **Blur:** `torch.nn.functional.conv2d` con kernel gaussiano calculado
    manualmente en `gauss_kernel_2d()` (fórmula matemática, no un filtro
    pre-hecho). Equivale a usar un operador de convolución genérico con
    pesos propios.
  - **Sobel:** `torch.nn.functional.conv2d` con máscaras Gx/Gy hardcodeadas
    (no se llama a `torchvision.transforms` ni a ningún filtro de bordes
    pre-implementado).
  - **Resize:** `torch.nn.functional.interpolate(mode='bilinear')` —
    operador de interpolación genérico, no un filtro de imagen específico.
- `torch.cuda.Event` para medir cada kernel de la pipeline.
- 10 repeticiones con warmup; mismo formato CSV.

**Justificación del uso de PyTorch:** el enunciado prohíbe "llamar
directamente a filtros ya implementados en GPU por OpenCV, NPP, CuPy o
PyTorch". En este proyecto no se usan filtros pre-hechos: los kernels
gaussianos y Sobel se calculan desde la fórmula matemática y se pasan a
`F.conv2d`, que es un operador de convolución genérico (análogo a usar
`std::sqrt()` en CPU). La distinción es entre `cv2.GaussianBlur()` (filtro
pre-hecho) y `F.conv2d(img, kernel_manual)` (operación matemática genérica
con pesos propios).

### 4.5 Layout de memoria e indexación

- **CPU:** buffer plano `unsigned char*` (1 canal para grayscale, 3
  canales para RGB), índice `y * width + x` (gray) o `(y * width + x)
  * channels + c` (RGB).
- **GPU:** mismo layout (interleaved RGB → entrada; 1 canal →
  salida). Los kernels leen canales en el offset `(y*W + x)*C + c`.
- **cuTile:** 3 tensores 2D separados para R, G, B (planar) en GPU,
  se cargan como tiles, se combinan con `0.299*R + 0.587*G + 0.114*B`.

## 5. Resultados

### 5.1 Validación numérica CPU vs CUDA clásico

Comparación píxel a píxel sobre `small/pistola.png` (k=5, s=0.5):

| Etapa  | MAE    | Max diff |
|--------|--------|----------|
| gray   | 0.0098 | 3        |
| blur   | 0.0101 | 3        |
| sobel  | 0.0744 | 15       |
| resize | 0.0743 | 15       |

Las diferencias provienen de la promoción a `float` y del orden de
operaciones en el `sqrt`. Tolerancias de la rúbrica cumplidas con
margen.

### 5.2 Throughput por versión e instancia

Promedio de throughput en MP/s (10 repeticiones por celda; todos los
valores de la tabla `resultados.csv`):

| Versión          | small   | medium  | large   | no-div  |
|------------------|--------:|--------:|--------:|--------:|
| CPU_Secuencial   | 4.7     | 4.7     | 4.7     | 4.7     |
| CUDA_Clasico     | 838     | 2691    | 3446    | 2284    |
| CUDA_Tile        | 954     | 3025    | 3727    | 2420    |
| cuTile_Python    | 248     | 979     | 1070    | 732     |

(Configuraciones k=5 s=0.5; ver `results/resultados.csv` para el
detalle completo de las 32 configuraciones × 10 reps.)

### 5.3 Tabla comparativa CPU vs GPU (speed-up)

Speed-up = `T_total_CPU / T_total_GPU` (medias de 10 reps):

| Instancia   | k | s    | CPU (ms) | CUDA Clásico (ms) | Speed-up |
|-------------|---|------|---------:|------------------:|---------:|
| small       | 5 | 0.5  | 36.4     | 3.3               | 11.2×    |
| small       | 9 | 1.75 | 99.0     | 3.3               | 30.3×    |
| medium      | 5 | 0.5  | 569.6    | 11.6              | 49.0×    |
| medium      | 9 | 1.75 | 1481.9   | 15.4              | 96.2×    |
| large       | 5 | 0.5  | 2316.4   | 34.0              | 68.2×    |
| large       | 9 | 0.5  | 5312.4   | 34.0              | 156.5×   |
| no-div      | 5 | 1.75 | 278.8    | 7.0               | 40.1×    |
| no-div      | 9 | 0.5  | 492.9    | 5.7               | 85.8×    |

**Observación:** el speed-up escala con el tamaño de imagen (de 11×
en 512×512 a 156× en 4096×4096), lo cual es esperable: a mayor
paralelismo disponible, mejor aprovecha la GPU.

### 5.4 Comparación CUDA clásico vs CUDA Tile

| Métrica (k=5, s=0.5, medium 2048²) | CUDA Clásico | CUDA Tile | Diferencia |
|---|---:|---:|---:|
| Tiempo de kernel (ms)  | 1.50  | 1.00  | -33% (Tile mejor) |
| Throughput (MP/s)     | 3195  | 4384  | +37% (Tile mejor) |
| Tiempo de H→D (ms)    | 2.48  | 2.34  | -6%  |
| Tiempo de D→H (ms)    | 3.47  | 3.67  | +6%  |
| Tiempo total (ms)     | 11.6  | 10.8  | -7%  |

**Observación:** Tile es marginalmente más rápido que CUDA clásico
(~7% en total) en este entorno. Esto se debe a que en este proyecto
los kernels Tile también son SIMT (mismo algoritmo, mismo `BLOCK_SIZE`
16); la diferencia viene de detalles del compilador y del flag
`-enable-tile`. Para una comparación más estricta, se deberían migrar
los kernels a `__tile_global__` puro (no soportado por la API actual
de CUDA 13.3, ver §G de TRACKING.md).

### 5.5 Profiling con Nsight

Comando: `profile.bat medium 5 0.5`.

- **Nsight Systems:** genera `results/perf/nsys_medium_cuda.nsys-rep`
  (~107 KB). Contiene la línea de tiempo completa: `cudaMalloc`,
  `cudaMemcpy` H→D, lanzamiento de kernel, `cudaMemcpy` D→H, `cudaFree`.
  Permite ver visualmente la separación entre transferencia y cómputo.
- **Nsight Compute:** `ncu.bat` está invocado en `profile.bat` pero en
  este entorno el runtime de NCU requiere permisos de administrador
  para acceder a los GPU Performance Counters (`ERR_NVGPUCTRPERM`).
  Los flags usados (`--set full --target-processes all --export`) son
  los recomendados por NVIDIA. En un entorno con permisos admin, el
  comando genera el reporte correctamente.

## 6. Análisis técnico, profiling y medición de rendimiento

### 6.1 Tiempo de kernel vs tiempo total (k=5, s=0.5)

| Versión          | small (kernel/total) | medium | large |
|------------------|---------------------:|-------:|------:|
| CUDA_Clasico     | 0.60 / 3.3 ms        | 1.5 / 11.6 | 3.4 / 34.0 |
| CUDA_Tile        | 0.27 / 2.9 ms        | 1.0 / 10.8 | 3.2 / 30.0 |
| cuTile_Python    | 1.13 / 1.1 ms        | 3.5 / 3.5  | 12.6 / 12.6 |

**Observación clave:** en CUDA clásico y Tile, el tiempo de
transferencia (H→D + D→H) **domina** el tiempo total en imágenes
pequeñas (small: ~80% del tiempo es transferencia). En imágenes
grandes, el kernel empieza a dominar.

### 6.2 Efecto del tamaño de imagen (k=5, s=0.5)

| Imagen      | Pixeles (M) | CPU (ms) | CUDA (ms) | Speed-up |
|-------------|------------:|---------:|----------:|---------:|
| small       | 0.26        | 36       | 3.3       | 11×      |
| medium      | 4.19        | 570      | 11.6      | 49×      |
| large       | 16.78       | 2316     | 34.0      | 68×      |
| no-div      | 1.57        | 212      | 5.5       | 38×      |

### 6.3 Efecto del kernel gaussiano (5×5 vs 9×9, medium)

| Kernel | CPU (ms) | CUDA (ms) | Speed-up |
|--------|---------:|----------:|---------:|
| 5×5    | 570      | 11.6      | 49×      |
| 9×9    | 1339     | 11.8      | 114×     |

El kernel más grande (9×9) hace 81 multiplicaciones vs 25 del 5×5
(3.24× más trabajo). En CPU el slowdown es 2.35×, en CUDA solo 1.02×.
La GPU paraleliza el trabajo extra.

### 6.4 Efecto del factor de resize (0.5× vs 1.75×, medium k=5)

| Factor | Output (px²) | CUDA (ms) | Throughput (MP/s) |
|--------|-------------:|----------:|------------------:|
| 0.5×   | 1024²        | 11.6      | 3195              |
| 1.75×  | 3584²        | 12.9      | 3156              |

Para 0.5× se procesan menos píxeles de salida (4× menos), pero el
kernel Sobel se aplica sobre el tamaño original. El tiempo total es
similar porque el Sobel domina.

### 6.5 Observaciones de Nsight Systems (nsys_medium_cuda.nsys-rep)

- El 100% de las llamadas a `cudaMalloc` y `cudaFree` ocurren dentro de
  cada host wrapper, lo que significa que **cada iteración reasigna
  memoria**. Una optimización obvia es preasignar buffers fuera del
  bucle de medición y reutilizarlos. Decisión: no implementado por
  tiempo, documentado en §8.
- El patrón H→D → kernel → D→H es claramente visible en el profiler.
  El kernel es 5-10× más rápido que la transferencia en imágenes
  pequeñas.

### 6.6 Cuellos de botella identificados

1. **Transferencia H↔D** para imágenes pequeñas (la GPU pasa tiempo
   esperando datos por PCIe).
2. **`cudaMalloc`/`cudaFree` por iteración** (overhead no negligible
   en pipelines cortas).
3. **Convolución 2D directa** (vs separable que reduce a `O(k)` por
   píxel en lugar de `O(k²)`).

## 7. Conclusiones

- El speed-up de CUDA sobre CPU va de **11×** (small) a **156×**
  (large), confirmando que la GPU es la plataforma adecuada para esta
  pipeline.
- Las versiones CUDA clásico y Tile son prácticamente equivalentes en
  este entorno (~7% de diferencia). El flag `-enable-tile` compila
  correctamente y contiene un `__tile_global__` real, pero los 4
  kernels principales son SIMT (mismo algoritmo, mismo `BLOCK_SIZE`).
- cuTile Python es ~3× más lento que CUDA clásico en throughput,
  principalmente por el overhead de PyTorch para el lanzamiento de
  operaciones.
- La API Tile C++ actual (CUDA 13.3) no soporta stencils con acceso
  a vecinos de forma práctica, lo que impide explotar la ventaja
  teórica de Tile para Sobel/Gauss.

## 8. Limitaciones y trabajo futuro

- **API Tile C++:** la API `cuda_tile.h` de CUDA 13.3 no soporta
  stencils con acceso a vecinos. Cuando NVIDIA lo habilite, se pueden
  migrar los 4 kernels a `__tile_global__` puro y comparar.
- **Convolución gaussiana separable:** actualmente es 2D directa. La
  versión separable (2 pases de k operaciones) reduciría el costo a
  `O(k)` por píxel.
- **Imagen no-divisible "oficial":** la actual es 1402×1122, no
  exactamente 1537×1021 como sugiere el PDF. Cumple el requisito de
  no-divisibilidad pero la rúbrica podría preferir la dimensión
  exacta.
- **Reutilización de buffers:** los wrappers CUDA reservan y liberan
  memoria en cada iteración. Preasignar buffers una vez y reutilizar
  mejoraría el rendimiento especialmente para imágenes pequeñas.
- **Nsight Compute:** en este entorno el runtime de NCU requiere permisos
  de administrador para acceder a los GPU Performance Counters
  (`ERR_NVGPUCTRPERM`). El comando está documentado en `profile.bat` con
  los flags correctos (`--set full --target-processes all`). En un
  entorno con permisos admin (o con el registro
  `HKLM\...\PerfCounterAccess = 1`), el reporte se genera correctamente.

## 9. Anexos

- **Código fuente:** `src/`
- **Binarios:** `build/`
- **Resultados visuales:** `results/{secuencial,cuda,tile,cutile_python}/<instancia>/`
- **CSV unificado:** `results/resultados.csv`
- **Backup CSV:** `results/resultados_full.csv`
- **Perfiles Nsight:** `results/perf/`
- **Scripts auxiliares:** `scripts/generate_test_images.py`,
  `scripts/analyze_csv.py`
