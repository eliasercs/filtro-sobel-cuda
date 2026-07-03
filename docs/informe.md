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

| # | Versión            | Toolchain                            | Estado |
|---|--------------------|--------------------------------------|--------|
| 1 | CPU secuencial     | `nvcc -std=c++17`                    | ✅ referencia |
| 2 | CUDA C++ clásico   | `nvcc -std=c++17 -arch=native`       | ✅ kernels SIMT + Occupancy API |
| 3 | CUDA Tile C++      | `nvcc -std=c++20 -enable-tile`       | ✅ `__tile_global__` de referencia + kernels SIMT (ver §5) |
| 4 | cuTile Python      | `cuda.tile 1.4.0` + `torch 2.12.1+cu130` | ✅ `@ct.kernel` para grayscale + etapas restantes en PyTorch |

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
- Resto de etapas (blur, sobel, resize) en PyTorch sobre GPU
  (`torch.nn.functional.conv2d`, `interpolate`).
- `torch.cuda.Event` para medir cada kernel de la pipeline.
- 10 repeticiones con warmup; mismo formato CSV.

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

(Para llenar con los datos reales de `results/resultados.csv` después
de ejecutar `run_all.bat`)

### 5.3 Tabla comparativa CPU vs GPU

(Para llenar con datos del CSV)

### 5.4 Profiling con Nsight

(Para llenar ejecutando `profile.bat medium 5 0.5` y adjuntando las
observaciones de los reportes `nsys_*` y `ncu_*`)

## 6. Análisis técnico, profiling y medición de rendimiento

### 6.1 Speed-up CPU vs GPU

(Pendiente de poblar con datos del CSV)

### 6.2 Efecto del tamaño de imagen

(Pendiente)

### 6.3 Efecto del kernel gaussiano (5×5 vs 9×9)

(Pendiente)

### 6.4 Efecto del factor de resize (0.5× vs 1.75×)

(Pendiente)

### 6.5 Observaciones de Nsight

(Pendiente ejecutar `profile.bat`)

## 7. Conclusiones

(Pendiente)

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

## 9. Anexos

- **Código fuente:** `src/`
- **Binarios:** `build/`
- **Resultados visuales:** `results/{secuencial,cuda,tile,cutile_python}/<instancia>/`
- **CSV unificado:** `results/resultados.csv`
- **Perfiles Nsight:** `results/perf/`
