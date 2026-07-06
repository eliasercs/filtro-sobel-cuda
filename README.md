# Filtro Sobel CUDA — INFO1195 Actividad 4

Pipeline de procesamiento de imágenes RGB con cuatro versiones comparables:
referencia CPU, CUDA C++ clásico (sin Tile), CUDA Tile C++ y una etapa
complementaria en cuTile Python. La pipeline aplica conversión a luminancia,
desenfoque gaussiano, operador de Sobel y redimensionamiento bilineal.

## 1. Requisitos

- **CUDA Toolkit 13.x** (probado con 13.3) con `nvcc` y `cl.exe` (MSVC) en Windows,
  o `g++` en Linux.
- **GPU NVIDIA** con compute capability ≥ 8.0 (probado en RTX 3060, sm_86).
- **Python 3.10+** con `pip`.
- **GNU Make** (opcional) o usar `build.bat` (Windows) / `build.sh` (Linux).

Dependencias Python:

```bash
# Windows
py -3.10 -m pip install Pillow numpy cuda.tile reportlab
py -3.10 -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu130

# Linux
python3 -m pip install Pillow numpy cuda.tile reportlab
python3 -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu130
```

> **Nota:** `reportlab` solo es necesario si se desea regenerar el PDF del
> informe desde `docs/informe.md` (`py -3.10 scripts/build_pdf.py`).
```

## 2. Compilación

### Windows

```bat
build.bat
```

### Linux

```bash
./build.sh
```

### Cross-platform (Make)

```bash
make
```

En Windows el Makefile y `build.bat` usan `-ccbin` para localizar `cl.exe`
de MSVC. Si la instalación de Visual Studio/BuildTools usa una ruta
distinta, actualizar la variable `CCBIN` en ambos archivos. En Linux ese
flag se omite automáticamente y `nvcc` usa `g++`.

Los tres métodos producen los mismos binarios en `build/`:

| Binario           | Etapas        | Toolchain                 |
|-------------------|---------------|---------------------------|
| `secuencial` / `secuencial.exe` | CPU     | `nvcc -std=c++17`         |
| `cuda` / `cuda.exe`             | GPU SIMT | `nvcc -std=c++17`         |
| `tile` / `tile.exe`             | GPU + `-enable-tile` | `nvcc -std=c++20 -enable-tile` |

El flag `-arch=native` se usa para detectar la arquitectura de la GPU local.

## 3. Ejecución

Todas las versiones comparten la misma CLI:

```bash
<binario> --instance=<small|medium|large|no-divisible> \
           --kernel-size=<3|5|7|9|11> \
           [--scale=<factor>]
```

| Bandera         | Valores                                            |
|-----------------|----------------------------------------------------|
| `--instance`    | `small` (512×512), `medium` (2048×2048), `large` (4096×4096), `no-divisible` (~1402×1122) |
| `--kernel-size` | Tamaño del kernel gaussiano (impar ≥ 3)             |
| `--scale`       | Factor de resize bilineal, e.g. `0.5` o `1.75`      |
| `--separable`   | (solo `cuda.exe`) usar convolución separable 2-pases |

### Ejemplos

```bash
# Windows
build\secuencial.exe --instance=small   --kernel-size=5  --scale=0.5
build\cuda.exe       --instance=medium  --kernel-size=9  --scale=1.75
build\cuda.exe       --instance=medium  --kernel-size=5  --scale=0.5 --separable
build\tile.exe       --instance=large   --kernel-size=5  --scale=1.0
py -3.10 src\cutile_pipeline.py --instance=no-divisible --kernel-size=5 --scale=0.5

# Linux
build/secuencial --instance=small   --kernel-size=5  --scale=0.5
build/cuda       --instance=medium  --kernel-size=9  --scale=1.75
build/cuda       --instance=medium  --kernel-size=5  --scale=0.5 --separable
build/tile       --instance=large   --kernel-size=5  --scale=1.0
python3 src/cutile_pipeline.py --instance=no-divisible --kernel-size=5 --scale=0.5
```

### Ejecución batch (4 versiones × 4 instancias × 2 kernels × 2 escalas)

```bat
:: Windows
run_all.bat
```

```bash
# Linux
./run_all.sh
```

Esto pobla `results/resultados.csv` con 800 filas (4 versiones + gaussiano separable en CUDA = 5 configs × 4 inst × 2 k × 2 s × 10 reps)
más la cabecera.

## 4. Profiling con Nsight

```bash
# Windows
profile.bat medium 5 0.5

# Linux
./profile.sh medium 5 0.5
```

Genera reportes en `results/perf/`:

- `nsys_<instance>_cuda.nsys-rep` — línea de tiempo, transferencias y kernels
  (Nsight Systems 2026.1.3).
- `ncu_<instance>_cuda.ncu-rep` — métricas por kernel: occupancy, memoria,
  instrucciones (Nsight Compute 2026.2.0).

> **Nota:** Nsight Compute requiere permisos de administrador para acceder a
> los GPU Performance Counters. En Windows ejecutar PowerShell como administrador;
> en Linux configurar `sudo ncu ...` o habilitar el registro de performance counters.

## 5. Estructura del proyecto

```
filtro-sobel-cuda/
├── AGENTS.md                       # contexto para subagentes
├── TRACKING.md                     # checklist contra la rúbrica
├── README.md                       # este archivo
├── Makefile                        # build cross-platform (make)
├── build.bat                       # build en Windows
├── build.sh                        # build en Linux
├── run_all.bat                     # ejecutar las 4 versiones (Windows)
├── run_all.sh                      # ejecutar las 4 versiones (Linux)
├── profile.bat                     # invocar Nsight (Windows)
├── profile.sh                      # invocar Nsight (Linux)
├── release.bat                     # empaquetar .zip de entrega
├── Actividad_4_INFO1195_2026_Actualizado.pdf
├── .gitignore
├── data/                           # imágenes de entrada
│   ├── small/pistola.png           # 512×512
│   ├── medium/ak-47.png            # 2048×2048
│   ├── large/m16.png               # 4096×4096
│   └── no-divisible/automovil.png  # 1402×1122
├── src/
│   ├── image.{hpp,cpp}             # CPU referencia
│   ├── main.cpp                    # CLI + orquestador CPU
│   ├── cuda_kernels.{hpp,cu}       # CUDA clásico + Occupancy API
│   ├── main_cuda.cpp               # CLI + orquestador CUDA clásico
│   ├── tile_kernels.{hpp,cu}       # Tile C++ (-enable-tile) + SIMT
│   ├── main_tile.cpp               # CLI + orquestador Tile
│   ├── cutile_pipeline.py          # cuTile Python + orquestador
│   ├── stb_image.h
│   └── stb_image_write.h
├── scripts/
│   ├── build_pdf.py                # genera docs/informe.pdf desde .md
│   ├── analyze_csv.py              # tablas resumen desde resultados.csv
│   └── generate_test_images.py     # imágenes sintéticas si faltan
├── docs/
│   ├── informe.md                  # informe técnico (fuente)
│   └── informe.pdf                 # informe en PDF (generado con build_pdf.py)
├── build/                          # binarios compilados (gitignored)
└── results/
    ├── resultados.csv              # CSV unificado de las 4 versiones
    ├── resultados_full.csv         # copia de respaldo
    ├── secuencial/<instancia>/     # salidas CPU
    ├── cuda/<instancia>/           # salidas CUDA clásico
    ├── tile/<instancia>/           # salidas Tile C++
    ├── cutile_python/<instancia>/  # salidas cuTile Python
    └── perf/                       # reportes Nsight
```

## 6. Formato del CSV

`results/resultados.csv` tiene 23 columnas:

| Columna                      | Significado                                           |
|------------------------------|-------------------------------------------------------|
| `Version`                    | `CPU_Secuencial`, `CUDA_Clasico`, `CUDA_Tile`, `cuTile_Python` |
| `Instancia`                  | `small` / `medium` / `large` / `no-divisible`         |
| `Imagen`                     | nombre del archivo de entrada                         |
| `DimensionesOrig`            | `WxH`                                                |
| `KernelSize`                 | tamaño del kernel gaussiano                           |
| `Scale`                      | factor de resize                                      |
| `Bloque`                     | `N/A` (CPU), `Dinamico_API` (CUDA), `16x16(SIMT_en_Tile)`, `16x16(cuTile)` |
| `Repeticion`                 | 1..10                                                 |
| `T_*_kernel_ms`              | tiempo de kernel por etapa (sin transferencias)       |
| `T_*_HtoD_ms`, `T_*_DtoH_ms` | tiempo de transferencia por etapa                     |
| `T_Total_ms`                 | suma de las 4 etapas (incluye transferencias)         |
| `Throughput_MPps`            | megapíxeles/segundo (basado en tiempo de kernel)      |
| `Herramienta`                | `chrono` / `CUDA_Events` / `torch.cuda.Event`        |

## 7. Decisión técnica sobre Tile C++

La API `cuda_tile.h` de CUDA 13.3 está habilitada en el proyecto
(`tileIdentityKernel` real en `src/tile_kernels.cu`, compilado con
`-enable-tile -std=c++20`), pero **no soporta stencils con acceso a
vecinos de forma práctica**. Las 4 etapas de la pipeline corren como SIMT
dentro del mismo `.cu`, lo cual está oficialmente soportado en CUDA 13.3
("CUDA C++ supports both Tile and SIMT code in the same translation unit").
Ver `TRACKING.md` sección G para más detalle.

## 8. Referencias

- Enunciado: `Actividad_4_INFO1195_2026_Actualizado.pdf`
- `stb_image.h`: https://github.com/nothings/stb
- cuTile Python: https://github.com/NVIDIA/cutile-python
- CUDA Tile C++: docs en `cuda_tile.h` (CUDA Toolkit)
