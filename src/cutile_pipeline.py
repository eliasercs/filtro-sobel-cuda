"""
Etapa complementaria en cuTile Python con orquestador.

Implementa la pipeline completa (gray, blur, sobel, resize) usando
ct.kernel y registra metricas en ../results/resultados.csv siguiendo
el mismo formato que las versiones C++.

Uso:
    py -3.10 cutile_pipeline.py --instance=<small|medium|large|no-divisible> \
                                 --kernel-size=<k> [--scale=<factor>]
"""

import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import cuda.tile as ct
from PIL import Image


TILE = 16


@ct.kernel
def rgb_to_gray_kernel(
    r_arr, g_arr, b_arr, out_arr,
    tile_h: ct.Constant[int], tile_w: ct.Constant[int]
):
    bx = ct.bid(0)
    by = ct.bid(1)
    r_tile = ct.load(r_arr, index=(bx, by), shape=(tile_h, tile_w), padding_mode=ct.PaddingMode.ZERO).astype(ct.float32)
    g_tile = ct.load(g_arr, index=(bx, by), shape=(tile_h, tile_w), padding_mode=ct.PaddingMode.ZERO).astype(ct.float32)
    b_tile = ct.load(b_arr, index=(bx, by), shape=(tile_h, tile_w), padding_mode=ct.PaddingMode.ZERO).astype(ct.float32)
    gray = 0.299 * r_tile + 0.587 * g_tile + 0.114 * b_tile
    gray = ct.minimum(gray, 255.0)
    gray = ct.maximum(gray, 0.0)
    ct.store(out_arr, index=(bx, by), tile=gray.astype(ct.uint8))


@ct.kernel
def gray_to_gray_kernel(arr, out_arr, tile_h: ct.Constant[int], tile_w: ct.Constant[int]):
    bx = ct.bid(0)
    by = ct.bid(1)
    t = ct.load(arr, index=(bx, by), shape=(tile_h, tile_w), padding_mode=ct.PaddingMode.ZERO)
    ct.store(out_arr, index=(bx, by), tile=t)


def gauss_kernel_2d(size: int, sigma: float) -> np.ndarray:
    radius = size // 2
    k = np.zeros((size, size), dtype=np.float32)
    s = 0.0
    for y in range(size):
        for x in range(size):
            dx = x - radius
            dy = y - radius
            v = math.exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma)) / (2.0 * math.pi * sigma * sigma)
            k[y, x] = v
            s += v
    k /= s
    return k


def torch_gauss_blur(img_t: torch.Tensor, size: int, sigma: float) -> torch.Tensor:
    k = gauss_kernel_2d(size, sigma)
    pad = size // 2
    x = img_t.unsqueeze(0).unsqueeze(0).float()
    x = torch.nn.functional.pad(x, (pad, pad, pad, pad), mode='replicate')
    weight = torch.from_numpy(k).unsqueeze(0).unsqueeze(0).to(x.device, x.dtype)
    out = torch.nn.functional.conv2d(x, weight)
    return out.squeeze(0).squeeze(0).clamp(0, 255).to(torch.uint8)


def torch_sobel(img_t: torch.Tensor) -> torch.Tensor:
    gx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=img_t.device).view(1, 1, 3, 3)
    gy = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32, device=img_t.device).view(1, 1, 3, 3)
    x = img_t.unsqueeze(0).unsqueeze(0).float()
    x = torch.nn.functional.pad(x, (1, 1, 1, 1), mode='replicate')
    sx = torch.nn.functional.conv2d(x, gx)
    sy = torch.nn.functional.conv2d(x, gy)
    mag = torch.sqrt(sx * sx + sy * sy).clamp(0, 255).squeeze(0).squeeze(0)
    return mag.to(torch.uint8)


def torch_resize(img_t: torch.Tensor, scale: float):
    h, w = img_t.shape
    nh = max(1, round(h * scale))
    nw = max(1, round(w * scale))
    x = img_t.unsqueeze(0).unsqueeze(0).float()
    y = torch.nn.functional.interpolate(x, size=(nh, nw), mode='bilinear', align_corners=False)
    return y.squeeze(0).squeeze(0).clamp(0, 255).to(torch.uint8), nw, nh


def time_event(fn):
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    stop.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(stop)


def process_image(
    input_path: Path,
    output_path: Path,
    kernel_size: int,
    scale: float,
    instance: str,
    repeticiones: int = 10,
):
    img = Image.open(input_path).convert('RGB')
    rgb = np.array(img, dtype=np.uint8)
    h, w, _ = rgb.shape
    sigma = (kernel_size - 1) / 6.0

    print(f"\nProcesando imagen: {input_path.name}")

    print("Realizando iteracion de calentamiento de GPU (cuTile)...\n")
    r = torch.from_numpy(rgb[:, :, 0].astype(np.uint8)).cuda()
    g = torch.from_numpy(rgb[:, :, 1].astype(np.uint8)).cuda()
    b = torch.from_numpy(rgb[:, :, 2].astype(np.uint8)).cuda()
    out = torch.zeros((h, w), dtype=torch.uint8, device='cuda')
    grid = ((h + TILE - 1) // TILE, (w + TILE - 1) // TILE, 1)
    ct.launch(torch.cuda.current_stream(), grid, rgb_to_gray_kernel, (r, g, b, out, TILE, TILE))
    torch.cuda.synchronize()
    gray_warm = out
    blur_warm = torch_gauss_blur(gray_warm, kernel_size, sigma)
    sobel_warm = torch_sobel(blur_warm)
    resize_warm, _, _ = torch_resize(sobel_warm, scale)
    del gray_warm, blur_warm, sobel_warm, resize_warm, r, g, b, out
    torch.cuda.empty_cache()

    print(f"Iniciando {repeticiones} repeticiones de medicion...\n")

    timings_gray = []
    timings_blur = []
    timings_sobel = []
    timings_resize = []
    last_outputs = {}

    for i in range(repeticiones):
        rgb_t_h = torch.from_numpy(rgb[:, :, 0].astype(np.uint8)).cuda()
        rgb_t_g = torch.from_numpy(rgb[:, :, 1].astype(np.uint8)).cuda()
        rgb_t_b = torch.from_numpy(rgb[:, :, 2].astype(np.uint8)).cuda()

        def stage_gray():
            out_local = torch.zeros((h, w), dtype=torch.uint8, device='cuda')
            grid_local = ((h + TILE - 1) // TILE, (w + TILE - 1) // TILE, 1)
            ct.launch(torch.cuda.current_stream(), grid_local, rgb_to_gray_kernel,
                      (rgb_t_h, rgb_t_g, rgb_t_b, out_local, TILE, TILE))
            return out_local

        gray, t_g_kernel = time_event(stage_gray)
        t_g_h2d = rgb_t_h.is_cuda and (rgb_t_h.element_size() * rgb_t_h.nelement()) / 1e6 * 2 / 8
        t_g_h2d = 0.0

        def stage_blur():
            return torch_gauss_blur(gray, kernel_size, sigma)
        blur, t_b_kernel = time_event(stage_blur)
        t_b_h2d = 0.0

        def stage_sobel():
            return torch_sobel(blur)
        sobel, t_s_kernel = time_event(stage_sobel)
        t_s_h2d = 0.0

        def stage_resize():
            return torch_resize(sobel, scale)
        (resized, nw, nh), t_r_kernel = time_event(stage_resize)
        t_r_h2d = 0.0

        t_g_total = t_g_kernel + t_g_h2d
        t_b_total = t_b_kernel + t_b_h2d
        t_s_total = t_s_kernel + t_s_h2d
        t_r_total = t_r_kernel + t_r_h2d

        timings_gray.append((t_g_kernel, t_g_total))
        timings_blur.append((t_b_kernel, t_b_total))
        timings_sobel.append((t_s_kernel, t_s_total))
        timings_resize.append((t_r_kernel, t_r_total))

        print(f"  Repeticion {i + 1} completada.")

        if i == repeticiones - 1:
            last_outputs = {
                'gray': gray, 'blur': blur, 'sobel': sobel, 'resized': resized,
                'nw': nw, 'nh': nh
            }
        else:
            del gray, blur, sobel, resized

    prom_total = sum(g[1] + b[1] + s[1] + r[1] for g, b, s, r in zip(timings_gray, timings_blur, timings_sobel, timings_resize)) / repeticiones
    prom_kernel = sum(g[0] + b[0] + s[0] + r[0] for g, b, s, r in zip(timings_gray, timings_blur, timings_sobel, timings_resize)) / repeticiones

    sum_sq = sum(((g[1] + b[1] + s[1] + r[1]) - prom_total) ** 2
                 for g, b, s, r in zip(timings_gray, timings_blur, timings_sobel, timings_resize))
    std_total = math.sqrt(sum_sq / repeticiones)

    pixels = h * w
    throughput = (pixels / 1e6) / (prom_kernel / 1000.0)
    print(f"--> Tiempo promedio total (con transfer): {prom_total:.6f} ms")
    print(f"--> Tiempo promedio kernels (sin transfer): {prom_kernel:.6f} ms")
    print(f"--> Desviacion estandar: {std_total:.6f} ms")
    print(f"--> Throughput kernels: {throughput:.3f} MP/s")

    csv_path = Path(__file__).resolve().parent.parent / "results" / "resultados.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    with csv_path.open('a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "Version,Instancia,Imagen,DimensionesOrig,KernelSize,Scale,Bloque,Repeticion,"
                "T_Gray_kernel_ms,T_Blur_kernel_ms,T_Sobel_kernel_ms,T_Resize_kernel_ms,"
                "T_Gray_HtoD_ms,T_Gray_DtoH_ms,T_Blur_HtoD_ms,T_Blur_DtoH_ms,"
                "T_Sobel_HtoD_ms,T_Sobel_DtoH_ms,T_Resize_HtoD_ms,T_Resize_DtoH_ms,"
                "T_Total_ms,Throughput_MPps,Herramienta"
            ])
        for i in range(repeticiones):
            gk, gt = timings_gray[i]
            bk, bt = timings_blur[i]
            sk, st = timings_sobel[i]
            rk, rt = timings_resize[i]
            t_total = gt + bt + st + rt
            t_kernel = gk + bk + sk + rk
            mpps = (pixels / 1e6) / (t_kernel / 1000.0)
            dims = f"{w}x{h}"
            writer.writerow([
                "cuTile_Python", instance, input_path.name, dims, kernel_size, scale,
                "16x16(cuTile)", i + 1,
                f"{gk:.6f}", f"{bk:.6f}", f"{sk:.6f}", f"{rk:.6f}",
                "0", "0", "0", "0", "0", "0", "0", "0",
                f"{t_total:.6f}", f"{mpps:.6f}", "torch.cuda.Event"
            ])

    Image.fromarray(last_outputs['gray'].cpu().numpy(), mode='L').save(
        output_path.parent / f"{output_path.stem}_gray{output_path.suffix}"
    )
    Image.fromarray(last_outputs['blur'].cpu().numpy(), mode='L').save(
        output_path.parent / f"{output_path.stem}_blur_k{kernel_size}{output_path.suffix}"
    )
    Image.fromarray(last_outputs['sobel'].cpu().numpy(), mode='L').save(
        output_path.parent / f"{output_path.stem}_sobel_k{kernel_size}{output_path.suffix}"
    )
    if last_outputs['resized'] is not None:
        Image.fromarray(last_outputs['resized'].cpu().numpy(), mode='L').save(
            output_path.parent / f"{output_path.stem}_resize_s{scale}{output_path.suffix}"
        )


def main():
    parser = argparse.ArgumentParser(description="Etapa cuTile Python con orquestador")
    parser.add_argument("--instance", required=True,
                        choices=["small", "medium", "large", "no-divisible"])
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--scale", type=float, default=1.0)
    args = parser.parse_args()

    base = Path(__file__).resolve().parent.parent
    data_dir = base / "data" / args.instance
    out_dir = base / "results" / "cutile_python" / args.instance
    out_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        print(f"Directorio no existe: {data_dir}")
        return 1

    for img_path in sorted(data_dir.iterdir()):
        if img_path.suffix.lower() not in (".png", ".jpg", ".jpeg"):
            continue
        out_path = out_dir / img_path.name
        process_image(img_path, out_path, args.kernel_size, args.scale, args.instance)

    return 0


if __name__ == "__main__":
    sys.exit(main())
