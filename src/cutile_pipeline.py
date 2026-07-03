"""
Etapa complementaria en cuTile Python.

Implementa conversion RGB -> luminancia usando CUDA Tile Python.
Compara contra la version CPU y CUDA C++ clasica.

Uso:
    py -3.10 cutile_pipeline.py --instance=<small|medium|large|no-divisible> \
                                 --kernel-size=<k> [--scale=<factor>]
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import cuda.tile as ct
from PIL import Image


TILE = 16


@ct.kernel
def rgb_to_gray_kernel(
    r_arr,
    g_arr,
    b_arr,
    out_arr,
    tile_h: ct.Constant[int],
    tile_w: ct.Constant[int]
):
    bx = ct.bid(0)
    by = ct.bid(1)

    r_tile = ct.load(
        r_arr, index=(bx, by), shape=(tile_h, tile_w),
        padding_mode=ct.PaddingMode.ZERO
    ).astype(ct.float32)
    g_tile = ct.load(
        g_arr, index=(bx, by), shape=(tile_h, tile_w),
        padding_mode=ct.PaddingMode.ZERO
    ).astype(ct.float32)
    b_tile = ct.load(
        b_arr, index=(bx, by), shape=(tile_h, tile_w),
        padding_mode=ct.PaddingMode.ZERO
    ).astype(ct.float32)

    gray = 0.299 * r_tile + 0.587 * g_tile + 0.114 * b_tile

    gray = ct.minimum(gray, 255.0)
    gray = ct.maximum(gray, 0.0)

    ct.store(
        out_arr, index=(bx, by),
        tile=gray.astype(ct.uint8)
    )


def rgb_to_gray_cutile(rgb_image: np.ndarray) -> np.ndarray:
    h, w, _ = rgb_image.shape
    r = torch.from_numpy(rgb_image[:, :, 0].astype(np.uint8)).cuda()
    g = torch.from_numpy(rgb_image[:, :, 1].astype(np.uint8)).cuda()
    b = torch.from_numpy(rgb_image[:, :, 2].astype(np.uint8)).cuda()

    out = torch.zeros((h, w), dtype=torch.uint8, device='cuda')

    grid = (
        (h + TILE - 1) // TILE,
        (w + TILE - 1) // TILE,
        1
    )
    ct.launch(
        torch.cuda.current_stream(), grid, rgb_to_gray_kernel,
        (r, g, b, out, TILE, TILE)
    )
    torch.cuda.synchronize()

    return out.cpu().numpy()


def process_image(input_path: Path, output_path: Path) -> None:
    img = Image.open(input_path).convert('RGB')
    rgb = np.array(img, dtype=np.uint8)

    gray = rgb_to_gray_cutile(rgb)

    Image.fromarray(gray, mode='L').save(output_path)
    print(f"Grayscale guardado: {output_path} ({gray.shape[1]}x{gray.shape[0]})")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Etapa cuTile Python: RGB -> luminancia"
    )
    parser.add_argument(
        "--instance",
        required=True,
        choices=["small", "medium", "large", "no-divisible"]
    )
    parser.add_argument(
        "--kernel-size", type=int, default=5,
        help="Ignorado, presente por compatibilidad CLI con secuencial/cuda/tile"
    )
    parser.add_argument(
        "--scale", type=float, default=1.0,
        help="Ignorado, presente por compatibilidad CLI con secuencial/cuda/tile"
    )
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
        out_path = out_dir / f"{img_path.stem}_gray.png"
        print(f"Procesando: {img_path.name}")
        process_image(img_path, out_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
