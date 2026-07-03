"""
Genera imagenes sinteticas para los 4 tamanos del diseno experimental.

Por defecto respeta las imagenes existentes. Use --force para sobrescribir.

Uso:
    py -3.10 scripts\generate_test_images.py
    py -3.10 scripts\generate_test_images.py --force
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def make_image(width: int, height: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)

    bg = rng.integers(40, 90, size=(height, width, 3), dtype=np.uint8)
    img = bg.copy()

    n_shapes = max(8, (width * height) // (80 * 80))
    for _ in range(n_shapes):
        cx = rng.integers(0, width)
        cy = rng.integers(0, height)
        r = rng.integers(10, max(15, min(width, height) // 8))
        color = tuple(int(c) for c in rng.integers(60, 255, size=3))
        thickness = rng.integers(1, max(2, r // 3))

        shape = rng.choice(['rect', 'circle', 'line'])
        if shape == 'rect':
            x0 = max(0, cx - r); x1 = min(width - 1, cx + r)
            y0 = max(0, cy - r); y1 = min(height - 1, cy + r)
            img[y0:y1, x0] = color
            img[y0:y1, x1] = color
            img[y0, x0:x1] = color
            img[y1, x0:x1] = color
        elif shape == 'circle':
            yy, xx = np.ogrid[:height, :width]
            mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
            if mask.any():
                edge = ((xx - cx) ** 2 + (yy - cy) ** 2 <= r * r) & (
                    (xx - cx) ** 2 + (yy - cy) ** 2 >= (r - thickness) ** 2
                )
                img[edge] = color
        else:
            angle = rng.uniform(0, np.pi)
            dx = int(r * np.cos(angle)); dy = int(r * np.sin(angle))
            steps = max(2, 2 * r)
            for t in np.linspace(0, 1, steps):
                px = int(cx + dx * t); py = int(cy + dy * t)
                if 0 <= px < width and 0 <= py < height:
                    img[py, px] = color

    for _ in range(n_shapes // 2):
        cx = rng.integers(0, width)
        cy = rng.integers(0, height)
        r = rng.integers(20, max(30, min(width, height) // 6))
        color = tuple(int(c) for c in rng.integers(120, 255, size=3))
        yy, xx = np.ogrid[:height, :width]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
        img[mask] = color

    noise = rng.integers(-15, 15, size=(height, width, 3), dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


INSTANCES = {
    'small':        (512,  512),
    'medium':       (2048, 2048),
    'large':        (4096, 4096),
    'no-divisible': (1402, 1122),
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true',
                        help='Sobrescribir imagenes existentes')
    args = parser.parse_args()

    base = Path(__file__).resolve().parent.parent
    data_dir = base / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)

    for inst, (w, h) in INSTANCES.items():
        out_dir = data_dir / inst
        out_dir.mkdir(parents=True, exist_ok=True)

        existing = list(out_dir.glob('*.png'))
        if existing and not args.force:
            print(f"[skip] {inst}: {len(existing)} imagen(es) ya existen. Use --force para sobrescribir.")
            continue

        seed = sum(ord(c) for c in inst)
        img = make_image(w, h, seed)
        out_path = out_dir / f"synthetic_{inst}.png"
        Image.fromarray(img, mode='RGB').save(out_path)
        print(f"[ok] {inst}: {out_path} ({w}x{h})")

    return 0


if __name__ == '__main__':
    sys.exit(main())
