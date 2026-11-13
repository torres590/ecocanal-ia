"""Genera el logotipo institucional de ECOCANAL IA sin dependencias externas."""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT = BASE_DIR / "assets" / "logo.png"
SIZE = 512


def _bezier(p0, p1, p2, p3, steps: int = 40):
    points = []
    for index in range(steps + 1):
        t = index / steps
        u = 1 - t
        x = u**3 * p0[0] + 3 * u**2 * t * p1[0] + 3 * u * t**2 * p2[0] + t**3 * p3[0]
        y = u**3 * p0[1] + 3 * u**2 * t * p1[1] + 3 * u * t**2 * p2[1] + t**3 * p3[1]
        points.append((x, y))
    return points


def _gradient_background() -> Image.Image:
    image = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    pixels = image.load()
    for y in range(SIZE):
        for x in range(SIZE):
            ratio = (x + y) / (2 * SIZE)
            start = (11, 61, 46)
            end = (6, 38, 29)
            color = tuple(
                int(start[channel] + (end[channel] - start[channel]) * ratio)
                for channel in range(3)
            )
            pixels[x, y] = (*color, 255)
    mask = Image.new("L", (SIZE, SIZE), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (12, 12, SIZE - 12, SIZE - 12), radius=118, fill=255
    )
    clipped = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    clipped.paste(image, (0, 0), mask)
    return clipped


def _drop_shape() -> list[tuple[float, float]]:
    top = (256, 104)
    left = _bezier(top, (224, 142), (156, 222), (156, 302))
    center = (256, 302)
    radius = 100
    bottom = []
    for step in range(41):
        angle = math.pi + (math.pi * step / 40)
        bottom.append(
            (center[0] + radius * math.cos(angle), center[1] + radius * math.sin(angle))
        )
    right = list(
        reversed(_bezier(top, (288, 142), (356, 222), (356, 302), steps=40)
        )
    )
    return left + bottom + right


def _leaf_points() -> list[tuple[float, float]]:
    # Hoja estilizada de dos curvas, apuntando hacia arriba.
    base = (248, 330)
    tip = (248, 196)
    left = _bezier(base, (190, 274), (204, 220), tip, steps=36)
    right = _bezier(tip, (292, 220), (306, 274), base, steps=36)
    return left + right


def build_logo() -> Image.Image:
    image = _gradient_background()
    draw = ImageDraw.Draw(image, "RGBA")

    # Halo tecnológico.
    halo = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    halo_draw = ImageDraw.Draw(halo)
    halo_draw.ellipse((74, 74, 438, 438), outline=(70, 185, 142, 45), width=3)
    halo_draw.ellipse((96, 96, 416, 416), outline=(99, 183, 216, 38), width=2)
    image = Image.alpha_composite(image, halo)
    draw = ImageDraw.Draw(image, "RGBA")

    # Onda de agua inferior.
    wave = []
    for step in range(61):
        x = 108 + (296 * step / 60)
        y = 384 + 9 * math.sin((step / 60) * math.pi * 2.2)
        wave.append((x, y))
    draw.line(wave, fill=(99, 183, 216, 150), width=8, joint="curve")
    draw.line(wave, fill=(127, 224, 192, 150), width=3, joint="curve")

    # Gota: sombra, relleno y contorno.
    drop = _drop_shape()
    shadow = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow)
    shadow_draw.polygon([(x + 6, y + 10) for x, y in drop], fill=(0, 0, 0, 70))
    shadow = shadow.filter(ImageFilter.GaussianBlur(8))
    image = Image.alpha_composite(image, shadow)
    draw = ImageDraw.Draw(image, "RGBA")
    draw.polygon(drop, fill=(212, 247, 235, 255))
    draw.line(drop + [drop[0]], fill=(127, 224, 192, 255), width=5, joint="curve")

    # Hoja central.
    draw.polygon(_leaf_points(), fill=(31, 138, 112, 255))
    draw.line([(248, 316), (248, 222)], fill=(11, 61, 46, 255), width=6)
    for y_offset in (258, 286, 312):
        draw.line([(248, y_offset), (224, y_offset - 18)], fill=(11, 61, 46, 255), width=4)
        draw.line([(248, y_offset), (272, y_offset - 18)], fill=(11, 61, 46, 255), width=4)

    # Nodos de tecnología ambiental.
    draw.line([(332, 172), (382, 138), (402, 172)], fill=(99, 183, 216, 180), width=3)
    for x, y in ((332, 172), (382, 138), (402, 172)):
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=(99, 183, 216, 230))
    return image


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    build_logo().save(OUTPUT)
    print(f"Logo generado en: {OUTPUT}")


if __name__ == "__main__":
    main()
