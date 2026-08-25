#!/usr/bin/env python3
"""Esporta un campione casuale del dataset di unione con la bbox disegnata,
per un controllo visivo rapido dei problemi più macroscopici.

Ad ogni esecuzione la cartella di output viene svuotata e ripopolata con un
nuovo campione casuale (nessun seed fisso): per rigenerare basta rilanciare
lo script.
"""
import argparse
import random
import shutil
from pathlib import Path

from PIL import Image, ImageDraw

import config

UNION_DIR = config.UNION_DIR
OUT_DIR = config.UNION_REVIEW_SAMPLE_DIR
BOX_COLOR = config.BOX_COLOR
BOX_WIDTH = config.BOX_WIDTH


def draw_boxes(image_path: Path, label_path: Path, dest: Path) -> None:
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        w_px, h_px = im.size
        draw = ImageDraw.Draw(im)
        if label_path.exists():
            for line in label_path.read_text().splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                _, xc, yc, w, h = parts
                xc, yc, w, h = float(xc) * w_px, float(yc) * h_px, float(w) * w_px, float(h) * h_px
                box = (xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2)
                draw.rectangle(box, outline=BOX_COLOR, width=BOX_WIDTH)
        im.save(dest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-n", "--num-samples", type=int, default=150,
                         help="Numero di immagini nel campione (default: 150)")
    args = parser.parse_args()

    images = sorted((UNION_DIR / "images").iterdir())
    sample = random.sample(images, min(args.num_samples, len(images)))

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True)

    for img_path in sample:
        label_path = UNION_DIR / "labels" / f"{img_path.stem}.txt"
        draw_boxes(img_path, label_path, OUT_DIR / img_path.name)

    print(f"{len(sample)} immagini campionate (su {len(images)}) con bbox disegnata in {OUT_DIR}")


if __name__ == "__main__":
    main()
