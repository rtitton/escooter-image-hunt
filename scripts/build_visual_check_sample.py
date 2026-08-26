#!/usr/bin/env python3
"""Esporta un campione casuale del dataset di unione con la bbox disegnata,
per un controllo visivo rapido dei problemi più macroscopici.

Oltre alle bbox escooter (dalle label del dataset di unione) disegna anche,
con un colore diverso, le bbox di classe "person" già calcolate dal filtro
varietà di select_images.py e cache su disco (orientazione originale, 0°):
nessuna nuova inferenza YOLO viene eseguita da questo script. Sul lato alto
di ogni bbox person che ne tocca una escooter è scritto il rapporto fra le
aree (bbox escooter / bbox person), utile per tarare a occhio le soglie di
sovrapposizione usate da select_images.py per il filtro conducente incluso.

Ad ogni esecuzione la cartella di output viene svuotata e ripopolata con un
nuovo campione casuale (nessun seed fisso): per rigenerare basta rilanciare
lo script.
"""
import argparse
import json
import random
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

import config

# UNION_DIR = config.UNION_DIR
# OUT_DIR = config.UNION_REVIEW_SAMPLE_DIR
# L'OUT_DIR sarà passato come argomento della riga di comando
BOX_COLOR = config.BOX_COLOR
BOX_WIDTH = config.BOX_WIDTH
RATIO_FONT = ImageFont.load_default(size=24)
PERSON_BOX_COLOR = config.PERSON_BOX_COLOR
VARIETY_CACHE_PATH = config.VARIETY_CACHE_PATH
PERSON_CLASS_ID = config.PERSON_CLASS_ID


def load_person_boxes_index(cache_path: Path) -> dict:
    """Mappa (dataset_id, nome_file) -> lista di bbox 'persona' (xyxy in
    pixel, orientazione originale), lette dalla cache del filtro varietà.
    Lo split non serve come chiave: i nomi file sono già unici entro un
    dataset (v. build_union_dataset.py, che li usa così com'è per il dataset
    di unione)."""
    if not cache_path.exists():
        return {}
    cache = json.loads(cache_path.read_text())
    index = {}
    for key, entry in cache.items():
        dataset_id, _split, _images, filename = key.split("/")
        detections_0 = entry.get("detections", {}).get("0", [])
        index[(dataset_id, filename)] = [tuple(det[1:]) for det in detections_0 if det[0] == PERSON_CLASS_ID]
    return index


def person_boxes_for_union_image(img_path: Path, person_index: dict) -> list:
    """Recupera le bbox 'persona' cache per un'immagine del dataset di unione,
    il cui nome è "{dataset_id}__{nome originale}" (v. build_union_dataset.py)."""
    dataset_id, _sep, orig_stem = img_path.stem.partition("__")
    filename = orig_stem + img_path.suffix
    return person_index.get((dataset_id, filename), [])


def box_area(box: tuple) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)

def box_height(box: tuple) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, y2 - y1)

def intersection_area(box_a: tuple, box_b: tuple) -> float:
    x1, y1 = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    x2, y2 = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def best_matching_escooter_box(person_box: tuple, escooter_boxes: list) -> tuple | None:
    """Bbox escooter che si sovrappone di più a person_box (per area di
    intersezione), o None se nessuna la tocca."""
    overlapping = [b for b in escooter_boxes if intersection_area(person_box, b) > 0]
    if not overlapping:
        return None
    return max(overlapping, key=lambda b: intersection_area(person_box, b))


def draw_boxes(image_path: Path, label_path: Path, person_boxes: list, dest: Path) -> None:
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        w_px, h_px = im.size
        draw = ImageDraw.Draw(im)
        escooter_boxes = []
        if label_path.exists():
            for line in label_path.read_text().splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                _, xc, yc, w, h = parts
                xc, yc, w, h = float(xc) * w_px, float(yc) * h_px, float(w) * w_px, float(h) * h_px
                box = (xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2)
                escooter_boxes.append(box)
                draw.rectangle(box, outline=BOX_COLOR, width=BOX_WIDTH)
        for person_box in person_boxes:
            draw.rectangle(person_box, outline=PERSON_BOX_COLOR, width=BOX_WIDTH)
            escooter_box = best_matching_escooter_box(person_box, escooter_boxes)
            if escooter_box is None:
                continue
            escooter_box_height = box_height(escooter_box)
            person_box_height = box_height(person_box)
            ratio = box_height(escooter_box) / box_height(person_box)
            x1, y1, _x2, _y2 = person_box
            label = f"{ratio:.2f}"
            label = f"{ratio:.2f} ({escooter_box_height:.0f}px / {person_box_height:.0f}px)"
            # text_box = draw.textbbox((x1, y1), label, font=RATIO_FONT, anchor="lb")
            # draw.rectangle(text_box, fill=PERSON_BOX_COLOR)
            draw.text((x1, y1), label, font=RATIO_FONT, fill="black", anchor="lb")
        im.save(dest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-n", "--num-samples", type=int, default=150,
                         help="Numero di immagini nel campione (default: 150)")
    parser.add_argument("-d", "--data-root", type=Path, default=config.UNION_DIR,
                         help=f"Cartella dati (default: {config.UNION_DIR})")
    parser.add_argument("-o", "--out-dir", type=Path, default=config.UNION_REVIEW_SAMPLE_DIR,
                         help=f"Cartella di output (default: {config.UNION_REVIEW_SAMPLE_DIR})")
    args = parser.parse_args()

    images = sorted((args.data_root / "images").iterdir())
    sample = random.sample(images, min(args.num_samples, len(images)))

    if args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True)

    person_index = load_person_boxes_index(VARIETY_CACHE_PATH)
    for img_path in sample:
        label_path = args.data_root / "labels" / f"{img_path.stem}.txt"
        person_boxes = person_boxes_for_union_image(img_path, person_index)
        draw_boxes(img_path, label_path, person_boxes, args.out_dir / img_path.name)

    print(f"{len(sample)} immagini campionate (su {len(images)}) con bbox disegnata in {args.out_dir}")


if __name__ == "__main__":
    main()
