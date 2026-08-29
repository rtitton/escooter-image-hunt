#!/usr/bin/env python3
"""Costruisce un campione di immagini a partire da data/image_index.json,
filtrando su condizioni a piacere sugli attributi di ciascuna voce
dell'indice (dataset_id, split, image_path, width, height, excluded,
exclusion_stage, exclusion_reason), e le salva in una cartella con le
bounding box escooter disegnate sopra.

Utile per ispezionare a occhio un sottoinsieme di immagini scelto in base
alle decisioni prese da select_images.py (es. solo le scartate per soglia
di area, solo quelle di un certo dataset, solo quelle oltre una certa
dimensione, ecc.) senza dover rilanciare la pipeline né toccare il codice.

Il filtro è un'espressione Python valutata su ogni voce dell'indice, con i
suoi campi disponibili come variabili:

    python3 scripts/build_index_sample.py \
        --filter "exclusion_stage == 'cheap' and 'lontana' in (exclusion_reason or '')" \
        --out-dir data/tmp/sample_faraway

Ad ogni esecuzione la cartella di output viene svuotata e ripopolata.
"""
import argparse
import json
import random
import shutil
from pathlib import Path

import yaml
from PIL import Image, ImageDraw, ImageFont

import config
import dataset_index
from select_images import escooter_class_indices, read_boxes

IMAGE_INDEX_PATH = config.IMAGE_INDEX_PATH
DATA_ROOT = config.DATA_ROOT
BOX_COLOR = config.BOX_COLOR
BOX_WIDTH = config.BOX_WIDTH

# builtin sicuri, utili nelle espressioni --filter senza esporre l'intero
# namespace dei builtin
SAFE_BUILTINS = {
    "abs": abs, "min": min, "max": max, "len": len,
    "str": str, "int": int, "float": float, "bool": bool,
    "any": any, "all": all,
}


def load_index() -> list[dict]:
    return json.loads(IMAGE_INDEX_PATH.read_text())


def matches(record: dict, expr: str) -> bool:
    try:
        return bool(eval(expr, {"__builtins__": SAFE_BUILTINS}, record))
    except NameError as e:
        raise SystemExit(f"--filter: attributo sconosciuto ({e}). Attributi disponibili: {sorted(record)}")


def escooter_boxes_for(record: dict, entry: dict | None, yaml_cache: dict) -> list:
    """Bbox escooter (xyxy in pixel) per l'immagine, lette dal file di label
    del dataset deduplicato corrispondente."""
    if entry is None:
        return []
    repo_root = DATA_ROOT.parent
    dedup_dir = repo_root / entry["dedup_dir"]
    label_path = dedup_dir / record["split"] / "labels" / (Path(record["image_path"]).stem + ".txt")
    if not label_path.exists():
        return []

    if entry["id"] not in yaml_cache:
        data_yaml_path = dedup_dir / "data.yaml"
        names = yaml.safe_load(data_yaml_path.read_text()).get("names", []) if data_yaml_path.exists() else []
        yaml_cache[entry["id"]] = escooter_class_indices(entry, names)
    escooter_idx = yaml_cache[entry["id"]]

    w_px, h_px = record["width"], record["height"]
    boxes = []
    for cls, xc, yc, w, h in read_boxes(label_path):
        if cls not in escooter_idx:
            continue
        xc, yc, w, h = xc * w_px, yc * h_px, w * w_px, h * h_px
        boxes.append((xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2))
    return boxes


def _text_with_background(draw: ImageDraw.ImageDraw, xy: tuple, text: str,
                          font: ImageFont.ImageFont, anchor: str = "la") -> None:
    """Scrive `text` con un rettangolo di sfondo per renderlo leggibile
    sopra qualsiasi immagine."""
    x0, y0, x1, y1 = draw.textbbox(xy, text, font=font, anchor=anchor)
    draw.rectangle((x0 - 2, y0 - 1, x1 + 2, y1 + 1), fill=(0, 0, 0))
    draw.text(xy, text, font=font, fill=(255, 255, 255), anchor=anchor)


def draw_boxes(image_path: Path, boxes: list, dest: Path) -> None:
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        img_w, img_h = im.size
        img_area = img_w * img_h
        draw = ImageDraw.Draw(im)
        font = ImageFont.load_default()
        for box in boxes:
            draw.rectangle(box, outline=BOX_COLOR, width=BOX_WIDTH)
            x0, y0, x1, y1 = box
            box_w, box_h = x1 - x0, y1 - y0
            ratio = (box_w * box_h) / img_area if img_area else 0.0
            label = f"area/img={ratio:.4f}  h={box_h:.0f}px  w={box_w:.0f}px"
            _text_with_background(draw, (x0, max(0, y0 - 12)), label, font)
        _text_with_background(draw, (2, img_h - 2),
                              f"immagine: h={img_h}px  w={img_w}px", font, anchor="ld")
        im.save(dest)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--filter", default="True",
                         help="Espressione Python valutata su ogni voce dell'indice (default: nessun filtro, "
                              "tutte le immagini). Campi disponibili: dataset_id, split, image_path, width, "
                              "height, excluded, exclusion_stage, exclusion_reason.")
    parser.add_argument("-n", "--num-samples", type=int, default=150,
                         help="Numero massimo di immagini nel campione, scelte a caso fra quelle che soddisfano "
                              "il filtro (default: 150; 0 = tutte)")
    parser.add_argument("-o", "--out-dir", type=Path, required=True, help="Cartella di output")
    args = parser.parse_args()

    records = load_index()
    matching = [r for r in records if matches(r, args.filter)]
    print(f"{len(matching)}/{len(records)} immagini dell'indice soddisfano il filtro.")

    sample = matching if not args.num_samples else random.sample(matching, min(args.num_samples, len(matching)))

    if args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True)

    dataset_entries = {e["id"]: e for e in dataset_index.load_index()}
    yaml_cache: dict = {}
    for record in sample:
        image_path = Path(record["image_path"])
        entry = dataset_entries.get(record["dataset_id"])
        boxes = escooter_boxes_for(record, entry, yaml_cache)
        dest = args.out_dir / f"{record['dataset_id']}__{image_path.name}"
        draw_boxes(image_path, boxes, dest)

    print(f"{len(sample)} immagini campionate con bbox disegnata in {args.out_dir}")


if __name__ == "__main__":
    main()
