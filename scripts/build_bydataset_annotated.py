#!/usr/bin/env python3
"""Esporta la selezione finale (di norma il dataset di unione,
data/processed/union/) in una cartella per ogni dataset sorgente, con tutte
le immagini annotate: su ciascuna vengono disegnate solo le bounding box
escooter lette dai label.

A differenza di build_visual_check_sample.py — che esporta un campione
casuale in un'unica cartella — qui vengono copiate *tutte* le immagini della
selezione e restano separate per dataset di origine, ricavato dal prefisso
del nome file (`<dataset_id>__<nome originale>`, v. build_union_dataset.py):

    data/processed/bydataset/<dataset_id>/<dataset_id>__<nome>.jpg

Ad ogni esecuzione la cartella di output viene svuotata e ripopolata.

Con --decisions-file si può filtrare la selezione con un file di decisioni
prodotto da review_app.py (data/processed/union/review_decisions.json):
di default vengono tenute solo le immagini con decisione "select" o
"reserve"; con --include-discard si tengono tutte quelle con una decisione.
Senza --decisions-file vengono esportate tutte le immagini della sorgente.
"""
import argparse
import json
import shutil
from pathlib import Path

from PIL import Image, ImageDraw

import config

BOX_COLOR = config.BOX_COLOR
BOX_WIDTH = config.BOX_WIDTH


def load_selected_names(decisions_file: Path | None, include_discard: bool) -> set[str] | None:
    """Nomi immagine da tenere secondo il file di decisioni, o None se non è
    stato passato alcun file (= tieni tutto)."""
    if decisions_file is None:
        return None
    if not decisions_file.exists():
        raise SystemExit(f"File di decisioni non trovato: {decisions_file}")
    decisions = json.loads(decisions_file.read_text())
    keep = {"select", "reserve", "discard"} if include_discard else {"select", "reserve"}
    return {name for name, decision in decisions.items() if decision in keep}


def read_boxes(label_path: Path, w_px: int, h_px: int) -> list[tuple[float, float, float, float]]:
    """Bbox in pixel (xyxy) dal file di label YOLO. Il dataset di unione
    contiene già solo istanze escooter (classe rimappata a 80), quindi si
    disegnano tutte le righe valide senza filtrare sulla classe."""
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        _cls, xc, yc, w, h = parts
        xc, yc, w, h = float(xc) * w_px, float(yc) * h_px, float(w) * w_px, float(h) * h_px
        boxes.append((xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2))
    return boxes


def draw_annotated(img_path: Path, label_path: Path, dest: Path) -> int:
    with Image.open(img_path) as im:
        im = im.convert("RGB")
        w_px, h_px = im.size
        draw = ImageDraw.Draw(im)
        boxes = read_boxes(label_path, w_px, h_px)
        for box in boxes:
            draw.rectangle(box, outline=BOX_COLOR, width=BOX_WIDTH)
        dest.parent.mkdir(parents=True, exist_ok=True)
        im.save(dest)
    return len(boxes)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-d", "--data-root", type=Path, default=config.UNION_DIR,
                        help=f"Sorgente con images/ e labels/ (default: {config.UNION_DIR})")
    parser.add_argument("-o", "--out-dir", type=Path, default=config.BYDATASET_DIR,
                        help=f"Cartella di output (default: {config.BYDATASET_DIR})")
    parser.add_argument("--decisions-file", type=Path, default=None,
                        help="File JSON di review_app.py per filtrare la selezione "
                             "(default: nessun filtro, esporta tutto)")
    parser.add_argument("--include-discard", action="store_true",
                        help="Con --decisions-file, tieni anche le immagini con decisione 'discard'")
    parser.add_argument("--limit", type=int, default=None,
                        help="Esporta solo le prime N immagini (per test su campione)")
    args = parser.parse_args()

    images_dir = args.data_root / "images"
    labels_dir = args.data_root / "labels"
    if not images_dir.is_dir():
        raise SystemExit(f"Cartella immagini non trovata: {images_dir}")

    selected = load_selected_names(args.decisions_file, args.include_discard)

    images = sorted(p for p in images_dir.iterdir() if p.is_file())
    if selected is not None:
        images = [p for p in images if p.name in selected]
    if args.limit:
        images = images[:args.limit]

    if args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True)

    per_dataset_images: dict[str, int] = {}
    per_dataset_boxes: dict[str, int] = {}
    for img_path in images:
        dataset_id = img_path.stem.partition("__")[0]
        label_path = labels_dir / f"{img_path.stem}.txt"
        n_boxes = draw_annotated(img_path, label_path, args.out_dir / dataset_id / img_path.name)
        per_dataset_images[dataset_id] = per_dataset_images.get(dataset_id, 0) + 1
        per_dataset_boxes[dataset_id] = per_dataset_boxes.get(dataset_id, 0) + n_boxes

    for dataset_id in sorted(per_dataset_images):
        print(f"{dataset_id}: {per_dataset_images[dataset_id]} immagini, "
              f"{per_dataset_boxes[dataset_id]} bbox escooter")
    print(f"Totale: {len(images)} immagini in {len(per_dataset_images)} dataset -> {args.out_dir}")


if __name__ == "__main__":
    main()
