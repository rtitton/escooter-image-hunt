#!/usr/bin/env python3
"""Pre-annota con un modello monoclasse (default: la baseline) uno o più
insiemi di sole immagini, per prepararli alla review manuale con
review_app.py. Costruisce un dataset YOLO (images/ + labels/) con le
immagini COPIATE (nome file invariato) e, per ognuna, le box della classe
escooter scritte con ESCOOTER_CLASS_ID (80) dal modello alla soglia --conf.

La soglia è bassa di proposito (default 0.2) e l'inferenza è a imgsz 960
(default): la baseline a 960 ritrova più oggetti piccoli, e in una review
manuale togliere un falso positivo costa meno che aggiungere un box mancato.

Dopo questo passo si aggiungono le classi COCO con annotate_coco_classes.py
(--source-dir = la cartella di output di questo script).

Avvio (dalla radice del repo):
    python3 scripts/dataset/preannotate_escooter.py IMMAGINI_DIR [IMMAGINI_DIR ...] --out-dir DIR
        [--weights PESI] [--imgsz 960] [--conf 0.2] [--device 0]

Ogni IMMAGINI_DIR è una cartella di immagini (o con sottocartella images/).
Ad ogni esecuzione la cartella di output viene svuotata e ripopolata.
"""
import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

import os
import sys
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")))  # config.py sta in scripts/
import config

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image_dirs", type=Path, nargs="+")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--weights", type=Path, default=Path("models/baseline_26n_escooter/weights/best.pt"))
    ap.add_argument("--class-index", type=int, default=0, help="Indice della classe escooter nel modello (default 0)")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--conf", type=float, default=0.2)
    ap.add_argument("--device", default=0)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    images = []
    for d in args.image_dirs:
        src = d / "images" if (d / "images").is_dir() else d
        images += sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise SystemExit("nessuna immagine trovata")
    names = [p.name for p in images]
    if len(set(names)) != len(names):
        raise SystemExit("nomi file duplicati fra le cartelle di input")

    if args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    out_images, out_labels = args.out_dir / "images", args.out_dir / "labels"
    out_images.mkdir(parents=True)
    out_labels.mkdir()

    model = YOLO(str(args.weights))
    n_boxes = n_with = 0
    for i in range(0, len(images), args.batch):
        chunk = images[i:i + args.batch]
        results = model.predict([str(p) for p in chunk], conf=args.conf, imgsz=args.imgsz,
                                classes=[args.class_index], device=args.device, verbose=False)
        for p, res in zip(chunk, results):
            shutil.copy2(p, out_images / p.name)
            lines = []
            for (cx, cy, w, h) in res.boxes.xywhn.cpu().numpy():
                lines.append(f"{config.ESCOOTER_CLASS_ID} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            (out_labels / f"{p.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            n_boxes += len(lines)
            n_with += bool(lines)
    print(f"{len(images)} immagini, {n_with} con almeno una box, {n_boxes} box escooter totali -> {args.out_dir}")


if __name__ == "__main__":
    main()
