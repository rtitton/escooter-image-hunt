#!/usr/bin/env python3
"""Come infer_videos.py, ma su un test set di immagini (cartella con images/ e,
facoltativamente, labels/ in formato YOLO): per ogni modello salva le immagini
con le box predette (verdi) e, se --gt, le box reali (rosse, sottili) della
classe escooter (--gt-class, default 80 come nei test set *_final), in
<out-dir>/<modello>/<nome-test-set>/<immagine>.jpg.

La struttura è la stessa di infer_videos.py, quindi infer_contact_sheets.py
affianca i modelli senza modifiche (--src <out-dir>). Modelli come in
infer_videos.py: NOME=PESI[:INDICE_CLASSE[:IMGSZ]].

Avvio (dalla radice del repo):
    python3 scripts/training/infer_images.py data/processed/video_testset_final --out-dir tmp/inferences_testset \\
        --model NOME=PESI[:INDICE_CLASSE[:IMGSZ]] ... [--conf 0.25] [--device 0] [--gt]

Ad ogni esecuzione le directory dei modelli indicati vengono svuotate e ricreate.
"""
import argparse
import shutil
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent))
from infer_videos import JPEG_QUALITY, draw_boxes, parse_model  # noqa: E402

GT_COLOR = (0, 0, 255)  # BGR, rosso
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def draw_gt(img, label_file: Path, gt_class: int) -> None:
    if not label_file.is_file():
        return
    h, w = img.shape[:2]
    for line in label_file.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5 or int(parts[0]) != gt_class:
            continue
        cx, cy, bw, bh = (float(v) for v in parts[1:5])
        p1 = (round((cx - bw / 2) * w), round((cy - bh / 2) * h))
        p2 = (round((cx + bw / 2) * w), round((cy + bh / 2) * h))
        cv2.rectangle(img, p1, p2, GT_COLOR, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("testset", type=Path, help="Cartella con images/ (e labels/)")
    ap.add_argument("--out-dir", type=Path, default=Path("tmp/inferences_testset"))
    ap.add_argument("--model", action="append", dest="models", required=True, metavar="NOME=PESI[:CLASSE[:IMGSZ]]")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default=0)
    ap.add_argument("--gt", action="store_true", help="Disegna anche le box reali (rosse)")
    ap.add_argument("--gt-class", type=int, default=80)
    args = ap.parse_args()

    images = sorted(p for p in (args.testset / "images").iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise SystemExit(f"Nessuna immagine in {args.testset / 'images'}")
    models = [parse_model(s) for s in args.models]
    for name, weights, _, _ in models:
        if not weights.is_file():
            raise SystemExit(f"{name}: pesi non trovati: {weights}")

    for name, weights, class_index, imgsz in models:
        model_dir = args.out_dir / name
        if model_dir.exists():
            shutil.rmtree(model_dir)
        out = model_dir / args.testset.name
        out.mkdir(parents=True)
        model = YOLO(str(weights))
        for p in images:
            img = cv2.imread(str(p))
            res = model.predict(img, conf=args.conf, imgsz=imgsz, classes=[class_index], device=args.device,
                                verbose=False)[0]
            if args.gt:
                draw_gt(img, args.testset / "labels" / f"{p.stem}.txt", args.gt_class)
            draw_boxes(img, zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()))
            cv2.imwrite(str(out / f"{p.stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        print(f"{name} (classe {class_index}, imgsz {imgsz}): {len(images)} immagini -> {out}")


if __name__ == "__main__":
    main()
