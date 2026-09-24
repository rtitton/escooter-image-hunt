#!/usr/bin/env python3
"""Valida uno o più pesi YOLO monoclasse (classe 80, escooter, rimappata a 0
in training da --classes) su uno o più test set esterni, ciascuno una
cartella con sottocartelle images/ e labels/ (label in classe 80, come
prodotti da review_app.py + materialize_union_reviewed.py).

Per ogni test set costruisce in EVAL_WORKDIR (config.py) una copia di
lavoro con i label rimappati 80 -> 0 (le immagini sono linkate, non
copiate) e un data.yaml a singola classe, poi lancia model.val() di
Ultralytics e stampa una tabella comparativa (precision, recall, mAP50,
mAP50-95) modello x test set. Salva anche il report completo in JSON.

In aggiunta (--size-breakdown, di default attivo) rifà una passata di
inferenza a conf --size-conf con matching IoU>=--size-iou fatto a mano
(niente pycocotools, non è nel progetto), bucket small/medium/large sulle
soglie COCO (area in pixel <32^2, <96^2, >=96^2):
  - recall per taglia: dei box reali di quella taglia, quanti sono stati
    trovati — risponde a "il modello perde soprattutto i piccoli?"
  - falsi positivi per taglia (della bbox predetta, non del box reale che
    manca): utile per capire se i falsi positivi (es. persone scambiate
    per escooter) si concentrano tra le predizioni piccole.
Non è una vera AP per taglia in stile COCOeval (richiederebbe integrare su
tutte le soglie di confidenza): è calcolato a una sola soglia di
confidenza, pensato per una lettura rapida, non per un numero definitivo.

Ad ogni esecuzione le sottocartelle di EVAL_WORKDIR usate vengono svuotate
e ricreate.

Avvio:
    python3 scripts/eval_testset.py \
        --weights runs/detect/data/runs/escooter_only_11n/weights/best.pt \
        --testset data/processed/video_testset_final data/processed/holdout_kickboard_a75qx_final \
        [--device cpu] [--imgsz 640] [--out data/tmp/eval_report.json]
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from ultralytics import YOLO

import config

ESCOOTER_CLASS_ID = config.ESCOOTER_CLASS_ID

# soglie COCO per l'area (in pixel^2) dei bucket small/medium/large
SIZE_BUCKETS = [("small", 0, 32 * 32), ("medium", 32 * 32, 96 * 96), ("large", 96 * 96, float("inf"))]


def size_bucket(area_px: float) -> str:
    for name, lo, hi in SIZE_BUCKETS:
        if lo <= area_px < hi:
            return name
    return SIZE_BUCKETS[-1][0]


def yolo_boxes_to_xyxy(lines: list[str], img_w: int, img_h: int) -> np.ndarray:
    """Righe "classe xc yc w h" normalizzate -> array Nx4 xyxy in pixel."""
    boxes = []
    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            continue
        _, xc, yc, w, h = (float(v) for v in parts)
        boxes.append([(xc - w / 2) * img_w, (yc - h / 2) * img_h, (xc + w / 2) * img_w, (yc + h / 2) * img_h])
    return np.array(boxes, dtype=float).reshape(-1, 4)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU tra ogni riga di a (Na,4) ed ogni riga di b (Nb,4), formato xyxy."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ix1, iy1 = np.maximum(ax1, bx1), np.maximum(ay1, by1)
    ix2, iy2 = np.minimum(ax2, bx2), np.minimum(ay2, by2)
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = ((ax2 - ax1) * (ay2 - ay1))
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return np.where(union > 0, inter / union, 0.0)


def size_breakdown(model: YOLO, images_dir: Path, labels_dir: Path, imgsz: int, device: str,
                    conf: float, iou_thr: float) -> dict:
    """Recall per taglia dei box reali (small/medium/large) e conteggio dei
    falsi positivi per taglia della bbox predetta, con matching IoU greedy
    (predizioni in ordine di confidenza decrescente). V. docstring del
    modulo per i limiti di questo calcolo rispetto a una vera AP per
    taglia."""
    images = sorted(images_dir.iterdir())
    recall_counts = {name: {"tp": 0, "fn": 0} for name, _, _ in SIZE_BUCKETS}
    fp_counts = {name: 0 for name, _, _ in SIZE_BUCKETS}
    for i in range(0, len(images), 16):
        batch = images[i:i + 16]
        results = model.predict([str(p) for p in batch], conf=conf, imgsz=imgsz, device=device, verbose=False)
        for img_path, res in zip(batch, results):
            img_w, img_h = Image.open(img_path).size
            lbl_path = labels_dir / f"{img_path.stem}.txt"
            gt_lines = lbl_path.read_text().splitlines() if lbl_path.exists() else []
            gt = yolo_boxes_to_xyxy(gt_lines, img_w, img_h)
            pred_xyxy = res.boxes.xyxy.cpu().numpy() if len(res.boxes) else np.zeros((0, 4))
            pred_conf = res.boxes.conf.cpu().numpy() if len(res.boxes) else np.zeros((0,))
            order = np.argsort(-pred_conf)
            ious = iou_matrix(pred_xyxy, gt)
            gt_matched = np.zeros(len(gt), dtype=bool)
            for pi in order:
                # tra i box reali ancora liberi (uno già assegnato a una predizione più
                # confidente non è più disponibile, anche se fosse il migliore in IoU)
                free = np.where(~gt_matched)[0]
                if len(free):
                    gj = free[ious[pi, free].argmax()]
                    if ious[pi, gj] >= iou_thr:
                        gt_matched[gj] = True
                        continue
                pred_area = (pred_xyxy[pi, 2] - pred_xyxy[pi, 0]) * (pred_xyxy[pi, 3] - pred_xyxy[pi, 1])
                fp_counts[size_bucket(pred_area)] += 1
            for gj, box in enumerate(gt):
                area = (box[2] - box[0]) * (box[3] - box[1])
                bucket = size_bucket(area)
                recall_counts[bucket]["tp" if gt_matched[gj] else "fn"] += 1
    return {"recall_by_gt_size": recall_counts, "false_positives_by_pred_size": fp_counts}


def prepare_eval_dataset(testset_dir: Path, work_dir: Path) -> Path:
    """Rimappa i label di testset_dir (classe ESCOOTER_CLASS_ID) a classe 0
    dentro work_dir, linkando le immagini invece di copiarle, e scrive un
    data.yaml a singola classe. Ritorna il path del data.yaml."""
    if work_dir.exists():
        shutil.rmtree(work_dir)
    images_out = work_dir / "images"
    labels_out = work_dir / "labels"
    labels_out.mkdir(parents=True)
    images_out.mkdir(parents=True)
    # symlink dei singoli file (non della cartella: Ultralytics risolve i symlink di
    # directory in check_det_dataset, e la sostituzione images/->labels/ ricadrebbe sui
    # label originali in classe 80 invece che su quelli rimappati qui sotto)
    for img in (testset_dir / "images").iterdir():
        (images_out / img.name).symlink_to(img.resolve())

    n_boxes = 0
    n_other_class = 0
    for lbl in sorted((testset_dir / "labels").glob("*.txt")):
        lines_out = []
        for line in lbl.read_text().splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            cls = int(float(parts[0]))
            if cls != ESCOOTER_CLASS_ID:
                n_other_class += 1
                continue
            lines_out.append("0 " + " ".join(parts[1:]))
            n_boxes += 1
        (labels_out / lbl.name).write_text("\n".join(lines_out) + ("\n" if lines_out else ""))
    if n_other_class:
        print(f"  ATTENZIONE: {n_other_class} box di classi diverse da {ESCOOTER_CLASS_ID} ignorate in {testset_dir}")

    data_yaml = work_dir / "data.yaml"
    data_yaml.write_text(yaml.safe_dump({
        "path": str(work_dir.resolve()),
        "train": "images",  # non usato in validazione, ma richiesto dallo schema data.yaml
        "val": "images",
        "nc": 1,
        "names": {0: "escooter"},
    }, sort_keys=False))

    n_images = sum(1 for _ in images_out.iterdir())
    print(f"  {testset_dir.name}: {n_images} immagini, {n_boxes} box escooter -> {data_yaml}")
    return data_yaml


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=Path, nargs="+", required=True, help="uno o più file .pt da valutare")
    ap.add_argument("--testset", type=Path, nargs="+", required=True,
                     help="una o più cartelle test set (images/ + labels/, label in classe 80)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="cpu", help="default cpu per non contendere la GPU con eventuali training in corso")
    ap.add_argument("--work-dir", type=Path, default=config.EVAL_WORKDIR)
    ap.add_argument("--out", type=Path, default=None, help="report JSON completo (default: <work-dir>/report.json)")
    ap.add_argument("--size-breakdown", action=argparse.BooleanOptionalAction, default=True,
                     help="scomposizione recall/falsi-positivi per taglia small/medium/large (default: attiva)")
    ap.add_argument("--size-conf", type=float, default=0.25, help="soglia di confidenza per --size-breakdown")
    ap.add_argument("--size-iou", type=float, default=0.5, help="soglia IoU di match per --size-breakdown")
    args = ap.parse_args()

    print("Preparazione test set:")
    yamls = {}
    for ts in args.testset:
        yamls[ts] = prepare_eval_dataset(ts, args.work_dir / ts.name)

    report = {}
    rows = []
    for w in args.weights:
        model = YOLO(str(w))
        for ts, data_yaml in yamls.items():
            print(f"\n== {w.parent.parent.name} su {ts.name} ==")
            metrics = model.val(data=str(data_yaml), imgsz=args.imgsz, device=args.device,
                                 split="val", plots=False, save_json=False, verbose=False)
            row = {
                "weights": str(w),
                "testset": ts.name,
                "precision": round(float(metrics.box.mp), 4),
                "recall": round(float(metrics.box.mr), 4),
                "mAP50": round(float(metrics.box.map50), 4),
                "mAP50-95": round(float(metrics.box.map), 4),
            }
            if args.size_breakdown:
                images_dir = data_yaml.parent / "images"
                labels_dir = data_yaml.parent / "labels"
                sb = size_breakdown(model, images_dir, labels_dir, args.imgsz, args.device,
                                     args.size_conf, args.size_iou)
                row["size_breakdown"] = sb
            rows.append(row)
            report.setdefault(str(w), {})[ts.name] = row

    header = f"{'modello':30s} {'test set':30s} {'P':>7s} {'R':>7s} {'mAP50':>7s} {'mAP50-95':>9s}"
    print("\n" + header)
    print("-" * len(header))
    for r in rows:
        print(f"{Path(r['weights']).parent.parent.name:30s} {r['testset']:30s} "
              f"{r['precision']:7.3f} {r['recall']:7.3f} {r['mAP50']:7.3f} {r['mAP50-95']:9.3f}")

    if args.size_breakdown:
        print(f"\nScomposizione per taglia (conf>={args.size_conf}, IoU>={args.size_iou}; "
              "small <32²px, medium <96²px, large oltre):")
        header2 = f"{'modello':30s} {'test set':30s} {'taglia':8s} {'n_gt':>6s} {'recall':>8s} {'FP':>5s}"
        print(header2)
        print("-" * len(header2))
        for r in rows:
            sb = r["size_breakdown"]
            for name, _, _ in SIZE_BUCKETS:
                c = sb["recall_by_gt_size"][name]
                n_gt = c["tp"] + c["fn"]
                recall_s = c["tp"] / n_gt if n_gt else float("nan")
                fp = sb["false_positives_by_pred_size"][name]
                recall_str = f"{recall_s:8.3f}" if n_gt else f"{'--':>8s}"
                print(f"{Path(r['weights']).parent.parent.name:30s} {r['testset']:30s} {name:8s} "
                      f"{n_gt:6d} {recall_str} {fp:5d}")

    out = args.out or (args.work_dir / "report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(f"\nReport completo: {out}")


if __name__ == "__main__":
    main()
