#!/usr/bin/env python3
"""Confronta due modelli sulle 80 classi di COCO val2017.

Salva metriche Ultralytics (non COCOeval ufficiale) e differenze per classe
in JSON e CSV. Precision/recall standard usano la soglia F1 del singolo
modello; le colonne *_at_conf usano la stessa confidence per entrambi,
interpolando le curve Ultralytics a IoU 0.5.

Le predizioni delle classi aggiuntive restano attive, ma non entrano nella
media COCO. Non viene modificato il dataset sorgente.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import ultralytics
import yaml
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--trained", type=Path, required=True)
    parser.add_argument("--coco-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--end2end", choices=("true", "false"), default=None,
                        help="Forza la modalità end-to-end (true) o NMS (false) su entrambi i modelli. Default: "
                             "quella dei pesi, che deve coincidere. Serve con un modello allenato in modalità NMS "
                             "(end2end=False): il baseline va valutato in NMS, non end-to-end")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence comune per le colonne *_at_conf")
    args = parser.parse_args()
    if not 0 < args.conf < 1:
        parser.error("--conf deve essere compreso fra 0 e 1")
    for weights in (args.baseline, args.trained):
        if not weights.is_file():
            parser.error(f"Pesi non trovati: {weights}")
    coco = args.coco_dir.resolve()
    images = coco / "images/val2017"
    labels = coco / "labels/val2017"
    image_files = sorted(images.glob("*.jpg"))
    if len(image_files) != 5000 or not labels.is_dir():
        parser.error("Servono tutte le 5000 immagini COCO val2017 e labels/val2017")
    missing = [p.name for p in image_files if not (labels / f"{p.stem}.txt").is_file()]
    if missing:
        annotations = coco / "annotations/instances_val2017.json"
        if not annotations.is_file():
            parser.error(f"Label mancanti per {len(missing)} immagini; serve {annotations} per verificarle")
        annotated_ids = {item["image_id"] for item in json.loads(annotations.read_text())["annotations"]}
        missing_annotated = [name for name in missing if int(Path(name).stem) in annotated_ids]
        if missing_annotated:
            parser.error(f"Label mancanti per immagini annotate: {missing_annotated[0]}")
        print(f"{len(missing)} immagini senza label: verificate prive di annotazioni COCO.", flush=True)

    baseline = YOLO(str(args.baseline.resolve()))
    trained = YOLO(str(args.trained.resolve()))
    names = baseline.names
    if set(names) != set(range(80)):
        parser.error("Il modello baseline deve avere le 80 classi COCO")
    if any(trained.names.get(i) != names[i] for i in range(80)):
        parser.error("Le prime 80 classi del modello allenato non coincidono con la baseline")
    if args.end2end is None and baseline.model.end2end != trained.model.end2end:
        parser.error("I due modelli devono usare la stessa modalita end-to-end/NMS (oppure --end2end true|false)")
    end2end = baseline.model.end2end if args.end2end is None else args.end2end == "true"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    data_yaml = args.out_dir / "coco_val.yaml"
    # Directory, non val2017.txt: evita COCOeval automatico, incompatibile
    # con la classe 80 aggiuntiva nella mappatura COCO80 -> COCO91.
    data_yaml.write_text(yaml.safe_dump({
        "path": str(coco), "train": "images/train2017",
        "val": "images/val2017", "names": names,
    }))
    settings = dict(
        data=str(data_yaml.resolve()), imgsz=args.imgsz, device=args.device,
        batch=args.batch, workers=args.workers, conf=0.001, iou=0.7,
        max_det=300, half=False, rect=True, augment=False, plots=False,
        save_json=False, verbose=False, project=str(args.out_dir.resolve()),
    )
    if args.end2end is not None:
        settings["end2end"] = end2end
    report = {
        "ultralytics_version": ultralytics.__version__,
        "dataset": str(images), "images": len(image_files),
        "metric_backend": "Ultralytics native, not official COCOeval",
        "end2end": end2end, "fixed_confidence": args.conf,
        "settings": settings, "models": {},
    }
    metric_keys = ("precision", "recall", "AP50", "AP50_95",
                   "precision_at_conf", "recall_at_conf")
    for tag, model, weights in (
        ("baseline", baseline, args.baseline), ("trained", trained, args.trained),
    ):
        metrics = model.val(**settings, name=tag)
        box = metrics.box
        index = {int(cls): i for i, cls in enumerate(box.ap_class_index)}
        if set(index) != set(range(80)):
            raise RuntimeError(f"{tag}: metriche mancanti o inattese per COCO: {sorted(index)}")
        rows = []
        for cls in range(80):
            i = index[cls]
            p, r, ap50, ap = box.class_result(i)
            rows.append(dict(
                class_id=cls, name=names[cls], instances=int(metrics.nt_per_class[cls]),
                precision=float(p), recall=float(r), AP50=float(ap50), AP50_95=float(ap),
                precision_at_conf=float(np.interp(args.conf, box.px, box.p_curve[i])),
                recall_at_conf=float(np.interp(args.conf, box.px, box.r_curve[i])),
            ))
        report["models"][tag] = {
            "weights": str(weights.resolve()), "classes": rows,
            "mean": {key: float(np.mean([row[key] for row in rows])) for key in metric_keys},
        }
        (args.out_dir / f"{tag}.json").write_text(json.dumps(report["models"][tag], indent=2) + "\n")

    comparison = []
    for original, updated in zip(report["models"]["baseline"]["classes"],
                                 report["models"]["trained"]["classes"]):
        row = {key: original[key] for key in ("class_id", "name", "instances")}
        for key in metric_keys:
            row[f"baseline_{key}"] = original[key]
            row[f"trained_{key}"] = updated[key]
            row[f"delta_{key}_pp"] = 100 * (updated[key] - original[key])
        comparison.append(row)
    report["comparison"] = comparison
    report["mean_delta_pp"] = {
        key: 100 * (report["models"]["trained"]["mean"][key] -
                    report["models"]["baseline"]["mean"][key])
        for key in metric_keys
    }
    with (args.out_dir / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison[0]))
        writer.writeheader()
        writer.writerows(comparison)
    (args.out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("\nMedie COCO e differenze in punti percentuali:")
    for key in metric_keys:
        print(f"{key}: {report['models']['baseline']['mean'][key]:.4f} -> "
              f"{report['models']['trained']['mean'][key]:.4f} "
              f"({report['mean_delta_pp'][key]:+.2f} pp)")
    print("\nClassi con le maggiori regressioni AP50-95:")
    for row in sorted(comparison, key=lambda row: row["delta_AP50_95_pp"])[:10]:
        print(f"{row['name']:20s} {row['delta_AP50_95_pp']:+.2f} pp")
    print(f"\nReport: {args.out_dir / 'report.json'}")


if __name__ == "__main__":
    main()
