#!/usr/bin/env python3
"""Fa girare uno o più modelli sui video di una cartella e salva, per ogni
modello, i frame con le bounding box della classe escooter disegnate: un
frame ogni --step (default 10) del video sorgente, in
<out-dir>/<modello>/<video>/<video>_f<indice-frame>.jpg. Serve a confrontare
a occhio i modelli sugli stessi frame (una directory per modello, stessi nomi
file), non a misurarli: per le metriche v. eval_testset.py.

Ogni modello è dato come NOME=PESI[:INDICE_CLASSE[:IMGSZ]]: INDICE_CLASSE è
l'indice della classe escooter nel modello (0 per i monoclasse, 80 per il
multiclasse a 81 classi, 3 per multiclass_11n; default 0), IMGSZ la
risoluzione di inferenza (default 640; i modelli allenati a 960 vanno usati
a 960). Vengono disegnate solo le box di quella classe, anche per i
multiclasse, così le directory sono confrontabili. Senza --model si usano i
modelli di MODELLI_DEFAULT.

Avvio (dalla radice del repo):
    python3 scripts/training/infer_videos.py /mnt/x/media [--out-dir inferences]
        [--step 10] [--conf 0.25] [--device 0]
        [--model NOME=PESI[:INDICE_CLASSE[:IMGSZ]] ...]

Ad ogni esecuzione le directory dei modelli indicati vengono svuotate e
ricreate.
"""
import argparse
import shutil
from pathlib import Path

import cv2
from ultralytics import YOLO

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
JPEG_QUALITY = 90
BOX_COLOR = (0, 200, 0)  # BGR
RUNS = "runs/detect/data/runs"
MODELLI_DEFAULT = [
    f"escooter_only_11n={RUNS}/escooter_only_11n/weights/best.pt",
    f"escooter_only_26s={RUNS}/escooter_only_26s/weights/best.pt",
    f"baseline_26n={RUNS}/baseline_26n_escooter/weights/best.pt",
    f"imgsz960_26n={RUNS}/imgsz960_26n_escooter/weights/best.pt:0:960",
    f"multiclass81_26n={RUNS}/multiclass81_26n/weights/best.pt:80",
    f"multiclass_11n={RUNS}/multiclass_11n/weights/best.pt:3",
    "coco_escooter_yolo26n_1=models/coco_escooter_yolo26n_1/weights/best.pt:80",
    "coco_escooter_yolo11n_bt=models/coco_escooter_yolo11n_bt/weights/best.pt:80",
]


def parse_model(spec: str):
    name, _, rest = spec.partition("=")
    if not name or not rest:
        raise SystemExit(f"--model deve essere NOME=PESI[:INDICE_CLASSE[:IMGSZ]], ricevuto: {spec}")
    parts = rest.split(":")
    weights = Path(parts[0])
    class_index = int(parts[1]) if len(parts) > 1 else 0
    imgsz = int(parts[2]) if len(parts) > 2 else 640
    return name, weights, class_index, imgsz


def draw_boxes(frame, boxes) -> None:
    for (x1, y1, x2, y2), conf in boxes:
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(frame, p1, p2, BOX_COLOR, 3)
        label = f"escooter {conf:.2f}"
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        top = max(0, p1[1] - th - base - 4)
        cv2.rectangle(frame, (p1[0], top), (p1[0] + tw + 6, top + th + base + 4), BOX_COLOR, -1)
        cv2.putText(frame, label, (p1[0] + 3, top + th + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)


def process_video(model, video: Path, out_dir: Path, step: int, conf: float, class_index: int, imgsz: int,
                  device) -> int:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"  {video.name}: impossibile aprire, saltato")
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            result = model.predict(frame, conf=conf, imgsz=imgsz, classes=[class_index], device=device,
                                   verbose=False)[0]
            draw_boxes(frame, zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()))
            cv2.imwrite(str(out_dir / f"{video.stem}_f{idx:06d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            saved += 1
        idx += 1
    cap.release()
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("videos_dir", type=Path, help="Cartella con i video (ricorsiva)")
    parser.add_argument("--out-dir", type=Path, default=Path("inferences"),
                         help="Cartella di output (default: inferences, nella radice del repo)")
    parser.add_argument("--step", type=int, default=10, help="Un frame ogni N del video sorgente (default: 10)")
    parser.add_argument("--conf", type=float, default=0.25, help="Soglia di confidenza (default: 0.25)")
    parser.add_argument("--device", default=0, help="Device Ultralytics (default: 0)")
    parser.add_argument("--model", action="append", dest="models", metavar="NOME=PESI[:CLASSE[:IMGSZ]]",
                         help="Modello da usare, ripetibile (default: MODELLI_DEFAULT)")
    args = parser.parse_args()

    videos = sorted(p for p in args.videos_dir.rglob("*") if p.suffix.lower() in VIDEO_EXTENSIONS)
    if not videos:
        raise SystemExit(f"Nessun video in {args.videos_dir}")
    models = [parse_model(s) for s in (args.models or MODELLI_DEFAULT)]
    for name, weights, _, _ in models:
        if not weights.is_file():
            raise SystemExit(f"{name}: pesi non trovati: {weights}")

    print(f"{len(videos)} video, {len(models)} modelli, un frame ogni {args.step}, conf {args.conf}")
    for name, weights, class_index, imgsz in models:
        model_dir = args.out_dir / name
        if model_dir.exists():
            shutil.rmtree(model_dir)
        model = YOLO(str(weights))
        total = 0
        for video in videos:
            total += process_video(model, video, model_dir / video.stem, args.step, args.conf, class_index,
                                   imgsz, args.device)
        print(f"{name} (classe {class_index}, imgsz {imgsz}): {total} frame -> {model_dir}")


if __name__ == "__main__":
    main()
