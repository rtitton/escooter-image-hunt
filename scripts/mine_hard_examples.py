#!/usr/bin/env python3
"""Seleziona da un dataset nuovo (formato Ultralytics/YOLO già annotato) le
immagini su cui il modello attuale sbaglia, come candidate da aggiungere al
training dopo revisione manuale (hard-example mining).

Per ogni immagine con almeno un box escooter reale, fa inferenza a bassa
confidenza (MINING_CONF_LOW) e classifica ciascun box reale come:
    mancato  nessuna predizione con IoU >= MINING_IOU
    debole   predizione trovata ma con confidenza < MINING_CONF_HIT
    ok       predizione con confidenza >= MINING_CONF_HIT
Un'immagine è candidata se ha almeno un box mancato o debole. Usare i label
del dataset (invece del solo "il modello non trova nulla") lascia fuori le
immagini senza monopattino e quelle con label vuoti.

Filtri, nell'ordine:
  1. area (come select_images.py): scarta l'immagine se un box escooter è
     >= CLOSEUP_AREA_THRESHOLD (primo piano) o < FARAWAY_AREA_THRESHOLD
     (troppo piccolo/lontano). Con --no-area-filter viene saltato. Qui serve a
     non riempire le candidate di oggetti minuscoli (v. PIPELINE.md).
  2. ordinamento: più box mancati, poi più box deboli, poi confidenza minima
     più bassa.
  3. dedup pHash (distanza <= MINING_DEDUP_DISTANCE) contro il pool attuale e
     i test set esterni (--pool, default MINING_POOL_DIRS) e contro le
     candidate già tenute: una candidata quasi identica a un test set non
     deve finire nel training.
  4. taglio a --max-candidates.

Output in --out-dir (default data/processed/mined_<dataset-id>):
  images/, labels/   formato YOLO, nome "<dataset-id>__<split>__<originale>",
                     label con SOLO i box escooter, classe rimappata a
                     ESCOOTER_CLASS_ID (80): pronto per review_app.py e per gli
                     stessi passi a valle di union_reviewed (le classi COCO si
                     annotano dopo con annotate_coco_classes.py)
  mining_report.csv  tutte le immagini con box mancati/deboli e l'esito
                     (tenuta / scartata per area / quasi-duplicata di ...)

Il modello è tanto più esigente quanto più è buono: le candidate sono
difficili apposta per QUESTO modello, quindi non vanno usate per confrontare
modelli (v. eval_testset.py e i test set esterni per quello).

Ad ogni esecuzione la cartella di output viene svuotata e ripopolata, a meno
che contenga già un review_decisions.json (revisione manuale in corso): in
quel caso serve --force.

Avvio:
    python3 scripts/mine_hard_examples.py <dataset_dir> --weights best.pt \\
        --escooter-classes 1 [--splits train val test] [--device cpu] [--threads 4]
        [--out-dir DIR] [--dataset-id ID] [--max-candidates N] [--limit N]
        [--pool DIR ...] [--no-area-filter] [--force]

<dataset_dir> ha la struttura Ultralytics: images/<split>/ e labels/<split>/.
"""
import argparse
import csv
import random
import re
import resource
import shutil
from pathlib import Path

import imagehash
import numpy as np
import torch
from PIL import Image, ImageOps
from ultralytics import YOLO

import config
from eval_testset import iou_matrix
from review_app import compute_phashes

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
HASH_SUFFIX_RE = re.compile(r"-[0-9a-f]{8}$")  # suffisso "-<hash>" delle cartelle scaricate dalla piattaforma


def load_bgr(path: Path, max_side: int = 1280):
    """Immagine come array BGR con lato massimo ~max_side (>= 2x imgsz, quindi
    senza effetto sulla predizione), orientamento EXIF applicato come fanno cv2
    e Ultralytics. Image.draft scala già in decodifica JPEG: molto più veloce
    e leggero sulle foto da 12 Mpx, e le coordinate dei label sono
    normalizzate, quindi il ridimensionamento non le altera."""
    with Image.open(path) as im:
        im.draft("RGB", (max_side, max_side))
        im = ImageOps.exif_transpose(im).convert("RGB")
        if max(im.size) > max_side:
            im.thumbnail((max_side, max_side))
        return np.ascontiguousarray(np.asarray(im)[:, :, ::-1])


def dataset_id_of(dataset_dir: Path) -> str:
    return HASH_SUFFIX_RE.sub("", dataset_dir.name)


def list_images(dataset_dir: Path, splits: list[str]) -> list[tuple[str, Path]]:
    """(split, percorso immagine) per ogni immagine dei split richiesti."""
    items = []
    for split in splits:
        img_dir = dataset_dir / "images" / split
        if not img_dir.is_dir():
            print(f"  split '{split}' assente in {dataset_dir}, saltato")
            continue
        items += [(split, p) for p in sorted(img_dir.iterdir()) if p.suffix.lower() in IMAGE_EXTENSIONS]
    return items


def read_escooter_lines(label_path: Path, classes: set) -> list[list[str]]:
    """Righe [classe, xc, yc, w, h] (stringhe originali) dei soli box di classi escooter."""
    if not label_path.exists():
        return []
    rows = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) == 5 and int(float(parts[0])) in classes:
            rows.append(parts)
    return rows


def rel_to_xyxy(rows: list[list[str]], w: int, h: int) -> np.ndarray:
    boxes = []
    for _, xc, yc, bw, bh in rows:
        xc, yc, bw, bh = float(xc), float(yc), float(bw), float(bh)
        boxes.append([(xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h])
    return np.array(boxes, dtype=float).reshape(-1, 4)


def gt_status(rows: list[list[str]], result, iou_thr: float, conf_hit: float) -> dict:
    """Per ogni box reale, la confidenza della migliore predizione con IoU >=
    iou_thr (0 se nessuna). Ritorna i conteggi mancato/debole/ok e la
    confidenza minima fra i box reali."""
    h, w = result.orig_shape
    gt = rel_to_xyxy(rows, w, h)
    if len(result.boxes):
        pred = result.boxes.xyxy.cpu().numpy()
        conf = result.boxes.conf.cpu().numpy()
        ious = iou_matrix(gt, pred)  # (n_gt, n_pred)
        best = np.array([conf[ious[i] >= iou_thr].max() if (ious[i] >= iou_thr).any() else 0.0
                         for i in range(len(gt))])
    else:
        best = np.zeros(len(gt))
    return {
        "n_gt": len(gt),
        "n_missed": int((best == 0).sum()),
        "n_weak": int(((best > 0) & (best < conf_hit)).sum()),
        "n_ok": int((best >= conf_hit).sum()),
        "min_conf": float(best.min()),
    }


def area_rejected(rows: list[list[str]], min_area: float, max_area: float) -> bool:
    return any(float(w) * float(h) >= max_area or float(w) * float(h) < min_area for _, _, _, w, h in rows)


def pool_hashes(pool_dirs: list[Path]) -> list[tuple[int, str]]:
    """(pHash come intero, "<cartella>/<nome>") di ogni immagine dei pool, con cache su disco."""
    hashes = []
    for pool in pool_dirs:
        if not (pool / "images").is_dir():
            print(f"  pool {pool} senza sottocartella images/, saltato")
            continue
        cache = config.DATA_ROOT / "cache" / f"mining_phash_{pool.parent.name}_{pool.name}.json"
        cache.parent.mkdir(parents=True, exist_ok=True)
        for name, h in compute_phashes(pool, cache).items():
            hashes.append((int(h, 16), f"{pool.name}/{name}"))
    return hashes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset_dir", type=Path)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--escooter-classes", type=int, nargs="+", required=True,
                     help="indici delle classi che nel dataset sono monopattini (es. 1)")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--dataset-id", default=None, help="prefisso dei nomi in uscita (default: nome cartella senza -<hash>)")
    ap.add_argument("--max-candidates", type=int, default=config.MINING_MAX_CANDIDATES)
    ap.add_argument("--conf-low", type=float, default=config.MINING_CONF_LOW)
    ap.add_argument("--conf-hit", type=float, default=config.MINING_CONF_HIT)
    ap.add_argument("--iou", type=float, default=config.MINING_IOU)
    ap.add_argument("--min-area", type=float, default=config.FARAWAY_AREA_THRESHOLD)
    ap.add_argument("--max-area", type=float, default=config.CLOSEUP_AREA_THRESHOLD)
    ap.add_argument("--no-area-filter", action="store_true")
    ap.add_argument("--dedup-distance", type=int, default=config.MINING_DEDUP_DISTANCE)
    ap.add_argument("--pool", type=Path, nargs="*", default=None,
                     help=f"cartelle (con images/) contro cui deduplicare (default: {[str(p) for p in config.MINING_POOL_DIRS]})")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=4,
                     help="immagini per batch di inferenza (default 4: le immagini di partenza possono essere "
                          "di 4032x3024 e ogni batch le tiene in memoria a piena risoluzione)")
    ap.add_argument("--device", default="cpu", help="default cpu, per non contendere la GPU a un training in corso")
    ap.add_argument("--threads", type=int, default=4, help="thread CPU di torch (default 4, per lasciare spazio ad altri job)")
    ap.add_argument("--limit", type=int, default=None, help="elabora solo N immagini a caso (per prove rapide)")
    ap.add_argument("--force", action="store_true", help="svuota l'output anche se contiene review_decisions.json")
    args = ap.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    dataset_id = args.dataset_id or dataset_id_of(dataset_dir)
    out_dir = (args.out_dir or config.DATA_ROOT / "processed" / f"mined_{dataset_id}").resolve()
    if (out_dir / "review_decisions.json").exists() and not args.force:
        raise SystemExit(f"{out_dir} contiene review_decisions.json (revisione in corso): usa --force per sovrascrivere")
    classes = set(args.escooter_classes)
    torch.set_num_threads(args.threads)

    items = list_images(dataset_dir, args.splits)
    if args.limit and args.limit < len(items):
        items = random.Random(0).sample(items, args.limit)
    print(f"{dataset_id}: {len(items)} immagini da elaborare (classi escooter {sorted(classes)})")

    # solo le immagini con almeno un box escooter reale vanno in inferenza
    gt_rows, no_gt = {}, 0
    for split, path in items:
        rows = read_escooter_lines(dataset_dir / "labels" / split / f"{path.stem}.txt", classes)
        if rows:
            gt_rows[path] = (split, rows)
        else:
            no_gt += 1
    print(f"  {len(gt_rows)} con box escooter, {no_gt} senza (ignorate)")

    model = YOLO(str(args.weights))
    records = []
    n_boxes = {"ok": 0, "weak": 0, "missed": 0}
    paths = list(gt_rows)
    n_unreadable = 0
    for start in range(0, len(paths), args.batch):
        chunk, arrays = [], []
        for path in paths[start:start + args.batch]:
            try:
                arrays.append(load_bgr(path))
                chunk.append(path)
            except Exception as e:  # immagine corrotta: la saltiamo, senza fermare il resto
                print(f"  illeggibile, saltata: {path.name} ({e})")
                n_unreadable += 1
        if not chunk:
            continue
        results = model.predict(arrays, conf=args.conf_low, imgsz=args.imgsz, device=args.device, verbose=False)
        for path, r in zip(chunk, results):
            split, rows = gt_rows[path]
            st = gt_status(rows, r, args.iou, args.conf_hit)
            n_boxes["ok"] += st["n_ok"]; n_boxes["weak"] += st["n_weak"]; n_boxes["missed"] += st["n_missed"]
            if st["n_missed"] + st["n_weak"]:
                records.append({"split": split, "path": path, "rows": rows, **st, "status": ""})
        done = start + len(chunk)
        if done // 200 != (done - len(chunk)) // 200:
            print(f"  {done}/{len(paths)} immagini, {len(records)} con box mancati/deboli")

    total_boxes = sum(n_boxes.values())
    print(f"\nBox reali: {total_boxes} — ok {n_boxes['ok']}, deboli {n_boxes['weak']}, mancati {n_boxes['missed']} "
          f"(recall a conf>={args.conf_hit}: {n_boxes['ok'] / max(total_boxes, 1):.3f})")
    print(f"Immagini con almeno un box mancato o debole: {len(records)}/{len(paths)}")

    kept, n_area, n_dup_pool, n_dup_cand = [], 0, 0, 0
    if not args.no_area_filter:
        for rec in records:
            if area_rejected(rec["rows"], args.min_area, args.max_area):
                rec["status"] = "scartata: area"
                n_area += 1
    remaining = [rec for rec in records if not rec["status"]]
    remaining.sort(key=lambda rec: (-rec["n_missed"], -rec["n_weak"], rec["min_conf"]))

    pool = pool_hashes(args.pool if args.pool is not None else config.MINING_POOL_DIRS)
    print(f"Pool di dedup: {len(pool)} immagini; candidate dopo il filtro area: {len(remaining)}")
    kept_hashes = []
    for rec in remaining:
        with Image.open(rec["path"]) as im:
            im.draft("RGB", (512, 512))  # il pHash lavora comunque su 32x32: inutile decodificare a piena risoluzione
            h = int(str(imagehash.phash(im)), 16)
        near_pool = min(((h ^ ph).bit_count(), name) for ph, name in pool) if pool else (65, "")
        if near_pool[0] <= args.dedup_distance:
            rec["status"] = f"scartata: quasi-duplicata di {near_pool[1]} (distanza {near_pool[0]})"
            n_dup_pool += 1
            continue
        if any((h ^ kh).bit_count() <= args.dedup_distance for kh in kept_hashes):
            rec["status"] = "scartata: quasi-duplicata di una candidata migliore"
            n_dup_cand += 1
            continue
        if len(kept) >= args.max_candidates:
            rec["status"] = "scartata: oltre --max-candidates"
            continue
        rec["status"] = "tenuta"
        kept.append(rec)
        kept_hashes.append(h)

    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "images").mkdir(parents=True)
    (out_dir / "labels").mkdir(parents=True)
    for rec in kept:
        new_name = f"{dataset_id}__{rec['split']}__{rec['path'].name}"
        shutil.copy2(rec["path"], out_dir / "images" / new_name)
        lines = [f"{config.ESCOOTER_CLASS_ID} {xc} {yc} {w} {h}" for _, xc, yc, w, h in rec["rows"]]
        (out_dir / "labels" / f"{Path(new_name).stem}.txt").write_text("\n".join(lines) + "\n")
        rec["out_name"] = new_name

    with open(out_dir / "mining_report.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["immagine", "split", "n_gt", "n_mancati", "n_deboli", "conf_min", "esito", "nome_in_uscita"])
        for rec in sorted(records, key=lambda rec: (rec["status"] != "tenuta", -rec["n_missed"], -rec["n_weak"])):
            wr.writerow([rec["path"].name, rec["split"], rec["n_gt"], rec["n_missed"], rec["n_weak"],
                         f"{rec['min_conf']:.3f}", rec["status"], rec.get("out_name", "")])

    print(f"\nScartate per area: {n_area}; quasi-duplicate del pool/test set: {n_dup_pool}; "
          f"quasi-duplicate tra candidate: {n_dup_cand}")
    print(f"Candidate tenute: {len(kept)} -> {out_dir}")
    print(f"Report: {out_dir / 'mining_report.csv'}")
    print(f"Picco di memoria (RSS): {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MB")


if __name__ == "__main__":
    main()
