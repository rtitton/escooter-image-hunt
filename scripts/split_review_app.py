#!/usr/bin/env python3
"""Applicazione locale per revisionare a mano lo split train/valid di un
dataset YOLO già diviso da split_dataset.py (sottocartelle train/images,
train/labels, valid/images, valid/labels), e spostare immagini da uno
split all'altro con un clic — ogni spostamento sposta subito anche il
file label corrispondente e viene applicato sul dataset reale (non c'è
salvataggio separato: per tornare indietro basta spostare l'immagine di
nuovo dal lato opposto).

Nata per un problema specifico: con centinaia/migliaia di immagini, una
revisione manuale completa non è pensabile, e lo scopo principale non è
"guardarle tutte" ma trovare i casi peggiori di somiglianza fra i due
split (le foto sono spesso duplicate o quasi-duplicate fra dataset
Roboflow diversi, v. dedupe_phash.py — un quasi-duplicato finito per caso
uno in train e uno in valid gonfia le metriche di validazione). Quattro modalità:

  - "Coppie sospette": per ogni immagine di valid, tutte le immagini di
    train entro la soglia di distanza pHash (Hamming; vale la minore fra
    immagine normale e specchiata) o con lo stesso nome-base (stessa foto in
    dataset diversi, copie "_flip"), raggruppate in una
    riga e ordinate dalla più sospetta: un cluster di copie si risolve in
    una passata. Bottoni per spostare o escludere ciascuna, o per ignorare
    la coppia (non è un duplicato) così non ricompare.
  - "Copie sospette yolodet": stessa idea ma basata sulle detection
    ORIGINALI invece del pHash (v. yolodet_groups): due immagini sono
    candidate se hanno lo stesso insieme di classi nelle box originali
    (backup di review_app.py in --source-dir, o il label corrente se non
    modificato) e almeno N box si sovrappongono con IoU >= 1 - T, a parità di
    risoluzione (box normalizzate), anche con l'immagine di valid specchiata.
    Per ogni classe conta la box migliore (confidence più alta dalla cache
    --detection-cache, altrimenti l'area maggiore); "tutte le box" conta
    invece tutte quelle dell'immagine di valid. N e T si cambiano e si
    ricalcolano come la soglia pHash.
  - "Statistiche": istanze e dimensione dell'escooter in train e valid
    (istanze per immagine, area e lato lungo con percentili, fasce COCO
    piccola/media/grande, dettaglio per dataset), sui label attuali: si
    aggiorna dopo ogni spostamento o esclusione.
  - "Sfoglia": le due cartelle affiancate, poche miniature alla volta
    (default 4 per lato, configurabile), filtrabili per dataset sorgente
    (prefisso "<dataset_id>__" nel nome file) — per una scorsa manuale
    mirata, non esaustiva, quando si vuole ribilanciare a occhio.

Su ogni miniatura sono disegnate le bbox (escooter in rosso, person in
verde, altre classi COCO in blu).

Azioni per immagine (nelle coppie sospette e in Sfoglia):
  - sposta: passa dall'altro split (immagine + label);
  - escludi: sposta immagine + label in <dataset_dir>/excluded/<split>/,
    fuori dal dataset ma senza cancellarle. Sui veri duplicati è la mossa più
    pulita (eliminando la copia in train non si crea nessun duplicato
    interno né a train né a valid);
  - "Annulla ultima azione": ripristina l'ultimo spostamento o l'ultima
    esclusione registrata nel log.

Avvio:
    python3 scripts/split_review_app.py <dataset_dir> [--port 8767]
        [--source-dir DIR] [--detection-cache FILE]

<dataset_dir> deve contenere le sottocartelle train/{images,labels} e
valid/{images,labels} (es. data/processed/union_reviewed_coco_split).
Poi apri http://localhost:<port> nel browser.
"""
import argparse
import json
import mimetypes
import shutil
import time
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import imagehash
import numpy as np
from PIL import Image, ImageOps

import config
from review_app import compute_phashes, load_decisions, save_decisions, read_boxes_file
from split_dataset import CLASS_NAMES, base_key, dataset_of

SPLITS = ("train", "valid")
ESCOOTER_CLASS_ID = config.ESCOOTER_CLASS_ID


def other_split(split: str) -> str:
    return "valid" if split == "train" else "train"


def list_images(split_dir: Path) -> list:
    return sorted(p.name for p in (split_dir / "images").iterdir()) if (split_dir / "images").is_dir() else []


def boxes_of(split_dir: Path, name: str) -> list:
    """Tutte le bbox (classe, xc, yc, w, h) del label di un'immagine, per disegnarle."""
    return read_boxes_file(split_dir / "labels" / f"{Path(name).stem}.txt")


def move_files(src_dir: Path, dst_dir: Path, name: str) -> None:
    """Sposta immagine e label (se manca, ne crea uno vuoto a destinazione)
    da src_dir a dst_dir, entrambe cartelle con sottocartelle images/ e
    labels/. Rifiuta di sovrascrivere un'immagine già presente."""
    dst_img = dst_dir / "images" / name
    if dst_img.exists():
        raise FileExistsError(f"{name} è già presente in {dst_dir}")
    (dst_dir / "images").mkdir(parents=True, exist_ok=True)
    (dst_dir / "labels").mkdir(parents=True, exist_ok=True)
    shutil.move(str(src_dir / "images" / name), str(dst_img))
    lbl_src = src_dir / "labels" / f"{Path(name).stem}.txt"
    lbl_dst = dst_dir / "labels" / f"{Path(name).stem}.txt"
    if lbl_src.exists():
        shutil.move(str(lbl_src), str(lbl_dst))
    else:
        lbl_dst.write_text("")


def excluded_dir(dataset_dir: Path, split: str) -> Path:
    return dataset_dir / "excluded" / split


def count_excluded(dataset_dir: Path) -> int:
    return sum(len(list_images(excluded_dir(dataset_dir, s))) for s in SPLITS)


def hex_to_int(h: str) -> int:
    return int(h, 16)


def compute_mirror_phashes(split_dir: Path, cache_path: Path) -> dict:
    """pHash dell'immagine specchiata orizzontalmente, con cache come
    compute_phashes: il dataset contiene copie capovolte ("_flip", ma anche
    senza marcatura) che il pHash normale non avvicina all'originale."""
    cache = load_decisions(cache_path)
    names = list_images(split_dir)
    changed = False
    for name in names:
        path = split_dir / "images" / name
        stat = path.stat()
        entry = cache.get(name)
        if entry and entry.get("mtime") == stat.st_mtime and entry.get("size") == stat.st_size:
            continue
        with Image.open(path) as im:
            phash_hex = str(imagehash.phash(ImageOps.mirror(im)))
        cache[name] = {"mtime": stat.st_mtime, "size": stat.st_size, "phash": phash_hex}
        changed = True
    if changed:
        save_decisions(cache_path, cache)
    return {name: cache[name]["phash"] for name in names if name in cache}


MAX_MATCHES_PER_GROUP = 12  # oltre, la riga mostra "+N altre" per non allungare la pagina


def cross_split_groups(train_phash: dict, valid_phash: dict, train_mirror: dict, dismissed: set, threshold: float) -> list:
    """Per ogni immagine di valid, le immagini di train candidate duplicate:
    distanza di Hamming del pHash <= threshold, calcolata anche contro la
    versione specchiata (vale la minore; "mirrored" dice se è lo specchio),
    oppure stesso nome-base (base_key: stessa foto in dataset diversi,
    copie "_flip") a qualunque distanza. Le più vicine per prime, al massimo
    MAX_MATCHES_PER_GROUP; il resto in "extra". Le coppie in `dismissed`
    (chiave "valid|train") vengono saltate. I gruppi sono ordinati dal più
    sospetto (distanza minima) e omessi se non hanno nessun match."""
    train_items = [(name, hex_to_int(h), hex_to_int(train_mirror[name]), base_key(name))
                   for name, h in train_phash.items() if name in train_mirror]
    groups = []
    for vname, vhash in valid_phash.items():
        vhi = hex_to_int(vhash)
        vkey = base_key(vname)
        matches = []
        for tname, thi, tmi, tkey in train_items:
            d, dm = (vhi ^ thi).bit_count(), (vhi ^ tmi).bit_count()
            dist = min(d, dm)
            same_name = tkey == vkey
            if (dist <= threshold or same_name) and f"{vname}|{tname}" not in dismissed:
                matches.append({"train": tname, "distance": dist, "mirrored": dm < d, "same_name": same_name})
        if matches:
            matches.sort(key=lambda m: m["distance"])
            groups.append({
                "valid": vname, "matches": matches[:MAX_MATCHES_PER_GROUP],
                "extra": max(0, len(matches) - MAX_MATCHES_PER_GROUP),
            })
    groups.sort(key=lambda g: g["matches"][0]["distance"])
    return groups


# --- "Copie sospette yolodet": somiglianza basata sulle detection ORIGINALI ---
CONF_MATCH_MIN_IOU = 0.5  # IoU minima fra una box del label e una detection in cache per attribuirle quella confidence


def original_label_path(source_dir: Path, name: str) -> Path:
    """Label ORIGINALE di un'immagine nel dataset sorgente (union_reviewed_coco):
    la copia salvata da review_app.py prima della prima modifica manuale
    (review_label_backups/), se esiste; altrimenti il label corrente, mai
    modificato quindi ancora quello originale. Si legge dal sorgente e non dallo
    split, i cui label sono copie e possono già includere le modifiche."""
    stem = Path(name).stem
    backup = source_dir / "review_label_backups" / f"{stem}.txt"
    return backup if backup.exists() else source_dir / "labels" / f"{stem}.txt"


def to_xyxy(b) -> tuple:
    """[cls, xc, yc, w, h] -> (x1, y1, x2, y2), tutto normalizzato 0..1 (quindi
    "alla stessa risoluzione" per costruzione, come in find_detection_duplicates.py)."""
    _, xc, yc, w, h = b
    return (xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2)


def iou_xyxy(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def mirror_xyxy(b) -> tuple:
    return (1 - b[2], b[1], 1 - b[0], b[3])


def box_confidence(cache_entry, box):
    """Confidence della detection in cache (annotate_coco_classes.py) che
    corrisponde a una box del label originale: stessa classe, IoU massima
    >= CONF_MATCH_MIN_IOU. None se non trovata (es. l'escooter, classe 80,
    non è nella cache: viene dal dataset annotato, non dal modello COCO)."""
    if not cache_entry:
        return None
    h, w = cache_entry["orig_shape"]
    best_iou, conf = CONF_MATCH_MIN_IOU, None
    bx = to_xyxy(box)
    for cls, c, x1, y1, x2, y2 in cache_entry["detections"]:
        if cls == box[0]:
            i = iou_xyxy(bx, (x1 / w, y1 / h, x2 / w, y2 / h))
            if i >= best_iou:
                best_iou, conf = i, c
    return conf


def image_signature(source_dir: Path, name: str, cache_entry) -> dict:
    """{classe: {"top": xyxy, "all": [xyxy, ...]}} per le box originali
    dell'immagine. "top" è la box di riferimento della classe: quella con la
    confidence più alta in cache o, a parità/senza confidence, la più grande
    per area."""
    by_class = defaultdict(list)
    for b in read_boxes_file(original_label_path(source_dir, name)):
        by_class[b[0]].append(b)
    sig = {}
    for cls, boxes in by_class.items():
        top = max(boxes, key=lambda b: (box_confidence(cache_entry, b) or -1.0, b[3] * b[4]))
        sig[cls] = {"top": to_xyxy(top), "all": [to_xyxy(b) for b in boxes]}
    return sig


def count_overlaps(ref: dict, tgt: dict, min_iou: float, all_boxes: bool, mirrored: bool) -> tuple:
    """(n_sovrapposizioni, somma IoU) fra la firma di riferimento e quella di
    confronto, che devono avere lo stesso insieme di classi. Per default, per
    ogni classe, la sola box "top" del riferimento conta una sovrapposizione se
    copre con IoU >= min_iou una qualunque box della stessa classe dell'altra
    immagine; con all_boxes tutte le box del riferimento, accoppiate a quelle
    dell'altra (greedy per IoU decrescente, ognuna usata una volta). Con
    mirrored il riferimento è specchiato orizzontalmente."""
    n, total = 0, 0.0
    for cls, r in ref.items():
        mine = r["all"] if all_boxes else [r["top"]]
        if mirrored:
            mine = [mirror_xyxy(b) for b in mine]
        theirs = tgt[cls]["all"]
        pairs = sorted(((iou_xyxy(a, b), i, j) for i, a in enumerate(mine) for j, b in enumerate(theirs)), reverse=True)
        used_i, used_j = set(), set()
        for v, i, j in pairs:
            if v < min_iou:
                break
            if i in used_i or (all_boxes and j in used_j):
                continue
            used_i.add(i)
            used_j.add(j)
            n += 1
            total += v
    return n, total


def yolodet_groups(valid_sigs: dict, train_sigs: dict, dismissed: set, min_overlaps: int,
                   tolerance: float, all_boxes: bool) -> list:
    """Per ogni immagine di valid, le immagini di train con lo STESSO insieme di
    classi (se differiscono, non sono considerate simili) e almeno
    min_overlaps sovrapposizioni di box (v. count_overlaps) con IoU >= 1 -
    tolerance, anche contro il riferimento specchiato. Il riferimento è
    sempre l'immagine di valid. Ordinati dalla più sospetta (più
    sovrapposizioni, poi IoU media); stessi limiti di cross_split_groups."""
    train_by_classes = defaultdict(list)
    for name, sig in train_sigs.items():
        if sig:
            train_by_classes[frozenset(sig)].append(name)
    min_iou = 1.0 - tolerance
    groups = []
    for vname, vsig in valid_sigs.items():
        if not vsig:
            continue
        matches = []
        for tname in train_by_classes.get(frozenset(vsig), []):
            if f"{vname}|{tname}" in dismissed:
                continue
            best = max((count_overlaps(vsig, train_sigs[tname], min_iou, all_boxes, m) + (m,) for m in (False, True)),
                       key=lambda t: (t[0], t[1]))
            n, total, mirrored = best
            if n >= min_overlaps:
                matches.append({"train": tname, "overlaps": n, "iou": total / n, "mirrored": mirrored,
                                "same_name": base_key(tname) == base_key(vname)})
        if matches:
            matches.sort(key=lambda m: (-m["overlaps"], -m["iou"]))
            groups.append({
                "valid": vname, "matches": matches[:MAX_MATCHES_PER_GROUP],
                "extra": max(0, len(matches) - MAX_MATCHES_PER_GROUP),
                "classes": ", ".join(CLASS_NAMES[c] for c in sorted(vsig) if c < len(CLASS_NAMES)),
            })
    groups.sort(key=lambda g: (-g["matches"][0]["overlaps"], -g["matches"][0]["iou"]))
    return groups


# --- Scheda "Statistiche": istanze e dimensioni dell'escooter per split ---
PERCENTILES = (5, 25, 50, 75, 95)
COCO_SMALL_AREA, COCO_MEDIUM_AREA = 32 ** 2, 96 ** 2  # soglie COCO sull'area in pixel^2
MAX_INSTANCES_BUCKET = 5  # l'istogramma "istanze per immagine" raggruppa da qui in su ("5+")


def split_stats(dataset_dir: Path, image_size) -> dict:
    """Statistiche sulle box escooter (ESCOOTER_CLASS_ID) dei label ATTUALI di
    ogni split (escluse le immagini in excluded/), per vedere come sono
    distribuite fra train e valid: numero di istanze, istanze per immagine,
    dimensione (area in % dell'immagine, lato lungo e area in pixel, con le
    fasce COCO piccola/media/grande) e il dettaglio per dataset sorgente.
    image_size(path) -> (larghezza, altezza) in pixel."""
    rows = {s: [] for s in SPLITS}  # per split: (dataset, area_frac, lato_lungo_px, area_px)
    counts = {s: [] for s in SPLITS}  # per split: istanze escooter di ogni immagine
    images = {s: defaultdict(int) for s in SPLITS}  # per split: dataset -> n immagini
    for split in SPLITS:
        split_dir = dataset_dir / split
        for name in list_images(split_dir):
            ds = dataset_of(name)
            images[split][ds] += 1
            boxes = [b for b in boxes_of(split_dir, name) if b[0] == ESCOOTER_CLASS_ID]
            counts[split].append(len(boxes))
            if boxes:
                w_px, h_px = image_size(split_dir / "images" / name)
                for _, _xc, _yc, w, h in boxes:
                    rows[split].append((ds, w * h, max(w * w_px, h * h_px), w * w_px * h * h_px))

    def pct(values: list) -> dict:
        return {str(p): float(np.percentile(values, p)) for p in PERCENTILES} if values else {}

    def share(values: list, lo: float, hi: float) -> float:
        return sum(lo <= v < hi for v in values) / len(values) if values else 0.0

    splits = {}
    for split in SPLITS:
        areas = [r[3] for r in rows[split]]
        n_img = len(counts[split])
        splits[split] = {
            "images": n_img,
            "images_with_escooter": sum(c > 0 for c in counts[split]),
            "instances": len(rows[split]),
            "per_image": [
                sum((c == k) if k < MAX_INSTANCES_BUCKET else (c >= k) for c in counts[split]) / n_img if n_img else 0.0
                for k in range(MAX_INSTANCES_BUCKET + 1)
            ],
            "area_pct": pct([r[1] * 100 for r in rows[split]]),
            "long_side_px": pct([r[2] for r in rows[split]]),
            "coco": {
                "small": share(areas, 0, COCO_SMALL_AREA),
                "medium": share(areas, COCO_SMALL_AREA, COCO_MEDIUM_AREA),
                "large": share(areas, COCO_MEDIUM_AREA, float("inf")),
                "n_small": sum(a < COCO_SMALL_AREA for a in areas),
                "n_medium": sum(COCO_SMALL_AREA <= a < COCO_MEDIUM_AREA for a in areas),
            },
        }

    datasets = []
    for ds in sorted(set(images["train"]) | set(images["valid"])):
        entry = {"dataset": ds}
        for split in SPLITS:
            mine = [r for r in rows[split] if r[0] == ds]
            entry[split] = {
                "images": images[split].get(ds, 0), "instances": len(mine),
                "n_small": sum(r[3] < COCO_SMALL_AREA for r in mine),
                "median_area_pct": float(np.median([r[1] * 100 for r in mine])) if mine else None,
            }
        datasets.append(entry)
    return {"splits": splits, "datasets": datasets}


PAGE = """<!doctype html>
<html lang="it">
<head>
<meta charset="utf-8">
<title>Revisione split train/valid</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin: 0; background: #111; color: #eee; font-family: system-ui, sans-serif; font-size: 14px; }
  #bar { display: flex; justify-content: space-between; align-items: center; padding: 10px 16px; background: #1b1b1b; border-bottom: 1px solid #333; flex-wrap: wrap; gap: 12px; }
  #bar .counts span { margin-right: 16px; color: #ccc; }
  #tabs { display: flex; gap: 6px; align-items: center; }
  #tabs button { background: #222; color: #ccc; border: 1px solid #444; border-radius: 4px; padding: 6px 14px; cursor: pointer; }
  #tabs button.active { background: #2d5a2d; color: #fff; border-color: #3a7a3a; }
  #tabs button:hover { background: #333; }
  #tabs button#undo { margin-left: 16px; background: #4a3a20; border-color: #7a6030; color: #fff; }
  #tabs button#undo:hover { background: #5a4828; }
  main { padding: 16px; }
  .hidden { display: none !important; }
  select, input[type=number] { background: #222; color: #eee; border: 1px solid #444; border-radius: 4px; padding: 3px 6px; }
  button.small { background: #333; color: #eee; border: 1px solid #555; border-radius: 4px; padding: 3px 10px; cursor: pointer; }
  button.small:hover { background: #444; }
  button.move { background: #2d4d5a; border-color: #3a7a9a; }
  button.move:hover { background: #37627a; }
  button.exclude { background: #5a2d2d; border-color: #9a3a3a; }
  button.exclude:hover { background: #7a3737; }
  button.dismiss { background: #333; border-color: #666; }

  .imgwrap { position: relative; background: #000; border-radius: 4px; overflow: hidden; }
  .imgwrap img { position: absolute; left: 0; top: 0; width: 100%; height: 100%; object-fit: contain; }
  .imgwrap canvas { position: absolute; left: 0; top: 0; width: 100%; height: 100%; pointer-events: none; }

  #browse { display: flex; gap: 20px; }
  .col { flex: 1; min-width: 0; }
  .col h2 { font-size: 15px; margin: 0 0 8px; color: #9c9; }
  .col.valid h2 { color: #9bc; }
  .colbar { display: flex; align-items: center; gap: 8px; margin-bottom: 10px; flex-wrap: wrap; }
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 12px; }
  .card { background: #1b1b1b; border: 1px solid #333; border-radius: 6px; padding: 6px; }
  .card .imgwrap { width: 100%; aspect-ratio: 4/3; }
  .card .meta { font-size: 11px; color: #999; margin: 4px 0; word-break: break-all; }
  .card .meta b { color: #ccc; }
  .card .btns { display: flex; gap: 4px; margin-top: 4px; }
  .card .btns button { flex: 1; }
  .pageinfo { color: #999; font-size: 12px; }

  .pairs table { width: 100%; border-collapse: collapse; }
  .pairs th { text-align: left; color: #999; font-weight: normal; font-size: 12px; padding: 6px; border-bottom: 1px solid #333; }
  .pairs td { padding: 12px 6px; border-bottom: 1px solid #333; vertical-align: top; }
  .pairs .imgwrap { width: 220px; height: 165px; }
  .pairs .name { font-size: 11px; color: #999; width: 220px; word-break: break-all; margin: 4px 0; }
  .pairs .vcell { width: 236px; }
  .pairs .matches { display: flex; flex-wrap: wrap; gap: 14px; }
  .pairs .match { width: 220px; }
  .pairs .match .dist { font-size: 16px; font-weight: bold; margin-bottom: 3px; }
  .pairs .dist.low { color: #e66; }
  .pairs .dist.mid { color: #ea4; }
  .pairs .dist.high { color: #6a6; }
  .pairs .actions { display: flex; flex-direction: column; gap: 4px; width: 220px; }
  .pairs .extra { color: #999; align-self: center; }
  #stats table { border-collapse: collapse; margin-bottom: 24px; }
  #stats th { text-align: right; color: #999; font-weight: normal; font-size: 12px; padding: 5px 14px; border-bottom: 1px solid #333; }
  #stats th:first-child, #stats td:first-child { text-align: left; }
  #stats td { padding: 4px 14px; border-bottom: 1px solid #262626; text-align: right; }
  #stats td.sub { color: #888; padding-left: 28px; }
  #stats h3 { margin: 4px 0 8px; font-size: 14px; color: #ccc; font-weight: normal; }
  #topbar-pairs, .topbar { display: flex; align-items: center; gap: 14px; margin-bottom: 12px; flex-wrap: wrap; }
  #legend { color: #999; font-size: 12px; }
  #legend i { display: inline-block; width: 10px; height: 10px; margin: 0 3px 0 10px; vertical-align: middle; }
</style>
</head>
<body>
<div id="bar">
  <div class="counts">
    <span>train: <b id="c-train">-</b></span>
    <span>valid: <b id="c-valid">-</b></span>
    <span>esclusi: <b id="c-excluded">-</b></span>
    <span>azioni in questa sessione: <b id="c-moves">0</b></span>
  </div>
  <div id="tabs">
    <button id="tab-pairs" class="active">Coppie sospette</button>
    <button id="tab-yolo" title="Immagini con le stesse classi e box sovrapposte nelle detection originali">Copie sospette yolodet</button>
    <button id="tab-browse">Sfoglia</button>
    <button id="tab-stats" title="Istanze e dimensione dell'escooter in train e valid">Statistiche</button>
    <button id="undo" title="Ripristina l'ultimo spostamento o l'ultima esclusione">Annulla ultima azione</button>
  </div>
</div>

<main>
  <section id="pairs" class="pairs">
    <div id="topbar-pairs">
      <label>soglia distanza pHash &le; <input type="number" id="pairs-threshold" value="10" min="0" max="64" style="width:55px"></label>
      <button class="small" id="pairs-reload">Ricalcola</button>
      <span class="pageinfo" id="pairs-count"></span>
      <span id="legend"><i style="background:#ff3b3b"></i>escooter<i style="background:#3fd23f"></i>person<i style="background:#3b9bff"></i>altre classi</span>
    </div>
    <table>
      <thead><tr><th>valid</th><th>candidati duplicati in train (dal più vicino)</th></tr></thead>
      <tbody id="pairs-body"></tbody>
    </table>
  </section>

  <section id="pairs-yolo" class="pairs hidden">
    <div id="topbar-yolo" class="topbar">
      <label>sovrapposizioni minime N &ge; <input type="number" id="yolo-n" value="3" min="1" max="20" style="width:55px"></label>
      <label>tolleranza T <input type="number" id="yolo-t" value="0.3" min="0" max="1" step="0.05" style="width:65px"
        title="due box si considerano sovrapposte se IoU >= 1 - T"></label>
      <label title="Di default conta solo la box migliore (confidence più alta, o area maggiore) di ogni classe dell'immagine di valid; qui conta tutte le box, accoppiate una a una"><input type="checkbox" id="yolo-all"> tutte le box</label>
      <button class="small" id="yolo-reload">Ricalcola</button>
      <span class="pageinfo" id="yolo-count"></span>
    </div>
    <div class="pageinfo" style="margin-bottom:10px">Stesso insieme di classi nelle box ORIGINALI (prima delle tue modifiche); box scalate alla stessa risoluzione; sovrapposte se IoU &ge; 1 &minus; T, anche con l'immagine di valid specchiata.</div>
    <table>
      <thead><tr><th>valid</th><th>candidati in train (dal più sovrapposto)</th></tr></thead>
      <tbody id="yolo-body"></tbody>
    </table>
  </section>

  <section id="stats" class="hidden">
    <div class="topbar">
      <button class="small" id="stats-reload">Ricalcola</button>
      <span class="pageinfo">Solo la classe escooter, sui label attuali (escluse le immagini in excluded/). Fasce COCO sull'area in pixel: piccola &lt; 32&sup2;, media &lt; 96&sup2;.</span>
    </div>
    <div id="stats-body"></div>
  </section>

  <section id="browse" class="hidden">
    <div class="col train">
      <h2>TRAIN</h2>
      <div class="colbar">
        <select id="train-dataset"></select>
        <label>per pagina <input type="number" id="train-pagesize" value="4" min="1" max="50" style="width:50px"></label>
        <button class="small" id="train-prev">&larr;</button>
        <span class="pageinfo" id="train-pageinfo"></span>
        <button class="small" id="train-next">&rarr;</button>
      </div>
      <div class="grid" id="train-grid"></div>
    </div>
    <div class="col valid">
      <h2>VALID</h2>
      <div class="colbar">
        <select id="valid-dataset"></select>
        <label>per pagina <input type="number" id="valid-pagesize" value="4" min="1" max="50" style="width:50px"></label>
        <button class="small" id="valid-prev">&larr;</button>
        <span class="pageinfo" id="valid-pageinfo"></span>
        <button class="small" id="valid-next">&rarr;</button>
      </div>
      <div class="grid" id="valid-grid"></div>
    </div>
  </section>
</main>

<script>
const ESCOOTER_CLASS = __ESCOOTER_CLASS_ID__;
const state = { browseOffset: { train: 0, valid: 0 }, actions: 0, tab: "pairs" };

function ea(s) { return s.replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;"); }
function boxColor(cls) { return cls === ESCOOTER_CLASS ? "#ff3b3b" : cls === 0 ? "#3fd23f" : "#3b9bff"; }

function drawBoxes(img) {
  const wrap = img.parentElement, cv = wrap.querySelector("canvas");
  const cw = wrap.clientWidth, ch = wrap.clientHeight;
  cv.width = cw; cv.height = ch;
  const nw = img.naturalWidth, nh = img.naturalHeight;
  if (!nw || !nh) return;
  const sc = Math.min(cw / nw, ch / nh), dw = nw * sc, dh = nh * sc;
  const ox = (cw - dw) / 2, oy = (ch - dh) / 2;
  const boxes = JSON.parse(decodeURIComponent(img.dataset.boxes));
  boxes.sort((a, b) => (a[0] === ESCOOTER_CLASS) - (b[0] === ESCOOTER_CLASS));  // escooter per ultime, sopra le altre
  const ctx = cv.getContext("2d");
  for (const [cls, xc, yc, w, h] of boxes) {
    ctx.strokeStyle = boxColor(cls);
    ctx.lineWidth = cls === ESCOOTER_CLASS ? 3 : 1.5;
    ctx.strokeRect(ox + (xc - w / 2) * dw, oy + (yc - h / 2) * dh, w * dw, h * dh);
  }
}
window.addEventListener("resize", () => document.querySelectorAll("img[data-boxes]").forEach(drawBoxes));

function thumb(split, name, boxes) {
  return `<div class="imgwrap"><img src="/api/image/${split}/${encodeURIComponent(name)}"
    data-boxes="${encodeURIComponent(JSON.stringify(boxes))}" onload="drawBoxes(this)"><canvas></canvas></div>`;
}

async function refreshCounts() {
  const s = await (await fetch("/api/state")).json();
  document.getElementById("c-train").textContent = s.train_count;
  document.getElementById("c-valid").textContent = s.valid_count;
  document.getElementById("c-excluded").textContent = s.excluded_count;
  for (const split of ["train", "valid"]) {
    const sel = document.getElementById(split + "-dataset");
    const current = sel.value;
    sel.innerHTML = '<option value="">tutti i dataset</option>' +
      s.datasets[split].map(d => `<option value="${ea(d)}">${d}</option>`).join("");
    if (s.datasets[split].includes(current)) sel.value = current;
  }
}

function distClass(d) { return d <= 12 ? "low" : d <= 20 ? "mid" : "high"; }
function nEsc(boxes) { return boxes.filter(b => b[0] === ESCOOTER_CLASS).length; }

function matchHead(m) {
  if (m.overlaps !== undefined)
    return `<div class="dist ${m.overlaps >= 4 ? "low" : m.overlaps === 3 ? "mid" : "high"}">${m.overlaps} sovrapposizioni · IoU ${m.iou.toFixed(2)}${m.mirrored ? " <small>(specchio)</small>" : ""}${m.same_name ? " <small>(stesso nome)</small>" : ""}</div>`;
  return `<div class="dist ${distClass(m.distance)}">distanza ${m.distance}${m.mirrored ? " <small>(specchio)</small>" : ""}${m.same_name ? " <small>(stesso nome)</small>" : ""}</div>`;
}

function groupRows(groups) {
  return groups.map(g => `
    <tr>
      <td class="vcell">
        ${thumb("valid", g.valid, g.valid_boxes)}
        <div class="name">${g.valid}<br>${nEsc(g.valid_boxes)} bbox escooter${g.classes ? "<br>classi: " + g.classes : ""}</div>
        <div class="actions">
          <button class="small exclude" data-act="exclude" data-split="valid" data-name="${ea(g.valid)}">escludi da valid</button>
          <button class="small move" data-act="move" data-split="valid" data-name="${ea(g.valid)}">sposta valid &rarr; train</button>
          <button class="small dismiss" data-act="dismiss" data-valid="${ea(g.valid)}"
            data-trains="${ea(JSON.stringify(g.matches.map(m => m.train)))}">nessuno è un duplicato</button>
        </div>
      </td>
      <td><div class="matches">
        ${g.matches.map(m => `
          <div class="match">
            ${matchHead(m)}
            ${thumb("train", m.train, m.boxes)}
            <div class="name">${m.train}<br>${nEsc(m.boxes)} bbox escooter</div>
            <div class="actions">
              <button class="small exclude" data-act="exclude" data-split="train" data-name="${ea(m.train)}">escludi da train</button>
              <button class="small move" data-act="move" data-split="train" data-name="${ea(m.train)}">sposta train &rarr; valid</button>
              <button class="small dismiss" data-act="dismiss" data-valid="${ea(g.valid)}" data-train="${ea(m.train)}">non è un duplicato</button>
            </div>
          </div>`).join("")}
        ${g.extra ? `<div class="extra">+${g.extra} altre sotto soglia</div>` : ""}
      </div></td>
    </tr>`).join("");
}

function showGroups(data, bodyId, countId) {
  if (data.error) {
    document.getElementById(bodyId).innerHTML = "";
    document.getElementById(countId).textContent = data.error;
    return;
  }
  const nPairs = data.groups.reduce((n, g) => n + g.matches.length + g.extra, 0);
  document.getElementById(countId).textContent = `${data.groups.length} immagini di valid con candidati, ${nPairs} coppie`;
  document.getElementById(bodyId).innerHTML = groupRows(data.groups);
}

async function loadPairs() {
  const th = document.getElementById("pairs-threshold").value;
  showGroups(await (await fetch(`/api/groups?threshold=${th}`)).json(), "pairs-body", "pairs-count");
}

async function loadYolo() {
  const n = document.getElementById("yolo-n").value, t = document.getElementById("yolo-t").value;
  const all = document.getElementById("yolo-all").checked ? 1 : 0;
  document.getElementById("yolo-count").textContent = "calcolo in corso...";
  showGroups(await (await fetch(`/api/groups-yolodet?min_overlaps=${n}&tolerance=${t}&all_boxes=${all}`)).json(), "yolo-body", "yolo-count");
}

function statsTable(s) {
  const f1 = x => x === null || x === undefined ? "-" : x.toFixed(1);
  const pc = x => (x * 100).toFixed(1) + "%";
  const tr = (label, fn, cls) => `<tr><td${cls ? ` class="${cls}"` : ""}>${label}</td><td>${fn("train")}</td><td>${fn("valid")}</td></tr>`;
  const sp = s.splits;
  let html = `<h3>Escooter per split</h3><table><thead><tr><th></th><th>train</th><th>valid</th></tr></thead><tbody>`;
  html += tr("immagini", k => sp[k].images);
  html += tr("immagini con escooter", k => sp[k].images_with_escooter);
  html += tr("istanze escooter", k => sp[k].instances);
  html += tr("istanze per immagine (media)", k => sp[k].images ? (sp[k].instances / sp[k].images).toFixed(2) : "-");
  ["0", "1", "2", "3", "4", "5+"].forEach((lab, i) => { html += tr(`immagini con ${lab} istanze`, k => pc(sp[k].per_image[i]), "sub"); });
  html += tr("area box, % dell'immagine", () => "", "");
  [5, 25, 50, 75, 95].forEach(p => { html += tr(`p${p}`, k => f1(sp[k].area_pct[p]), "sub"); });
  html += tr("lato lungo, px", () => "", "");
  [5, 25, 50, 75, 95].forEach(p => { html += tr(`p${p}`, k => f1(sp[k].long_side_px[p]), "sub"); });
  html += tr("fasce COCO (area px)", () => "", "");
  html += tr("piccole", k => `${pc(sp[k].coco.small)} (${sp[k].coco.n_small})`, "sub");
  html += tr("medie", k => `${pc(sp[k].coco.medium)} (${sp[k].coco.n_medium})`, "sub");
  html += tr("grandi", k => pc(sp[k].coco.large), "sub");
  html += `</tbody></table><h3>Per dataset sorgente</h3><table><thead><tr><th>dataset</th><th>img train</th><th>img valid</th>
    <th>istanze train</th><th>istanze valid</th><th>piccole train</th><th>piccole valid</th><th>area med. train %</th><th>area med. valid %</th></tr></thead><tbody>`;
  for (const d of s.datasets)
    html += `<tr><td>${d.dataset}</td><td>${d.train.images}</td><td>${d.valid.images}</td><td>${d.train.instances}</td><td>${d.valid.instances}</td>
      <td>${d.train.n_small}</td><td>${d.valid.n_small}</td><td>${f1(d.train.median_area_pct)}</td><td>${f1(d.valid.median_area_pct)}</td></tr>`;
  return html + "</tbody></table>";
}

async function loadStats() {
  document.getElementById("stats-body").textContent = "calcolo in corso...";
  document.getElementById("stats-body").innerHTML = statsTable(await (await fetch("/api/stats")).json());
}

// ricarica la scheda attiva (dopo uno spostamento, un'esclusione, un "non è un duplicato")
async function reloadPairsTab() {
  if (state.tab === "yolo") await loadYolo();
  else if (state.tab === "stats") await loadStats();
  else await loadPairs();
}

async function loadBrowse(split) {
  const ds = document.getElementById(split + "-dataset").value;
  const limit = parseInt(document.getElementById(split + "-pagesize").value) || 4;
  const offset = state.browseOffset[split];
  const data = await (await fetch(`/api/list?split=${split}&dataset=${encodeURIComponent(ds)}&offset=${offset}&limit=${limit}`)).json();
  document.getElementById(split + "-pageinfo").textContent =
    data.total ? `${offset + 1}-${Math.min(offset + limit, data.total)} di ${data.total}` : "0 immagini";
  const arrow = split === "train" ? "&rarr; valid" : "&larr; train";
  document.getElementById(split + "-grid").innerHTML = data.items.map(it => `
    <div class="card">${thumb(split, it.name, it.boxes)}
      <div class="meta"><b>${it.dataset}</b><br>${it.name}<br>${it.boxes.filter(b => b[0] === ESCOOTER_CLASS).length} bbox escooter</div>
      <div class="btns">
        <button class="small move" data-act="move" data-split="${split}" data-name="${ea(it.name)}">sposta ${arrow}</button>
        <button class="small exclude" data-act="exclude" data-split="${split}" data-name="${ea(it.name)}">escludi</button>
      </div>
    </div>`).join("");
}

async function refreshAll() {
  await refreshCounts();
  await reloadPairsTab();
  await loadBrowse("train");
  await loadBrowse("valid");
}

async function post(path, body) {
  const r = await fetch(path, { method: "POST", body: JSON.stringify(body || {}) });
  return r.json();
}

document.addEventListener("click", async e => {
  const b = e.target.closest("button[data-act]");
  if (!b) return;
  const act = b.dataset.act;
  if (act === "dismiss") {
    await post("/api/dismiss-pair", {
      valid: b.dataset.valid,
      ...(b.dataset.trains ? { trains: JSON.parse(b.dataset.trains) } : { train: b.dataset.train }),
    });
    await reloadPairsTab();
    return;
  }
  const res = await post("/api/" + act, { name: b.dataset.name, from_split: b.dataset.split });
  if (!res.ok) { alert(res.error || "operazione fallita"); return; }
  state.actions++; document.getElementById("c-moves").textContent = state.actions;
  await refreshAll();
});

document.getElementById("undo").onclick = async () => {
  const res = await post("/api/undo");
  if (!res.ok) { alert(res.error || "niente da annullare"); return; }
  state.actions = Math.max(0, state.actions - 1); document.getElementById("c-moves").textContent = state.actions;
  await refreshAll();
};

for (const split of ["train", "valid"]) {
  const pageSize = () => parseInt(document.getElementById(split + "-pagesize").value) || 4;
  document.getElementById(split + "-prev").onclick = () => {
    state.browseOffset[split] = Math.max(0, state.browseOffset[split] - pageSize());
    loadBrowse(split);
  };
  document.getElementById(split + "-next").onclick = () => {
    state.browseOffset[split] += pageSize();
    loadBrowse(split);
  };
  for (const id of ["-dataset", "-pagesize"])
    document.getElementById(split + id).addEventListener("change", () => { state.browseOffset[split] = 0; loadBrowse(split); });
}
document.getElementById("pairs-reload").onclick = loadPairs;
document.getElementById("yolo-reload").onclick = loadYolo;

function showTab(name) {
  state.tab = name;
  for (const t of ["pairs", "yolo", "browse", "stats"]) {
    document.getElementById("tab-" + t).classList.toggle("active", name === t);
    document.getElementById(t === "yolo" ? "pairs-yolo" : t).classList.toggle("hidden", name !== t);
  }
  // ridisegna le bbox: a tab nascosto i canvas hanno dimensione 0
  if (name === "browse") { loadBrowse("train"); loadBrowse("valid"); } else reloadPairsTab();
}
document.getElementById("tab-pairs").onclick = () => showTab("pairs");
document.getElementById("tab-yolo").onclick = () => showTab("yolo");
document.getElementById("tab-browse").onclick = () => showTab("browse");
document.getElementById("tab-stats").onclick = () => showTab("stats");
document.getElementById("stats-reload").onclick = loadStats;

refreshCounts().then(loadPairs);
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    # Impostati su main() prima dell'avvio del server.
    dataset_dir: Path
    phash_cache_paths: dict  # {"train": Path, "valid": Path}
    mirror_cache_paths: dict  # {"train": Path}
    source_dir: Path  # dataset sorgente (union_reviewed_coco), per i label originali
    detection_cache_path: Path  # cache delle detection COCO (annotate_coco_classes.py)
    _detection_cache = None  # (mtime, dict), ricaricata se il file cambia
    _signatures: dict = {}  # nome -> (chiave mtime, firma), v. image_signature()
    _image_sizes: dict = {}  # percorso -> (mtime, (larghezza, altezza)), per le statistiche
    dismissed_path: Path
    move_log_path: Path

    def _json(self, obj, status: int = 200) -> None:
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _phashes(self, split: str) -> dict:
        return compute_phashes(self.dataset_dir / split, self.phash_cache_paths[split])

    def _mirror_phashes(self, split: str) -> dict:
        return compute_mirror_phashes(self.dataset_dir / split, self.mirror_cache_paths[split])

    def _detections(self) -> dict:
        mtime = self.detection_cache_path.stat().st_mtime
        cached = Handler._detection_cache
        if cached is None or cached[0] != mtime:
            Handler._detection_cache = (mtime, json.loads(self.detection_cache_path.read_text()))
        return Handler._detection_cache[1]

    def _signature(self, name: str, detections: dict) -> dict:
        """Firma delle box originali di un'immagine, memorizzata finché non cambia
        il label (o il suo backup): ricalcolare N e T non rilegge tutti i file."""
        label = original_label_path(self.source_dir, name)
        key = (label, label.stat().st_mtime if label.exists() else 0)
        cached = Handler._signatures.get(name)
        if cached is None or cached[0] != key:
            Handler._signatures[name] = (key, image_signature(self.source_dir, name, detections.get(name)))
        return Handler._signatures[name][1]

    def _image_size(self, path: Path) -> tuple:
        mtime = path.stat().st_mtime
        cached = Handler._image_sizes.get(path)
        if cached is None or cached[0] != mtime:
            with Image.open(path) as im:
                Handler._image_sizes[path] = (mtime, im.size)
        return Handler._image_sizes[path][1]

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)

        if parsed.path in ("/", "/index.html"):
            body = PAGE.replace("__ESCOOTER_CLASS_ID__", str(ESCOOTER_CLASS_ID)).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif parsed.path == "/api/state":
            train_names = list_images(self.dataset_dir / "train")
            valid_names = list_images(self.dataset_dir / "valid")
            self._json({
                "train_count": len(train_names), "valid_count": len(valid_names),
                "excluded_count": count_excluded(self.dataset_dir),
                "datasets": {
                    "train": sorted(set(dataset_of(n) for n in train_names)),
                    "valid": sorted(set(dataset_of(n) for n in valid_names)),
                },
            })

        elif parsed.path == "/api/list":
            split = qs.get("split", [""])[0]
            if split not in SPLITS:
                self.send_error(400, "split deve essere train o valid")
                return
            ds_filter = qs.get("dataset", [""])[0]
            offset = int(qs.get("offset", ["0"])[0])
            limit = int(qs.get("limit", ["4"])[0])
            split_dir = self.dataset_dir / split
            names = list_images(split_dir)
            if ds_filter:
                names = [n for n in names if dataset_of(n) == ds_filter]
            page = names[offset:offset + limit]
            items = [{"name": n, "dataset": dataset_of(n), "boxes": boxes_of(split_dir, n)} for n in page]
            self._json({"items": items, "total": len(names)})

        elif parsed.path.startswith("/api/image/"):
            rest = parsed.path[len("/api/image/"):]
            split, _, name = rest.partition("/")
            name = unquote(name)
            if split not in SPLITS:
                self.send_error(404)
                return
            path = self.dataset_dir / split / "images" / name
            if not path.exists():
                self.send_error(404)
                return
            data = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type(name)[0] or "application/octet-stream")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        elif parsed.path == "/api/groups":
            threshold = float(qs.get("threshold", ["10"])[0])
            dismissed = set(load_decisions(self.dismissed_path).keys())
            groups = cross_split_groups(self._phashes("train"), self._phashes("valid"), self._mirror_phashes("train"), dismissed, threshold)
            for g in groups:
                g["valid_boxes"] = boxes_of(self.dataset_dir / "valid", g["valid"])
                for m in g["matches"]:
                    m["boxes"] = boxes_of(self.dataset_dir / "train", m["train"])
            self._json({"groups": groups})

        elif parsed.path == "/api/stats":
            self._json(split_stats(self.dataset_dir, self._image_size))

        elif parsed.path == "/api/groups-yolodet":
            if not self.detection_cache_path.exists() or not (self.source_dir / "labels").is_dir():
                self._json({"error": f"servono la cache detection ({self.detection_cache_path}) e il dataset "
                                     f"sorgente ({self.source_dir}): v. --detection-cache e --source-dir"}, 400)
                return
            min_overlaps = int(qs.get("min_overlaps", ["3"])[0])
            tolerance = float(qs.get("tolerance", ["0.3"])[0])
            all_boxes = qs.get("all_boxes", ["0"])[0] == "1"
            dismissed = set(load_decisions(self.dismissed_path).keys())
            detections = self._detections()
            sigs = {split: {n: self._signature(n, detections) for n in list_images(self.dataset_dir / split)}
                    for split in SPLITS}
            groups = yolodet_groups(sigs["valid"], sigs["train"], dismissed, min_overlaps, tolerance, all_boxes)
            for g in groups:
                g["valid_boxes"] = boxes_of(self.dataset_dir / "valid", g["valid"])
                for m in g["matches"]:
                    m["boxes"] = boxes_of(self.dataset_dir / "train", m["train"])
            self._json({"groups": groups})

        else:
            self.send_error(404)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}

        if self.path in ("/api/move", "/api/exclude"):
            name, from_split = body["name"], body["from_split"]
            if from_split not in SPLITS:
                self._json({"ok": False, "error": "from_split non valido"}, 400)
                return
            action = self.path.rsplit("/", 1)[1]
            dst = self.dataset_dir / other_split(from_split) if action == "move" else excluded_dir(self.dataset_dir, from_split)
            try:
                move_files(self.dataset_dir / from_split, dst, name)
            except (FileNotFoundError, FileExistsError) as e:
                self._json({"ok": False, "error": str(e)}, 400)
                return
            log = load_decisions(self.move_log_path)
            log.setdefault("moves", []).append({"action": action, "name": name, "from": from_split, "ts": time.time()})
            save_decisions(self.move_log_path, log)
            self._json({"ok": True})

        elif self.path == "/api/undo":
            log = load_decisions(self.move_log_path)
            if not log.get("moves"):
                self._json({"ok": False, "error": "niente da annullare"})
                return
            last = log["moves"][-1]
            src = self.dataset_dir / other_split(last["from"]) if last.get("action", "move") == "move" \
                else excluded_dir(self.dataset_dir, last["from"])
            try:
                move_files(src, self.dataset_dir / last["from"], last["name"])
            except (FileNotFoundError, FileExistsError) as e:
                self._json({"ok": False, "error": str(e)}, 400)
                return
            log["moves"].pop()
            save_decisions(self.move_log_path, log)
            self._json({"ok": True, "restored": last["name"], "to_split": last["from"]})

        elif self.path == "/api/dismiss-pair":
            dismissed = load_decisions(self.dismissed_path)
            for train_name in body.get("trains") or [body["train"]]:
                dismissed[f"{body['valid']}|{train_name}"] = True
            save_decisions(self.dismissed_path, dismissed)
            self._json({"ok": True})

        else:
            self.send_error(404)

    def log_message(self, format: str, *args) -> None:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset_dir", type=Path,
                         help="Cartella dello split, con sottocartelle train/{images,labels} e valid/{images,labels}")
    parser.add_argument("--port", type=int, default=config.SPLIT_REVIEW_APP_PORT)
    parser.add_argument("--source-dir", type=Path, default=config.UNION_REVIEWED_COCO_DIR,
                         help="Dataset sorgente dello split, da cui leggere i label ORIGINALI (backup di "
                              f"review_app.py) per \"Copie sospette yolodet\" (default: {config.UNION_REVIEWED_COCO_DIR})")
    parser.add_argument("--detection-cache", type=Path, default=config.COCO_ANNOTATION_CACHE_PATH,
                         help="Cache delle detection COCO con confidence (annotate_coco_classes.py), "
                              f"per \"Copie sospette yolodet\" (default: {config.COCO_ANNOTATION_CACHE_PATH})")
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    for split in SPLITS:
        if not (dataset_dir / split / "images").is_dir() or not (dataset_dir / split / "labels").is_dir():
            parser.error(f"{dataset_dir / split} deve contenere le sottocartelle images/ e labels/")

    Handler.dataset_dir = dataset_dir
    Handler.phash_cache_paths = {
        "train": dataset_dir / "split_review_phash_cache_train.json",
        "valid": dataset_dir / "split_review_phash_cache_valid.json",
    }
    Handler.mirror_cache_paths = {"train": dataset_dir / "split_review_phash_mirror_cache_train.json"}
    Handler.source_dir = args.source_dir.resolve()
    Handler.detection_cache_path = args.detection_cache.resolve()
    Handler.dismissed_path = dataset_dir / "split_review_dismissed.json"
    Handler.move_log_path = dataset_dir / "split_review_moves.json"

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Dataset: {dataset_dir}")
    print(f"train: {len(list_images(dataset_dir / 'train'))} immagini, valid: {len(list_images(dataset_dir / 'valid'))} immagini")
    print(f"Log spostamenti: {Handler.move_log_path}")
    print(f"Apri http://localhost:{args.port} nel browser (Ctrl+C per fermare)")
    server.serve_forever()


if __name__ == "__main__":
    main()
