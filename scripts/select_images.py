#!/usr/bin/env python3
"""Seleziona le immagini "migliori" tra i dataset deduplicati in data/interim.

Criteri (vedi claude-instruct-01-automatic-image-selection.md):
1. filtri economici per immagine: presenza di almeno un'istanza escooter,
   nessuna istanza escooter troppo grande (primo piano), immagine non troppo
   piccola;
2. deduplicazione cross-dataset per contenuto (perceptual hash), a
   differenza di dedupe_augmented.py che opera solo entro un dataset;
3. filtro varietà: almeno un'istanza di una classe COCO (diversa da
   escooter) rilevata da un modello YOLO pretrained.

Non copia immagini: scrive un file di testo con un path per riga (relativo a
data/interim/) delle immagini candidate, e un log con il motivo di scarto di
ogni immagine esclusa. La copia in un dataset di unione è uno step separato.
"""
import argparse
from pathlib import Path

import imagehash
import numpy as np
import yaml
from PIL import Image

import dataset_index

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
SPLITS = ("train", "valid", "test")

CLOSEUP_AREA_THRESHOLD = 0.8  # area relativa (w*h) oltre la quale un'istanza escooter è "primo piano"
MIN_PIXELS = 160_000  # dimensione minima immagine (larghezza*altezza)
PHASH_DISTANCE_THRESHOLD = 5  # distanza di Hamming del perceptual hash sotto la quale due immagini sono quasi-duplicati
COCO_MODEL = "yolo11x.pt"
COCO_BATCH_SIZE = 16


def load_datasets() -> list[dict]:
    return [e for e in dataset_index.load_index() if e.get("dedup_dir")]


def escooter_class_indices(entry: dict, names: list[str]) -> set:
    wanted = set(entry.get("escooter_class_names") or [])
    return {i for i, name in enumerate(names) if name in wanted}


def read_boxes(label_path: Path) -> list:
    """Ritorna le righe di annotazione come (classe, xc, yc, w, h). Il dataset
    interim è già stato normalizzato a bbox da dedupe_augmented.py."""
    if not label_path.exists():
        return []
    boxes = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cls = int(parts[0])
        xc, yc, w, h = map(float, parts[1:])
        boxes.append((cls, xc, yc, w, h))
    return boxes


def iter_candidate_images(datasets: list[dict]):
    """Genera un dict per ogni immagine di ogni dataset deduplicato, con i
    bounding box già separati in escooter/altre classi."""
    repo_root = DATA_ROOT.parent
    for entry in datasets:
        dedup_dir = repo_root / entry["dedup_dir"]
        data_yaml_path = dedup_dir / "data.yaml"
        if not data_yaml_path.exists():
            continue
        names = yaml.safe_load(data_yaml_path.read_text()).get("names", [])
        escooter_idx = escooter_class_indices(entry, names)

        for split in SPLITS:
            img_dir = dedup_dir / split / "images"
            lbl_dir = dedup_dir / split / "labels"
            if not img_dir.exists():
                continue
            for img_path in sorted(img_dir.iterdir()):
                boxes = read_boxes(lbl_dir / (img_path.stem + ".txt"))
                yield {
                    "dataset_id": entry["id"],
                    "split": split,
                    "image_path": img_path,
                    "escooter_boxes": [b for b in boxes if b[0] in escooter_idx],
                    "other_boxes": [b for b in boxes if b[0] not in escooter_idx],
                }


def rel_label(item: dict) -> str:
    return f"{item['dataset_id']}/{item['split']}/images/{item['image_path'].name}"


def apply_cheap_filters(items: list, log: list) -> list:
    survivors = []
    for item in items:
        if not item["escooter_boxes"]:
            log.append(f"SCARTATA {rel_label(item)}  (nessuna istanza escooter)")
            continue
        if any(w * h >= CLOSEUP_AREA_THRESHOLD for _, _, _, w, h in item["escooter_boxes"]):
            log.append(f"SCARTATA {rel_label(item)}  (istanza escooter troppo grande, primo piano)")
            continue
        with Image.open(item["image_path"]) as im:
            w_px, h_px = im.size
        if w_px * h_px < MIN_PIXELS:
            log.append(f"SCARTATA {rel_label(item)}  (immagine troppo piccola: {w_px}x{h_px})")
            continue
        survivors.append(item)
    return survivors


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def phash_ints(items: list) -> np.ndarray:
    """Perceptual hash di ogni immagine, come interi a 64 bit (per confronto vettorizzato)."""
    values = np.empty(len(items), dtype=np.uint64)
    for i, item in enumerate(items):
        with Image.open(item["image_path"]) as im:
            values[i] = np.uint64(int(str(imagehash.phash(im)), 16))
    return values


def cluster_near_duplicates(hashes: np.ndarray, threshold: int, chunk_size: int = 1000) -> UnionFind:
    """Raggruppa gli indici con distanza di Hamming <= threshold, a blocchi
    per contenere la memoria (matrice N x N di distanze evitata in un colpo solo)."""
    n = len(hashes)
    uf = UnionFind(n)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        block = hashes[start:end]
        xor = block[:, None] ^ hashes[None, :]
        dist = np.bitwise_count(xor)
        close_i, close_j = np.where(dist <= threshold)
        for bi, j in zip(close_i, close_j):
            i = start + bi
            if i < j:
                uf.union(int(i), int(j))
    return uf


def dedupe_cross_dataset(items: list, log: list) -> list:
    if not items:
        return items
    print(f"Calcolo perceptual hash su {len(items)} immagini...")
    hashes = phash_ints(items)
    uf = cluster_near_duplicates(hashes, PHASH_DISTANCE_THRESHOLD)

    groups: dict = {}
    for i in range(len(items)):
        groups.setdefault(uf.find(i), []).append(i)

    survivors = []
    for idxs in groups.values():
        if len(idxs) == 1:
            survivors.append(items[idxs[0]])
            continue
        best = max(idxs, key=lambda i: len(items[i]["escooter_boxes"]) + len(items[i]["other_boxes"]))
        for i in idxs:
            if i == best:
                continue
            log.append(f"SCARTATA {rel_label(items[i])}  (quasi-duplicato di {rel_label(items[best])})")
        survivors.append(items[best])
    return survivors


def apply_variety_filter(items: list, log: list, limit: int | None = None) -> list:
    """Scarta le immagini in cui il modello YOLO pretrained COCO non rileva
    nessuna istanza (COCO non ha una classe escooter, quindi ogni rilevamento
    conta come 'varietà')."""
    from ultralytics import YOLO

    model = YOLO(COCO_MODEL)
    subset = items[:limit] if limit else items
    survivors = []
    for start in range(0, len(subset), COCO_BATCH_SIZE):
        batch = subset[start:start + COCO_BATCH_SIZE]
        paths = [str(it["image_path"]) for it in batch]
        results = model.predict(paths, verbose=False)
        for item, result in zip(batch, results):
            n_detected = 0 if result.boxes is None else len(result.boxes)
            if n_detected == 0:
                log.append(f"SCARTATA {rel_label(item)}  (nessuna istanza COCO rilevata)")
                continue
            survivors.append(item)
    return survivors


CANDIDATES_PATH = DATA_ROOT / "selected_images.txt"
LOG_PATH = DATA_ROOT / "logs" / "select_images.log"


def write_outputs(survivors: list, log: list, stage: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text(
        f"# Log selezione immagini (stadio eseguito: {stage})\n\n"
        f"## Immagini scartate ({len(log)})\n" + "\n".join(log) + "\n"
    )
    CANDIDATES_PATH.write_text(
        f"# Immagini candidate dopo lo stadio '{stage}' — un path per riga, relativo a data/interim/\n"
        + "\n".join(rel_label(item) for item in survivors) + "\n"
    )
    print(f"Candidati scritti in {CANDIDATES_PATH} ({len(survivors)} immagini)")
    print(f"Log scritto in {LOG_PATH}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["cheap", "dedup", "variety", "all"], default="all",
                         help="Fino a quale stadio eseguire (per testare incrementalmente); default: tutti")
    parser.add_argument("--limit", type=int, default=None,
                         help="Limita lo stadio variety alle prime N immagini sopravvissute (per test su campione)")
    args = parser.parse_args()

    datasets = load_datasets()
    print(f"{len(datasets)} dataset deduplicati trovati nell'indice.")

    items = list(iter_candidate_images(datasets))
    print(f"{len(items)} immagini totali nel pool di partenza.")

    log: list = []
    survivors = apply_cheap_filters(items, log)
    print(f"Dopo i filtri economici: {len(survivors)}/{len(items)} immagini sopravvissute "
          f"({len(items) - len(survivors)} scartate).")

    if args.stage == "cheap":
        write_outputs(survivors, log, args.stage)
        return

    before = len(survivors)
    survivors = dedupe_cross_dataset(survivors, log)
    print(f"Dopo la dedup cross-dataset: {len(survivors)}/{before} immagini sopravvissute "
          f"({before - len(survivors)} quasi-duplicati scartati).")

    if args.stage == "dedup":
        write_outputs(survivors, log, args.stage)
        return

    before = len(survivors)
    survivors = apply_variety_filter(survivors, log, limit=args.limit)
    print(f"Dopo il filtro varietà COCO: {len(survivors)}/{before} immagini sopravvissute "
          f"({before - len(survivors)} scartate).")
    write_outputs(survivors, log, args.stage)


if __name__ == "__main__":
    main()
