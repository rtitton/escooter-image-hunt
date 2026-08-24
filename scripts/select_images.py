#!/usr/bin/env python3
"""Seleziona le immagini "migliori" tra i dataset deduplicati in data/interim.

Criteri (vedi claude-instruct-01-automatic-image-selection.md):
1. filtri economici per immagine: presenza di almeno un'istanza escooter,
   nessuna istanza escooter troppo grande (primo piano), immagine non troppo
   piccola;
2. deduplicazione cross-dataset per contenuto (perceptual hash), a
   differenza di dedupe_augmented.py che opera solo entro un dataset;
3. filtro varietà: almeno un'istanza di una classe COCO (diversa da
   escooter) rilevata da un modello YOLO pretrained;
4. scarto delle immagini in cui una bbox escooter include il conducente:
   alcuni dataset sorgente annotano l'intera persona invece del solo
   monopattino, il che confonderebbe il training rispetto alla classe
   "persona". Si stima quanta parte di ogni bbox escooter è spiegata da una
   persona rilevata dal modello COCO. Il controllo è ripetuto su tutte e 4
   le orientazioni (0/90/180/270°) e non solo su quella originale: alcune
   immagini sorgente sono ruotate/flippate rispetto al contenuto reale (una
   persona in piedi appare sdraiata nel frame), e un rilevatore addestrato
   su foto diritte spesso manca la persona in quell'orientazione — un
   controllo euristico più economico (basato sul padding nero tipico delle
   immagini ruotate) si è rivelato inaffidabile su questi casi, da qui la
   scelta di controllare sempre tutte le orientazioni invece di provare a
   indovinare quali immagini ne hanno bisogno. Questo quadruplica il costo
   GPU del filtro varietà, compensato usando un modello più piccolo
   (yolo11l invece di yolo11x). Lo scarto è sull'immagine intera anche se
   una sola bbox è contaminata: escludere solo quella bbox lascerebbe
   nell'immagine un monopattino visibile ma non annotato, un falso negativo
   che confonderebbe il training almeno quanto il problema che si vuole
   risolvere.

Non copia immagini: scrive un file di testo con un path per riga (relativo a
data/interim/) delle immagini candidate, e un log con il motivo di scarto di
ogni immagine esclusa. Le immagini scartate per conducente incluso finiscono
in un elenco separato (data/flagged_rider_contamination.txt), da rivedere/
correggere manualmente in un secondo momento invece di buttarle: hanno
comunque superato tutti gli altri criteri di qualità. La copia effettiva
delle immagini (candidate o flaggate) in una cartella è uno step separato
(build_union_dataset.py).
"""
import argparse
import json
from pathlib import Path

import imagehash
import numpy as np
import yaml
from PIL import Image

import dataset_index

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
SPLITS = ("train", "valid", "test")

CLOSEUP_AREA_THRESHOLD = 0.4  # area relativa (w*h) oltre la quale un'istanza escooter è "primo piano"
MIN_PIXELS = 160_000  # dimensione minima immagine (larghezza*altezza)
PHASH_DISTANCE_THRESHOLD = 8  # distanza di Hamming del perceptual hash sotto la quale due immagini sono quasi-duplicati
COCO_MODEL = "yolo11l.pt"
COCO_BATCH_SIZE = 16
ROTATIONS = {0: None, 90: Image.ROTATE_90, 180: Image.ROTATE_180, 270: Image.ROTATE_270}
PERSON_CLASS_ID = 0  # classe "person" in COCO
RIDER_OVERLAP_THRESHOLD = 0.5  # frazione dell'area della bbox escooter coperta da una detection "persona" oltre la quale l'annotazione probabilmente include il conducente
VARIETY_CACHE_PATH = DATA_ROOT / "cache" / "variety_filter_cache.json"
VARIETY_CACHE_SAVE_EVERY = 20  # batch tra un salvataggio incrementale della cache e il successivo


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


def escooter_box_pixel_xyxy(box: tuple, w_px: int, h_px: int) -> tuple:
    _, xc, yc, w, h = box
    return (
        (xc - w / 2) * w_px,
        (yc - h / 2) * h_px,
        (xc + w / 2) * w_px,
        (yc + h / 2) * h_px,
    )


def containment_ratio(box_xyxy: tuple, other_xyxy: tuple) -> float:
    """Frazione dell'area di box_xyxy coperta dall'intersezione con other_xyxy."""
    x1, y1 = max(box_xyxy[0], other_xyxy[0]), max(box_xyxy[1], other_xyxy[1])
    x2, y2 = min(box_xyxy[2], other_xyxy[2]), min(box_xyxy[3], other_xyxy[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area = (box_xyxy[2] - box_xyxy[0]) * (box_xyxy[3] - box_xyxy[1])
    return inter / area if area > 0 else 0.0


def rotate_point(x: float, y: float, w: int, h: int, const) -> tuple:
    """Coordinate di (x, y) — punto in un'immagine w x h — dopo
    Image.transpose(const). Formule verificate empiricamente (vedi PIL
    Image.ROTATE_90/180/270)."""
    if const is None:
        return x, y
    if const == Image.ROTATE_90:
        return y, w - 1 - x
    if const == Image.ROTATE_180:
        return w - 1 - x, h - 1 - y
    if const == Image.ROTATE_270:
        return h - 1 - y, x
    raise ValueError(const)


def rotate_box(box_xyxy: tuple, w: int, h: int, const) -> tuple:
    x1, y1, x2, y2 = box_xyxy
    corners = [rotate_point(x, y, w, h, const) for x, y in ((x1, y1), (x2, y1), (x1, y2), (x2, y2))]
    xs, ys = [c[0] for c in corners], [c[1] for c in corners]
    return (min(xs), min(ys), max(xs), max(ys))


def variety_cache_key(item: dict) -> str:
    return rel_label(item)


def load_variety_cache(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def save_variety_cache(cache: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2))


def variety_cache_is_valid(entry: dict, image_path: Path) -> bool:
    """Un'immagine può cambiare contenuto pur mantenendo lo stesso path se il
    dataset viene rigenerato (es. dedupe_augmented sceglie un "keeper" diverso
    tra una run e l'altra): si invalida la entry se mtime/dimensione del file
    non corrispondono più a quelli registrati al momento dell'inferenza."""
    stat = image_path.stat()
    return entry.get("mtime") == stat.st_mtime and entry.get("size") == stat.st_size


def detections_from_result(result) -> list:
    """Estrae le detection di un risultato YOLO come liste JSON-serializzabili
    [classe, x1, y1, x2, y2]."""
    if result.boxes is None:
        return []
    return [
        [int(cls), *xyxy]
        for cls, xyxy in zip(result.boxes.cls.tolist(), result.boxes.xyxy.tolist())
    ]


def apply_variety_filter(
    items: list, log: list, limit: int | None = None, refresh_cache: bool = False,
    cache_path: Path = VARIETY_CACHE_PATH,
) -> tuple[list, list]:
    """Scarta le immagini in cui il modello YOLO pretrained COCO non rileva
    nessuna istanza nell'orientazione originale (COCO non ha una classe
    escooter, quindi ogni rilevamento conta come 'varietà'), e separa quelle
    in cui almeno una bbox escooter è per lo più coperta da una detection
    'persona' (probabile conducente incluso nell'annotazione sorgente).
    Il controllo di sovrapposizione persona/bbox è ripetuto su tutte e 4 le
    orientazioni (0/90/180/270°), non solo quella originale, perché alcune
    immagini sorgente sono ruotate/flippate e un rilevatore addestrato su
    foto diritte spesso manca la persona in quell'orientazione. L'intera
    immagine viene esclusa anche se una sola bbox è contaminata, per non
    lasciare monopattini visibili ma non annotati. Ritorna (sopravvissute,
    flaggate per conducente incluso).

    Le detection YOLO (tutte e 4 le orientazioni) sono cache su disco per
    (dataset, split, nome immagine): rigirare la pipeline dopo aver aggiunto
    un dataset non richiede più di rifare l'inferenza sulle immagini già
    processate in una run precedente."""
    subset = items[:limit] if limit else items
    cache = load_variety_cache(cache_path)

    if refresh_cache:
        to_infer = subset
    else:
        to_infer = [it for it in subset
                    if variety_cache_key(it) not in cache
                    or not variety_cache_is_valid(cache[variety_cache_key(it)], it["image_path"])]

    if to_infer:
        from tqdm import tqdm
        from ultralytics import YOLO

        print(f"Filtro varietà: {len(subset) - len(to_infer)}/{len(subset)} immagini già in cache, "
              f"inferenza YOLO su {len(to_infer)}.")
        model = YOLO(COCO_MODEL)
        try:
            for batch_i, start in enumerate(
                tqdm(range(0, len(to_infer), COCO_BATCH_SIZE), desc="filtro varietà (inferenza)", unit="batch")
            ):
                batch = to_infer[start:start + COCO_BATCH_SIZE]
                images = [Image.open(it["image_path"]).convert("RGB") for it in batch]

                results_by_deg = {}
                for deg, const in ROTATIONS.items():
                    variants = images if const is None else [im.transpose(const) for im in images]
                    results_by_deg[deg] = model.predict(variants, verbose=False)

                for im in images:
                    im.close()

                for i, item in enumerate(batch):
                    result0 = results_by_deg[0][i]
                    h_px, w_px = result0.orig_shape
                    stat = item["image_path"].stat()
                    cache[variety_cache_key(item)] = {
                        "mtime": stat.st_mtime,
                        "size": stat.st_size,
                        "orig_shape": [h_px, w_px],
                        "detections": {
                            str(deg): detections_from_result(results_by_deg[deg][i]) for deg in ROTATIONS
                        },
                    }

                if (batch_i + 1) % VARIETY_CACHE_SAVE_EVERY == 0:
                    save_variety_cache(cache, cache_path)
        finally:
            save_variety_cache(cache, cache_path)

    survivors = []
    flagged_rider = []
    for item in subset:
        entry = cache[variety_cache_key(item)]
        h_px, w_px = entry["orig_shape"]
        detections_by_deg = {int(deg): dets for deg, dets in entry["detections"].items()}

        if not detections_by_deg[0]:
            log.append(f"SCARTATA {rel_label(item)}  (nessuna istanza COCO rilevata)")
            continue

        contaminated = False
        for deg, const in ROTATIONS.items():
            person_boxes = [tuple(det[1:]) for det in detections_by_deg[deg] if det[0] == PERSON_CLASS_ID]
            if not person_boxes:
                continue
            for esc in item["escooter_boxes"]:
                esc_xyxy = rotate_box(escooter_box_pixel_xyxy(esc, w_px, h_px), w_px, h_px, const)
                if max(containment_ratio(esc_xyxy, p) for p in person_boxes) >= RIDER_OVERLAP_THRESHOLD:
                    contaminated = True
                    break
            if contaminated:
                break

        if contaminated:
            log.append(f"FLAGGATA {rel_label(item)}  (almeno una bbox escooter include probabilmente il conducente)")
            flagged_rider.append(item)
            continue

        survivors.append(item)
    return survivors, flagged_rider


CANDIDATES_PATH = DATA_ROOT / "selected_images.txt"
FLAGGED_RIDER_PATH = DATA_ROOT / "flagged_rider_contamination.txt"
LOG_PATH = DATA_ROOT / "logs" / "select_images.log"


def write_flagged(flagged: list) -> None:
    FLAGGED_RIDER_PATH.write_text(
        "# Immagini con bbox escooter che includono probabilmente il conducente "
        "(sovrapposizione con una detection 'persona' del modello COCO) — un path per riga, "
        "relativo a data/interim/. Hanno superato tutti gli altri criteri di qualità: da "
        "rivedere/correggere manualmente (vedi build_union_dataset.py --candidates-file), non da buttare.\n"
        + "\n".join(rel_label(item) for item in flagged) + "\n"
    )
    print(f"Immagini flaggate (conducente incluso) scritte in {FLAGGED_RIDER_PATH} ({len(flagged)} immagini)")


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
    parser.add_argument("--refresh-variety-cache", action="store_true",
                         help="Ignora la cache delle detection YOLO dello stadio variety e ricalcola tutto da zero")
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
    survivors, flagged_rider = apply_variety_filter(
        survivors, log, limit=args.limit, refresh_cache=args.refresh_variety_cache
    )
    print(f"Dopo il filtro varietà COCO: {len(survivors)}/{before} immagini sopravvissute "
          f"({before - len(survivors)} scartate, di cui {len(flagged_rider)} per conducente incluso in una bbox).")
    write_outputs(survivors, log, args.stage)
    write_flagged(flagged_rider)


if __name__ == "__main__":
    main()
