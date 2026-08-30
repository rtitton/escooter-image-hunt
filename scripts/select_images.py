#!/usr/bin/env python3
"""Seleziona le immagini "migliori" tra i dataset deduplicati in data/interim.

Criteri (vedi claude-instruct-01-automatic-image-selection.md):
1. filtri economici per immagine: presenza di almeno un'istanza escooter,
   nessuna istanza escooter troppo grande (primo piano) né troppo piccola
   (lontana), immagine non troppo piccola;
1-bis. (opzionale, --temporal-dedup) dedup temporale: assottiglia le
   sequenze di frame consecutivi estratti dallo stesso video (nomi tipo
   frame_00000, frame_00010), tenendo solo i keyframe. Eseguita fra i filtri
   economici e la dedup cross-dataset; v. dedupe_temporal();
2. deduplicazione cross-dataset per contenuto (perceptual hash), a
   differenza di dedupe_augmented.py che opera solo entro un dataset. I
   perceptual hash sono cache su disco per (dataset, split, nome immagine),
   come le detection YOLO dello stadio variety (v. sotto): rigirare la
   pipeline dopo aver aggiunto un dataset non richiede più di rihashare le
   immagini già processate in una run precedente;
3. filtro varietà: almeno VARIETY_MIN_INSTANCES istanze (default 1) di
   classi COCO (diverse da escooter) rilevate da un modello YOLO pretrained;
4. scarto delle immagini in cui una bbox escooter include il conducente:
   alcuni dataset sorgente annotano l'intera persona invece del solo
   monopattino, il che confonderebbe il training rispetto alla classe
   "persona". Si considera contaminata una coppia di bbox escooter/persona
   che si sovrappongono per almeno RIDER_OVERLAP_THRESHOLD (frazione
   dell'area della bbox escooter coperta dall'intersezione, v.
   containment_ratio()) e in cui la bbox escooter è più alta della bbox
   persona (rapporto fra le due altezze > RIDER_HEIGHT_RATIO_THRESHOLD):
   un'annotazione che include il conducente supera in altezza quella, più
   accurata, stimata dal modello COCO per la sola persona; il precondition
   di sovrapposizione evita di confrontare le altezze di una bbox escooter
   e di una persona che si trovano in punti diversi dell'immagine e non
   hanno nulla a che vedere fra loro. Il controllo è ripetuto su tutte e 4 le orientazioni (0/90/180/270°) e non
   solo su quella originale: alcune immagini sorgente sono ruotate/flippate
   rispetto al contenuto reale (una persona in piedi appare sdraiata nel
   frame), e un rilevatore addestrato su foto diritte spesso manca la
   persona in quell'orientazione — un controllo euristico più economico
   (basato sul padding nero tipico delle immagini ruotate) si è rivelato
   inaffidabile su questi casi, da qui la scelta di controllare sempre
   tutte le orientazioni invece di provare a indovinare quali immagini ne
   hanno bisogno. Questo quadruplica il costo GPU del filtro varietà,
   compensato usando un modello più piccolo (yolo11l invece di yolo11x).
   Lo scarto è sull'immagine intera anche se una sola bbox è contaminata:
   escludere solo quella bbox lascerebbe nell'immagine un monopattino
   visibile ma non annotato, un falso negativo che confonderebbe il
   training almeno quanto il problema che si vuole risolvere.

Le soglie CLOSEUP_AREA_THRESHOLD e FARAWAY_AREA_THRESHOLD (filtro 1),
VARIETY_MIN_INSTANCES (filtro 3) e PHASH_DISTANCE_THRESHOLD (filtro 2) sono di
default quelle di config/.env, ma possono essere sovrascritte per singolo
dataset dalle colonne omonime di datasets_to_download.csv (cella vuota o
"default" = valore di .env). Utile per trattare a parte sorgenti particolari,
p.es. footage con escooter piccoli (closeup più alto, faraway più basso,
varietà più permissiva) o molto ripetitiva (pHash più stretto). Per la dedup
cross-dataset la soglia di una coppia di immagini di dataset diversi è la più
stretta delle due (v. dataset_param() e cluster_near_duplicates()).

Non copia immagini: scrive un file di testo con un path per riga (relativo a
data/interim/) delle immagini candidate, e un log con il motivo di scarto di
ogni immagine esclusa. Le immagini scartate per conducente incluso finiscono
in un elenco separato (data/flagged_rider_contamination.txt), da rivedere/
correggere manualmente in un secondo momento invece di buttarle: hanno
comunque superato tutti gli altri criteri di qualità. Le immagini scartate
per soglia di area di una bbox escooter (troppo grande/primo piano o troppo
piccola/lontana) finiscono in un altro elenco separato
(data/flagged_area_threshold.txt), solo a scopo diagnostico. Per ogni
immagine esaminata (candidata o scartata) viene inoltre scritto un indice
(data/image_index.json) con i metadati principali (percorso completo,
dimensioni) e l'eventuale decisione di esclusione (stadio e motivo). La
copia effettiva delle immagini (candidate o flaggate) in una cartella è uno
step separato (build_union_dataset.py).
"""
import argparse
import json
import re
from pathlib import Path

import imagehash
import numpy as np
import yaml
from PIL import Image

import config
import dataset_index

DATA_ROOT = config.DATA_ROOT
SPLITS = ("train", "valid", "test")

CLOSEUP_AREA_THRESHOLD = config.CLOSEUP_AREA_THRESHOLD  # area relativa (w*h) oltre la quale un'istanza escooter è "primo piano"
FARAWAY_AREA_THRESHOLD = config.FARAWAY_AREA_THRESHOLD  # area relativa (w*h) sotto la quale un'istanza escooter è troppo piccola/lontana
MIN_PIXELS = config.MIN_PIXELS  # dimensione minima immagine (larghezza*altezza)
PHASH_DISTANCE_THRESHOLD = config.PHASH_DISTANCE_THRESHOLD  # distanza di Hamming del perceptual hash sotto la quale due immagini sono quasi-duplicati
TEMPORAL_MIN_SEQ = config.TEMPORAL_MIN_SEQ
TEMPORAL_KEEP_DISTANCE = config.TEMPORAL_KEEP_DISTANCE
TEMPORAL_MAX_GAP = config.TEMPORAL_MAX_GAP
COCO_MODEL = config.COCO_MODEL
COCO_BATCH_SIZE = config.COCO_BATCH_SIZE
# ROTATIONS = {0: None, 90: Image.ROTATE_90, 180: Image.ROTATE_180, 270: Image.ROTATE_270}
# ROTATIONS = {0: None, 180: Image.ROTATE_180}
ROTATIONS = {0: None}
PERSON_CLASS_ID = config.PERSON_CLASS_ID  # classe "person" in COCO
RIDER_OVERLAP_THRESHOLD = config.RIDER_OVERLAP_THRESHOLD  # frazione dell'area della bbox escooter coperta da una detection "persona" perché la coppia sia considerata (precondizione spaziale prima del confronto altezze)
RIDER_HEIGHT_RATIO_THRESHOLD = config.RIDER_HEIGHT_RATIO_THRESHOLD  # rapporto (altezza escooter / altezza persona) oltre il quale si considera il conducente incluso nell'annotazione
PHASH_CACHE_PATH = config.PHASH_CACHE_PATH
VARIETY_CACHE_PATH = config.VARIETY_CACHE_PATH
VARIETY_CACHE_SAVE_EVERY = config.VARIETY_CACHE_SAVE_EVERY  # batch tra un salvataggio incrementale della cache e il successivo
VARIETY_MIN_INSTANCES = config.VARIETY_MIN_INSTANCES  # istanze COCO minime nell'orientazione originale perché un'immagine sia di buona varietà

# Override per-dataset di CLOSEUP_AREA_THRESHOLD / PHASH_DISTANCE_THRESHOLD /
# VARIETY_MIN_INSTANCES, dalle colonne omonime di datasets_to_download.csv.
# Cella vuota o "default" -> vale il valore di config/.env qui sopra.
SELECTION_OVERRIDES = dataset_index.selection_overrides()


def dataset_param(dataset_id: str, column: str, default):
    """Valore effettivo di un parametro di selezione per un dataset: override
    dal CSV se presente, altrimenti il default di config/.env."""
    return SELECTION_OVERRIDES.get(dataset_id, {}).get(column, default)


def load_datasets() -> list[dict]:
    enabled = dataset_index.enabled_ids()
    print(f"enabled: {enabled}")
    return [e for e in dataset_index.load_index() if e.get("dedup_dir") and e["id"] in enabled]


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


def index_record(item: dict, w_px: int, h_px: int) -> dict:
    """Voce base dell'indice immagini: metadati principali (percorso completo,
    dimensioni) e nessuna decisione di esclusione ancora presa."""
    return {
        "dataset_id": item["dataset_id"],
        "split": item["split"],
        "image_path": str(item["image_path"]),
        "width": w_px,
        "height": h_px,
        "excluded": False,
        "exclusion_stage": None,
        "exclusion_reason": None,
    }


def mark_excluded(index: dict, item: dict, stage: str, reason: str) -> None:
    index[rel_label(item)].update(excluded=True, exclusion_stage=stage, exclusion_reason=reason)


def apply_cheap_filters(items: list, log: list, index: dict) -> tuple[list, list]:
    survivors = []
    flagged_area = []
    for item in items:
        with Image.open(item["image_path"]) as im:
            w_px, h_px = im.size
        index[rel_label(item)] = index_record(item, w_px, h_px)

        if not item["escooter_boxes"]:
            log.append(f"SCARTATA {rel_label(item)}  (nessuna istanza escooter)")
            mark_excluded(index, item, "cheap", "nessuna istanza escooter")
            continue
        closeup_thr = dataset_param(item["dataset_id"], "closeup_area_threshold", CLOSEUP_AREA_THRESHOLD)
        if any(w * h >= closeup_thr for _, _, _, w, h in item["escooter_boxes"]):
            log.append(f"SCARTATA {rel_label(item)}  (istanza escooter troppo grande, primo piano)")
            mark_excluded(index, item, "cheap", "istanza escooter troppo grande, primo piano")
            flagged_area.append(item)
            continue
        faraway_thr = dataset_param(item["dataset_id"], "faraway_area_threshold", FARAWAY_AREA_THRESHOLD)
        if any(w * h < faraway_thr for _, _, _, w, h in item["escooter_boxes"]):
            log.append(f"SCARTATA {rel_label(item)}  (istanza escooter troppo piccola, lontana)")
            mark_excluded(index, item, "cheap", "istanza escooter troppo piccola, lontana")
            flagged_area.append(item)
            continue
        if w_px * h_px < MIN_PIXELS:
            log.append(f"SCARTATA {rel_label(item)}  (immagine troppo piccola: {w_px}x{h_px})")
            mark_excluded(index, item, "cheap", f"immagine troppo piccola: {w_px}x{h_px}")
            continue
        survivors.append(item)
    return survivors, flagged_area


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


def phash_ints(items: list, cache_path: Path = PHASH_CACHE_PATH, refresh_cache: bool = False) -> np.ndarray:
    """Perceptual hash di ogni immagine, come interi a 64 bit (per confronto
    vettorizzato). Cache su disco per (dataset, split, nome immagine), come
    per le detection YOLO dello stadio variety: rigirare la pipeline dopo
    aver aggiunto un dataset non richiede più di rihashare le immagini già
    processate in una run precedente."""
    cache = load_json_cache(cache_path)

    if refresh_cache:
        to_hash = items
    else:
        to_hash = [it for it in items
                   if cache_key(it) not in cache
                   or not cache_entry_is_valid(cache[cache_key(it)], it["image_path"])]

    if to_hash:
        print(f"Perceptual hash: {len(items) - len(to_hash)}/{len(items)} immagini già in cache, "
              f"calcolo su {len(to_hash)}.")
        for item in to_hash:
            stat = item["image_path"].stat()
            with Image.open(item["image_path"]) as im:
                phash_hex = str(imagehash.phash(im))
            cache[cache_key(item)] = {"mtime": stat.st_mtime, "size": stat.st_size, "phash": phash_hex}
        save_json_cache(cache, cache_path)

    values = np.empty(len(items), dtype=np.uint64)
    for i, item in enumerate(items):
        values[i] = np.uint64(int(cache[cache_key(item)]["phash"], 16))
    return values


def cluster_near_duplicates(hashes: np.ndarray, thresholds: np.ndarray, chunk_size: int = 1000) -> UnionFind:
    """Raggruppa gli indici con distanza di Hamming abbastanza piccola, a
    blocchi per contenere la memoria (matrice N x N di distanze evitata in un
    colpo solo). `thresholds` è la soglia per-immagine (può variare per
    dataset, v. colonna phash_distance_threshold del CSV): due immagini sono
    quasi-duplicati solo se la loro distanza rientra nella soglia *più stretta*
    delle due (min), così un dataset con soglia bassa non viene assorbito in
    cluster ancorati a dataset più permissivi."""
    n = len(hashes)
    uf = UnionFind(n)
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        block = hashes[start:end]
        xor = block[:, None] ^ hashes[None, :]
        dist = np.bitwise_count(xor)
        pair_threshold = np.minimum(thresholds[start:end, None], thresholds[None, :])
        close_i, close_j = np.where(dist <= pair_threshold)
        for bi, j in zip(close_i, close_j):
            i = start + bi
            if i < j:
                uf.union(int(i), int(j))
    return uf


def dedupe_cross_dataset(
    items: list, log: list, index: dict, refresh_cache: bool = False, cache_path: Path = PHASH_CACHE_PATH,
) -> list:
    if not items:
        return items
    hashes = phash_ints(items, cache_path=cache_path, refresh_cache=refresh_cache)
    thresholds = np.array(
        [dataset_param(it["dataset_id"], "phash_distance_threshold", PHASH_DISTANCE_THRESHOLD) for it in items],
        dtype=np.int16,
    )
    uf = cluster_near_duplicates(hashes, thresholds)

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
            reason = f"quasi-duplicato di {rel_label(items[best])}"
            log.append(f"SCARTATA {rel_label(items[i])}  ({reason})")
            mark_excluded(index, items[i], "dedup", reason)
        survivors.append(items[best])
    return survivors


RF_SUFFIX_RE = re.compile(r"_(?:jpe?g|png)\.rf\.[0-9a-f]{32}$", re.IGNORECASE)
CLIP_FRAME_RE = re.compile(r"^(?P<clip>.*?)[ _-]?(?P<idx>\d+)$")
MAX_PLAUSIBLE_FRAME_IDX = 100_000_000  # oltre questo il "numero" nel nome è un timestamp, non un indice di frame


def clip_and_frame(image_path: Path) -> tuple[str, int | None]:
    """Da un nome file Roboflow (es. ``frame_00010_jpg.rf.<md5>.jpg``) ricava
    l'identità della ripresa di origine e l'indice del frame, quando il nome è
    una sequenza numerata. Ritorna ``(stem, None)`` se non lo è (nessun numero
    finale, o un numero così grande da essere un timestamp): in quel caso
    l'immagine non partecipa alla dedup temporale."""
    stem = RF_SUFFIX_RE.sub("", image_path.stem)
    m = CLIP_FRAME_RE.match(stem)
    if not m:
        return stem, None
    idx = int(m.group("idx"))
    if idx > MAX_PLAUSIBLE_FRAME_IDX:
        return stem, None
    return (m.group("clip") or "_", idx)


def dedupe_temporal(
    items: list, log: list, index: dict, refresh_cache: bool = False,
    cache_path: Path = PHASH_CACHE_PATH,
) -> list:
    """Assottiglia le sequenze di frame consecutivi estratti dallo stesso
    video: parecchi dataset sorgente sono campionamenti fitti di poche riprese
    (nomi tipo ``frame_00000``, ``frame_00010`` …). La dedup cross-dataset a
    soglia singola su questi frame o li tiene tutti (la catena di quasi-
    duplicati non si connette) o collassa un'intera panoramica a una sola
    immagine (catena transitiva).

    Qui invece le immagini sono raggruppate per ``(dataset, split, clip)`` —
    ``clip`` e indice di frame ricavati dal nome file — e ogni gruppo con
    almeno TEMPORAL_MIN_SEQ frame viene percorso in ordine di indice: un
    frame è scartato quando è visivamente vicino all'ultimo frame tenuto
    (Hamming del pHash < TEMPORAL_KEEP_DISTANCE) e a non più di
    TEMPORAL_MAX_GAP indici da esso. Primo e ultimo frame del gruppo si
    tengono sempre; superato TEMPORAL_MAX_GAP si tiene comunque il frame
    corrente, che diventa il nuovo riferimento. La somiglianza pHash è la
    vera salvaguardia contro i gruppi che sono in realtà raccolte di immagini
    distinte numerate (frame consecutivi non correlati → pHash lontano →
    tenuti). Sequenze sotto TEMPORAL_MIN_SEQ frame e nomi non numerati
    restano intatti e passano alla dedup cross-dataset.

    Opzionale: attivo solo con ``--temporal-dedup``. Riusa la cache dei
    perceptual hash dello stadio dedup.

    Nota: girando prima della dedup cross-dataset, che è un clustering a
    catena (single-linkage), questo filtro può far *aumentare* di poche
    unità il numero di candidate finali — rimuovendo un frame che faceva da
    ponte fra due immagini altrimenti non simili si spezza il suo cluster
    pHash in due, e ognuno dei due tiene il proprio rappresentante. È un
    effetto atteso: quei frame-ponte erano soppressi solo per transitività.
    """
    if not items:
        return items

    sequences: dict = {}
    for i, item in enumerate(items):
        clip, frame_idx = clip_and_frame(item["image_path"])
        if frame_idx is None:
            continue
        sequences.setdefault((item["dataset_id"], item["split"], clip), []).append((frame_idx, i))

    long_seqs = {k: v for k, v in sequences.items() if len(v) >= TEMPORAL_MIN_SEQ}
    if not long_seqs:
        print("Dedup temporale: nessuna sequenza numerata di almeno "
              f"{TEMPORAL_MIN_SEQ} frame, nessuno scarto.")
        return items

    hashes = phash_ints(items, cache_path=cache_path, refresh_cache=refresh_cache)

    dropped: set = set()
    for members in long_seqs.values():
        members.sort()
        kept_pos = 0
        kept_frame_idx = members[0][0]
        for k in range(1, len(members) - 1):  # primo e ultimo frame sempre tenuti
            frame_idx, i = members[k]
            kept_i = members[kept_pos][1]
            distance = int(np.bitwise_count(hashes[i] ^ hashes[kept_i]))
            similar = distance < TEMPORAL_KEEP_DISTANCE
            within_gap = (frame_idx - kept_frame_idx) <= TEMPORAL_MAX_GAP
            if similar and within_gap:
                dropped.add(i)
            else:
                kept_pos, kept_frame_idx = k, frame_idx

    survivors = []
    for i, item in enumerate(items):
        if i not in dropped:
            survivors.append(item)
            continue
        clip, _ = clip_and_frame(item["image_path"])
        reason = f"frame ridondante nella sequenza temporale '{clip}'"
        log.append(f"SCARTATA {rel_label(item)}  ({reason})")
        mark_excluded(index, item, "temporal", reason)
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


def cache_key(item: dict) -> str:
    return rel_label(item)


def load_json_cache(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def save_json_cache(cache: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2))


def cache_entry_is_valid(entry: dict, image_path: Path) -> bool:
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
    items: list, log: list, index: dict, limit: int | None = None, refresh_cache: bool = False,
    cache_path: Path = VARIETY_CACHE_PATH,
) -> tuple[list, list]:
    """Scarta le immagini in cui il modello YOLO pretrained COCO rileva meno
    di VARIETY_MIN_INSTANCES istanze nell'orientazione originale (COCO non ha
    una classe escooter, quindi ogni rilevamento conta come 'varietà'), e separa quelle
    in cui almeno una bbox escooter si sovrappone per almeno RIDER_OVERLAP_THRESHOLD
    a una detection 'persona' ed è più alta di RIDER_HEIGHT_RATIO_THRESHOLD
    volte quella persona (probabile conducente incluso nell'annotazione
    sorgente). Il controllo sovrapposizione+altezza escooter/persona è ripetuto su tutte
    e 4 le orientazioni (0/90/180/270°), non solo quella originale, perché
    alcune immagini sorgente sono ruotate/flippate e un rilevatore addestrato
    su foto diritte spesso manca la persona in quell'orientazione. L'intera
    immagine viene esclusa anche se una sola bbox è contaminata, per non
    lasciare monopattini visibili ma non annotati. Ritorna (sopravvissute,
    flaggate per conducente incluso).

    Le detection YOLO (tutte e 4 le orientazioni) sono cache su disco per
    (dataset, split, nome immagine): rigirare la pipeline dopo aver aggiunto
    un dataset non richiede più di rifare l'inferenza sulle immagini già
    processate in una run precedente."""
    subset = items[:limit] if limit else items
    cache = load_json_cache(cache_path)

    if refresh_cache:
        to_infer = subset
    else:
        to_infer = [it for it in subset
                    if cache_key(it) not in cache
                    or not cache_entry_is_valid(cache[cache_key(it)], it["image_path"])]

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
                    cache[cache_key(item)] = {
                        "mtime": stat.st_mtime,
                        "size": stat.st_size,
                        "orig_shape": [h_px, w_px],
                        "detections": {
                            str(deg): detections_from_result(results_by_deg[deg][i]) for deg in ROTATIONS
                        },
                    }

                if (batch_i + 1) % VARIETY_CACHE_SAVE_EVERY == 0:
                    save_json_cache(cache, cache_path)
        finally:
            save_json_cache(cache, cache_path)

    survivors = []
    flagged_rider = []
    for item in subset:
        entry = cache[cache_key(item)]
        h_px, w_px = entry["orig_shape"]
        detections_by_deg = {int(deg): dets for deg, dets in entry["detections"].items()}

        variety_min = dataset_param(item["dataset_id"], "variety_min_instances", VARIETY_MIN_INSTANCES)
        if len(detections_by_deg[0]) < variety_min:
            reason = f"{len(detections_by_deg[0])} istanze COCO rilevate, minimo richiesto {variety_min}"
            log.append(f"SCARTATA {rel_label(item)}  ({reason})")
            mark_excluded(index, item, "variety", reason)
            continue

        contaminated = False
        for deg, const in ROTATIONS.items():
            person_boxes = [tuple(det[1:]) for det in detections_by_deg[deg] if det[0] == PERSON_CLASS_ID]
            if not person_boxes:
                continue
            for esc in item["escooter_boxes"]:
                esc_xyxy = rotate_box(escooter_box_pixel_xyxy(esc, w_px, h_px), w_px, h_px, const)
                esc_height = esc_xyxy[3] - esc_xyxy[1]
                for p in person_boxes:
                    if containment_ratio(esc_xyxy, p) < RIDER_OVERLAP_THRESHOLD:
                        continue
                    if esc_height / (p[3] - p[1]) > RIDER_HEIGHT_RATIO_THRESHOLD:
                        contaminated = True
                        log.append(f"FLAGGATA {rel_label(item)}  (bbox escooter {esc_xyxy} sovrapposta a persona {p} in orientazione {deg}°)(altezza escooter {esc_height:.1f} / altezza persona {p[3] - p[1]:.1f} > {RIDER_HEIGHT_RATIO_THRESHOLD})")
                        break
                if contaminated:
                    break
            if contaminated:
                break

        if contaminated:
            reason = "almeno una bbox escooter include probabilmente il conducente"
            log.append(f"FLAGGATA {rel_label(item)}  ({reason})")
            mark_excluded(index, item, "variety", reason)
            flagged_rider.append(item)
            continue

        survivors.append(item)
    return survivors, flagged_rider


CANDIDATES_PATH = config.CANDIDATES_PATH
FLAGGED_RIDER_PATH = config.FLAGGED_RIDER_PATH
FLAGGED_AREA_PATH = config.FLAGGED_AREA_PATH
IMAGE_INDEX_PATH = config.IMAGE_INDEX_PATH
LOG_PATH = config.SELECT_IMAGES_LOG_PATH


def write_flagged(flagged: list) -> None:
    FLAGGED_RIDER_PATH.write_text(
        "# Immagini con bbox escooter che includono probabilmente il conducente "
        "(sovrapposizione con una detection 'persona' del modello COCO) — un path per riga, "
        "relativo a data/interim/. Hanno superato tutti gli altri criteri di qualità: da "
        "rivedere/correggere manualmente (vedi build_union_dataset.py --candidates-file), non da buttare.\n"
        + "\n".join(rel_label(item) for item in flagged) + "\n"
    )
    print(f"Immagini flaggate (conducente incluso) scritte in {FLAGGED_RIDER_PATH} ({len(flagged)} immagini)")


def write_flagged_area(flagged: list) -> None:
    FLAGGED_AREA_PATH.write_text(
        "# Immagini scartate per soglia di area di una bbox escooter (troppo grande/primo piano "
        "oltre CLOSEUP_AREA_THRESHOLD, o troppo piccola/lontana sotto FARAWAY_AREA_THRESHOLD) — "
        "un path per riga, relativo a data/interim/.\n"
        + "\n".join(rel_label(item) for item in flagged) + "\n"
    )
    print(f"Immagini flaggate (soglia area) scritte in {FLAGGED_AREA_PATH} ({len(flagged)} immagini)")


def write_image_index(index: dict) -> None:
    records = sorted(index.values(), key=lambda r: r["image_path"])
    IMAGE_INDEX_PATH.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n")
    print(f"Indice immagini scritto in {IMAGE_INDEX_PATH} ({len(records)} immagini)")


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
    parser.add_argument("--stage", choices=["cheap", "temporal", "dedup", "variety", "all"], default="all",
                         help="Fino a quale stadio eseguire (per testare incrementalmente); default: tutti. "
                              "Lo stadio 'temporal' è opzionale: viene eseguito solo con --temporal-dedup o --stage temporal")
    parser.add_argument("--temporal-dedup", action="store_true",
                         help="Abilita la dedup temporale (assottiglia le sequenze di frame consecutivi estratti "
                              "dallo stesso video) fra i filtri economici e la dedup cross-dataset. Disattivata di default")
    parser.add_argument("--limit", type=int, default=None,
                         help="Limita lo stadio variety alle prime N immagini sopravvissute (per test su campione)")
    parser.add_argument("--refresh-variety-cache", action="store_true",
                         help="Ignora la cache delle detection YOLO dello stadio variety e ricalcola tutto da zero")
    parser.add_argument("--refresh-phash-cache", action="store_true",
                         help="Ignora la cache dei perceptual hash dello stadio dedup e ricalcola tutto da zero")
    args = parser.parse_args()

    datasets = load_datasets()
    print(f"{len(datasets)} dataset abilitati e deduplicati trovati nell'indice.")

    items = list(iter_candidate_images(datasets))
    print(f"{len(items)} immagini totali nel pool di partenza.")

    log: list = []
    index: dict = {}
    survivors, flagged_area = apply_cheap_filters(items, log, index)
    print(f"Dopo i filtri economici: {len(survivors)}/{len(items)} immagini sopravvissute "
          f"({len(items) - len(survivors)} scartate, di cui {len(flagged_area)} per soglie di area).")
    write_flagged_area(flagged_area)

    if args.stage == "cheap":
        write_outputs(survivors, log, args.stage)
        write_image_index(index)
        return

    if args.temporal_dedup or args.stage == "temporal":
        before = len(survivors)
        survivors = dedupe_temporal(survivors, log, index, refresh_cache=args.refresh_phash_cache)
        print(f"Dopo la dedup temporale: {len(survivors)}/{before} immagini sopravvissute "
              f"({before - len(survivors)} frame ridondanti scartati).")

    if args.stage == "temporal":
        write_outputs(survivors, log, args.stage)
        write_image_index(index)
        return

    before = len(survivors)
    survivors = dedupe_cross_dataset(survivors, log, index, refresh_cache=args.refresh_phash_cache)
    print(f"Dopo la dedup cross-dataset: {len(survivors)}/{before} immagini sopravvissute "
          f"({before - len(survivors)} quasi-duplicati scartati).")

    if args.stage == "dedup":
        write_outputs(survivors, log, args.stage)
        write_image_index(index)
        return

    before = len(survivors)
    survivors, flagged_rider = apply_variety_filter(
        survivors, log, index, limit=args.limit, refresh_cache=args.refresh_variety_cache
    )
    print(f"Dopo il filtro varietà COCO: {len(survivors)}/{before} immagini sopravvissute "
          f"({before - len(survivors)} scartate, di cui {len(flagged_rider)} per conducente incluso in una bbox).")
    write_outputs(survivors, log, args.stage)
    write_flagged(flagged_rider)
    write_image_index(index)


if __name__ == "__main__":
    main()
