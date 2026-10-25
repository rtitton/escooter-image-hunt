#!/usr/bin/env python3
"""Trova immagini quasi-duplicate confrontando le detection YOLO ORIGINALI
(cache di annotate_coco_classes.py, --cache-path: le detection grezze di
yolo11l a conf>=0.1, prima di qualunque filtro o correzione manuale) —
un segnale indipendente dal pHash e dalle modifiche fatte finora in
review_app.py, utile quando il pHash non basta a intercettare un doppione.

Criterio (deliberatamente semplice, pensato per i doppioni plateali, non
per casi dubbi): due box, della STESSA classe, "si sovrappongono" se la
containment reciproca (intersezione / area di ciascuna, in entrambe le
direzioni) è >= --containment. Due immagini finiscono nello stesso gruppo
se hanno almeno --min-matches coppie di box così accoppiate (accoppiamento
greedy per containment decrescente, ogni box usata al più una volta). I
gruppi sono le componenti connesse del grafo delle coppie che soddisfano
il criterio (stesso approccio di dedupe_phash.py).

Le box sono normalizzate alla propria risoluzione (0..1, stirate al
quadrato unitario come il resto della pipeline, nessun letterbox) prima del
confronto: è il modo in cui "si scalano le due immagini alla stessa
risoluzione" senza bisogno di decodificare i pixel, usando solo i metadati
già in cache.

Non modifica né il dataset né le decisioni di review_app.py: scrive solo
un elenco di gruppi in --out-path, nello stesso formato a righe di
flagged_coco_overlap.txt (un'immagine per riga, gruppi separati da riga
vuota), per incrociarlo a mano.

Avvio:
    python3 scripts/find_detection_duplicates.py <dataset_dir>
        [--cache-path FILE] [--containment 0.9] [--min-matches 4] [--out-path FILE]
"""
import argparse
import itertools
from collections import defaultdict
from pathlib import Path

import config


def load_cache(cache_path: Path) -> dict:
    import json
    cache = json.loads(cache_path.read_text())
    cache.pop("__class_names__", None)
    return cache


def normalized_boxes_by_class(entry: dict) -> dict:
    """entry della cache -> {classe: [[x1,y1,x2,y2], ...]} normalizzati 0..1
    sulla risoluzione propria dell'immagine (orig_shape = [h, w])."""
    h, w = entry["orig_shape"]
    by_class = defaultdict(list)
    for cls, _conf, x1, y1, x2, y2 in entry["detections"]:
        by_class[int(cls)].append([x1 / w, y1 / h, x2 / w, y2 / h])
    return by_class


def containment(a: list, b: list) -> float:
    """Frazione dell'area di a coperta dall'intersezione con b (v.
    containment_ratio in annotate_coco_classes.py, stessa definizione)."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area = (a[2] - a[0]) * (a[3] - a[1])
    return inter / area if area > 0 else 0.0


def mutual_containment(a: list, b: list) -> float:
    """Il minimo delle due containment (a coperta da b, b coperta da a): per
    essere "la stessa box" serve che si coprano a vicenda, non solo che una
    contenga l'altra (altrimenti una box enorme "conterrebbe" qualunque box
    piccola al suo interno)."""
    return min(containment(a, b), containment(b, a))


def n_matches(boxes_a: dict, boxes_b: dict, threshold: float) -> int:
    """Numero di coppie di box accoppiate (stessa classe, mutual_containment
    >= threshold), accoppiamento greedy per containment decrescente, ogni
    box usata al più una volta per lato."""
    total = 0
    for cls in boxes_a.keys() & boxes_b.keys():
        A, B = boxes_a[cls], boxes_b[cls]
        pairs = []
        for i, a in enumerate(A):
            for j, b in enumerate(B):
                v = mutual_containment(a, b)
                if v >= threshold:
                    pairs.append((v, i, j))
        pairs.sort(reverse=True)
        used_a, used_b = set(), set()
        for v, i, j in pairs:
            if i in used_a or j in used_b:
                continue
            used_a.add(i)
            used_b.add(j)
            total += 1
    return total


def find_groups(cache: dict, names: list, min_matches: int, containment_thr: float) -> list:
    """Coppie candidate limitate alle immagini che condividono almeno una
    classe (indice inverso classe->immagini): evita di confrontare coppie
    che non potrebbero comunque raggiungere min_matches>=1, senza cambiare
    il risultato rispetto a un confronto completo di tutte le coppie."""
    boxes_by_name = {n: normalized_boxes_by_class(cache[n]) for n in names}
    by_class_images = defaultdict(set)
    for n, by_class in boxes_by_name.items():
        for cls in by_class:
            by_class_images[cls].add(n)

    candidate_pairs = set()
    for imgs in by_class_images.values():
        imgs = sorted(imgs)
        for a, b in itertools.combinations(imgs, 2):
            candidate_pairs.add((a, b))

    adjacency = defaultdict(set)
    n_checked = 0
    for a, b in candidate_pairs:
        n_checked += 1
        if n_matches(boxes_by_name[a], boxes_by_name[b], containment_thr) >= min_matches:
            adjacency[a].add(b)
            adjacency[b].add(a)

    # componenti connesse
    seen = set()
    groups = []
    for n in names:
        if n in seen or n not in adjacency:
            continue
        stack, comp = [n], set()
        while stack:
            x = stack.pop()
            if x in comp:
                continue
            comp.add(x)
            stack.extend(adjacency[x] - comp)
        seen |= comp
        groups.append(sorted(comp))
    groups.sort(key=len, reverse=True)
    print(f"coppie di immagini con almeno una classe in comune (controllate): {n_checked}")
    return groups


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset_dir", type=Path, help="Cartella del dataset (usa la sottocartella images/)")
    ap.add_argument("--cache-path", type=Path, default=config.COCO_ANNOTATION_CACHE_PATH)
    ap.add_argument("--containment", type=float, default=config.DUPLICATE_DETECTION_CONTAINMENT)
    ap.add_argument("--min-matches", type=int, default=config.DUPLICATE_DETECTION_MIN_MATCHES)
    ap.add_argument("--out-path", type=Path, default=config.FLAGGED_DETECTION_DUPLICATES_PATH)
    ap.add_argument("--limit", type=int, default=None, help="limita alle prime N immagini (per prove rapide)")
    args = ap.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    cache = load_cache(args.cache_path)
    names = sorted(p.name for p in (dataset_dir / "images").iterdir())
    missing = [n for n in names if n not in cache]
    names = [n for n in names if n in cache]
    if args.limit:
        names = names[:args.limit]
    print(f"immagini nel dataset: {len(names) + len(missing)}, in cache: {len(names)}, "
          f"assenti dalla cache (saltate): {len(missing)}")
    print(f"soglia containment: {args.containment}, min-matches: {args.min_matches}")

    groups = find_groups(cache, names, args.min_matches, args.containment)
    n_grouped = sum(len(g) for g in groups)
    print(f"gruppi trovati: {len(groups)}, immagini coinvolte: {n_grouped}/{len(names)}")

    lines = [
        f"# Gruppi di immagini quasi-duplicate per detection YOLO originali (v. find_detection_duplicates.py)",
        f"# soglia containment={args.containment}  min_matches={args.min_matches}  "
        f"sorgente cache={args.cache_path}",
        f"# {len(groups)} gruppi, {n_grouped} immagini coinvolte su {len(names)}",
    ]
    for i, g in enumerate(groups, 1):
        lines.append(f"\n# gruppo {i} ({len(g)} immagini)")
        lines.extend(g)
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    args.out_path.write_text("\n".join(lines) + "\n")
    print(f"scritto: {args.out_path}")


if __name__ == "__main__":
    main()
