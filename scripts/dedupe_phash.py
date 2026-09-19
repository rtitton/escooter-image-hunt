#!/usr/bin/env python3
"""Deduplica per contenuto (perceptual hash) il dataset union_reviewed,
raggruppando in cluster le immagini a distanza di Hamming <= soglia e
tenendone solo una per cluster: quella con più bbox escooter annotate (a
parità, il nome file più piccolo, per determinismo).

A differenza della dedup cross-dataset di select_images.py — eseguita
*prima* della selezione/review, con soglia stretta (config.PHASH_DISTANCE_
THRESHOLD, per non perdere varietà) — questo script opera *dopo* la review
manuale, su union_reviewed/, con una soglia scelta liberamente da riga di
comando: utile per stringere la dedup a posteriori (es. per un dataset di
validazione senza quasi-duplicati) senza dover rifare selezione o review.

Le immagini escluse (quelle "in più" di ogni cluster) non vengono
cancellate: restano, insieme alle rispettive label, in --out-dir/excluded/,
per poterle ispezionare o recuperare. Per ogni immagine esclusa,
--out-dir/excluded/nearest_selected/ contiene una copia dell'immagine
tenuta (col nome dell'esclusa, per confrontarle una a fianco all'altra), e
--out-dir/excluded/manifest.csv elenca esclusa/tenuta/distanza pHash.

Controllo di plausibilità via detection COCO (--coco-check, attivo di
default): il pHash è una hash strutturale/di luminosità grossolana e può
avvicinare per errore due foto diverse ma composte in modo simile (persona
al centro, sfondo chiaro/scuro a metà, ecc.). Per ogni coppia
esclusa/tenuta, se entrambe le immagini hanno già una voce nella cache
delle detection COCO di select_images.py (ce l'hanno di sicuro se sono
passate per il filtro varietà, come tutte quelle in union_reviewed/), si
confrontano gli istogrammi di classe COCO rilevate (similarità di Jaccard
pesata sui conteggi): se troppo diverse (< --coco-similarity-threshold),
la coppia viene "salvata" — non trattata come duplicato — perché il
contenuto reale sembra diverso nonostante il pHash vicino. Le coppie
salvate così restano nell'output principale, non in excluded/, e sono
segnalate a console e nel log.

Avvio:
    python3 scripts/dedupe_phash.py [soglia]
        [--source-dir DIR] [--out-dir DIR] [--phash-cache FILE]
        [--coco-similarity-threshold T] [--no-coco-check]

[soglia] è la distanza di Hamming massima fra due pHash perché due
immagini siano considerate quasi-duplicate (default: 10).
"""
import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np

import config
import review_app  # riusa compute_phashes() (cache mtime/size su disco)
import select_images  # riusa UnionFind e cluster_near_duplicates()


def count_boxes(lbl_path: Path) -> int:
    if not lbl_path.exists():
        return 0
    return sum(1 for line in lbl_path.read_text().splitlines() if line.split())


def dataset_of(name: str) -> str:
    i = name.find("__")
    return name[:i] if i != -1 else name


def copy_pair(name: str, src_images: Path, src_labels: Path, dst_images: Path, dst_labels: Path) -> None:
    shutil.copy2(src_images / name, dst_images / name)
    lbl_src = src_labels / f"{Path(name).stem}.txt"
    lbl_dst = dst_labels / f"{Path(name).stem}.txt"
    if lbl_src.exists():
        shutil.copy2(lbl_src, lbl_dst)
    else:
        lbl_dst.write_text("")


def load_coco_class_counts() -> dict[str, Counter]:
    """Istogramma delle classi COCO (orientazione originale) rilevate per
    ogni immagine di union_reviewed, indicizzato per nome file union (come
    in --source-dir/images/). Riusa la cache di select_images.py
    (config.IMAGE_INDEX_PATH per risalire a dataset/split dal nome file,
    config.VARIETY_CACHE_PATH per le detection): tutte le immagini di
    union_reviewed sono passate per il filtro varietà, quindi ci sono già,
    senza bisogno di nessuna inferenza aggiuntiva."""
    if not config.IMAGE_INDEX_PATH.exists() or not config.VARIETY_CACHE_PATH.exists():
        return {}
    index = json.loads(config.IMAGE_INDEX_PATH.read_text())
    variety = json.loads(config.VARIETY_CACHE_PATH.read_text())

    split_by_dataset_filename = {
        (e["dataset_id"], Path(e["image_path"]).name): e["split"] for e in index
    }
    counts: dict[str, Counter] = {}
    for ds, split_by_filename in _group_by_dataset(split_by_dataset_filename).items():
        for filename, split in split_by_filename.items():
            key = f"{ds}/{split}/images/{filename}"
            entry = variety.get(key)
            if entry is None:
                continue
            counts[f"{ds}__{filename}"] = Counter(d[0] for d in entry["detections"]["0"])
    return counts


def _group_by_dataset(split_by_dataset_filename: dict) -> dict[str, dict[str, str]]:
    grouped: dict[str, dict[str, str]] = {}
    for (ds, filename), split in split_by_dataset_filename.items():
        grouped.setdefault(ds, {})[filename] = split
    return grouped


def coco_similarity(a: str, b: str, coco_counts: dict[str, Counter]) -> float | None:
    """Similarità di Jaccard pesata sui conteggi fra gli istogrammi di
    classe COCO di due immagini: sum(min)/sum(max) sulle classi rilevate in
    almeno una delle due. None se manca la cache per una delle due, o se
    nessuna delle due ha detection (nulla su cui basare un confronto)."""
    ca, cb = coco_counts.get(a), coco_counts.get(b)
    if ca is None or cb is None:
        return None
    classes = set(ca) | set(cb)
    if not classes:
        return None
    num = sum(min(ca[c], cb[c]) for c in classes)
    den = sum(max(ca[c], cb[c]) for c in classes)
    return num / den if den else None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("threshold", type=int, nargs="?", default=config.PHASH_DISTANCE_THRESHOLD,
                         help=f"Distanza di Hamming massima fra due pHash perché due immagini siano "
                              f"quasi-duplicate (default: {config.PHASH_DISTANCE_THRESHOLD})")
    parser.add_argument("--source-dir", type=Path, default=config.UNION_REVIEWED_DIR,
                         help=f"Dataset sorgente, con images/ e labels/ (default: {config.UNION_REVIEWED_DIR})")
    parser.add_argument("--out-dir", type=Path, default=config.UNION_REVIEWED_PHASHDEDUP_DIR,
                         help=f"Cartella di destinazione (default: {config.UNION_REVIEWED_PHASHDEDUP_DIR})")
    parser.add_argument("--phash-cache", type=Path, default=None,
                         help="Cache dei perceptual hash, riusata fra esecuzioni "
                              "(default: <out-dir>/phash_cache.json)")
    parser.add_argument("--coco-similarity-threshold", type=float, default=0.5,
                         help="Similarità minima (0-1) fra gli istogrammi di classe COCO di una coppia "
                              "esclusa/tenuta perché resti considerata un quasi-duplicato; sotto soglia "
                              "la coppia viene salvata dall'esclusione (default: 0.5)")
    parser.add_argument("--no-coco-check", action="store_true",
                         help="Disattiva il controllo di plausibilità via detection COCO")
    args = parser.parse_args()

    source_dir = args.source_dir.resolve()
    src_images = source_dir / "images"
    src_labels = source_dir / "labels"
    if not src_images.is_dir() or not src_labels.is_dir():
        raise SystemExit(f"{source_dir} deve contenere le sottocartelle images/ e labels/")

    names = sorted(p.name for p in src_images.iterdir())
    if not names:
        raise SystemExit(f"Nessuna immagine trovata in {src_images}")

    out_dir = args.out_dir.resolve()
    cache_path = (args.phash_cache or out_dir / "phash_cache.json").resolve()
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Soglia distanza pHash: {args.threshold}")
    print(f"Calcolo/carico pHash per {len(names)} immagini di {source_dir}...")
    phashes = review_app.compute_phashes(source_dir, cache_path)

    hashes = np.array([np.uint64(int(phashes[n], 16)) for n in names], dtype=np.uint64)
    thresholds = np.full(len(names), args.threshold, dtype=np.int16)
    uf = select_images.cluster_near_duplicates(hashes, thresholds)

    groups: dict[int, list[int]] = {}
    for i in range(len(names)):
        groups.setdefault(uf.find(i), []).append(i)

    box_counts = [count_boxes(src_labels / f"{Path(n).stem}.txt") for n in names]

    keep = [False] * len(names)
    exclusions = []  # (nome_escluso, nome_tenuto, distanza)
    for idxs in groups.values():
        if len(idxs) == 1:
            keep[idxs[0]] = True
            continue
        best = min(idxs, key=lambda i: (-box_counts[i], names[i]))
        keep[best] = True
        for i in idxs:
            if i == best:
                continue
            dist = bin(int(hashes[i]) ^ int(hashes[best])).count("1")
            exclusions.append((names[i], names[best], dist))

    n_clusters_with_dupes = sum(1 for idxs in groups.values() if len(idxs) > 1)

    rescued = []  # (nome, keeper, distanza, coco_similarity)
    coco_counts: dict[str, Counter] = {}
    if not args.no_coco_check:
        coco_counts = load_coco_class_counts()
        if not coco_counts:
            print("ATTENZIONE: cache detection COCO non trovata "
                  f"({config.IMAGE_INDEX_PATH}, {config.VARIETY_CACHE_PATH}): controllo COCO disattivato.")
        else:
            still_excluded = []
            for name, keeper, dist in exclusions:
                sim = coco_similarity(name, keeper, coco_counts)
                if sim is not None and sim < args.coco_similarity_threshold:
                    rescued.append((name, keeper, dist, sim))
                else:
                    still_excluded.append((name, keeper, dist))
            exclusions = still_excluded

    out_images, out_labels = out_dir / "images", out_dir / "labels"
    excl_images, excl_labels = out_dir / "excluded" / "images", out_dir / "excluded" / "labels"
    excl_nearest = out_dir / "excluded" / "nearest_selected"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    for d in (out_images, out_labels, excl_images, excl_labels, excl_nearest):
        d.mkdir(parents=True, exist_ok=True)

    excluded_names = {name for name, _, _ in exclusions}
    for name in names:
        if name in excluded_names:
            copy_pair(name, src_images, src_labels, excl_images, excl_labels)
        else:
            copy_pair(name, src_images, src_labels, out_images, out_labels)

    # per ogni esclusa, copia l'immagine tenuta più vicina con lo stesso nome
    # dell'esclusa, cosi le si può confrontare una a fianco all'altra
    for name, keeper, _ in exclusions:
        shutil.copy2(src_images / keeper, excl_nearest / name)

    with (out_dir / "excluded" / "manifest.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["excluded", "nearest_selected", "phash_distance", "coco_similarity"])
        for name, keeper, dist in sorted(exclusions):
            sim = coco_similarity(name, keeper, coco_counts) if coco_counts else None
            writer.writerow([name, keeper, dist, f"{sim:.2f}" if sim is not None else ""])

    total_per_dataset: dict[str, int] = {}
    excluded_per_dataset: dict[str, int] = {}
    for name in names:
        ds = dataset_of(name)
        total_per_dataset[ds] = total_per_dataset.get(ds, 0) + 1
    for name, _, _ in exclusions:
        ds = dataset_of(name)
        excluded_per_dataset[ds] = excluded_per_dataset.get(ds, 0) + 1

    print()
    for ds in sorted(total_per_dataset):
        n_excl = excluded_per_dataset.get(ds, 0)
        if n_excl:
            print(f"{ds}: {n_excl} escluse su {total_per_dataset[ds]} immagini")
    print()
    print(f"Cluster di quasi-duplicati trovati: {n_clusters_with_dupes}")
    print(f"Immagini escluse: {len(exclusions)}/{len(names)}")
    if not args.no_coco_check and coco_counts:
        print(f"Immagini salvate dall'esclusione per detection COCO troppo diverse: {len(rescued)} "
              f"(soglia similarità {args.coco_similarity_threshold})")
    print(f"Immagini tenute: {len(names) - len(exclusions)}/{len(names)} -> {out_dir}")
    print(f"Immagini escluse salvate in: {out_dir / 'excluded'}")
    print(f"  (con il riferimento all'immagine tenuta più vicina in excluded/nearest_selected/ "
          f"e in excluded/manifest.csv)")

    log_path = config.DATA_ROOT / "logs" / f"dedupe_phash-t{args.threshold}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_lines = [
        f"# Log dedupe_phash (soglia={args.threshold}, sorgente={source_dir})",
        f"# {len(exclusions)}/{len(names)} immagini escluse, {n_clusters_with_dupes} cluster di quasi-duplicati, "
        f"{len(rescued)} salvate per detection COCO troppo diverse",
        "",
    ]
    for name, keeper, dist in sorted(exclusions):
        log_lines.append(f"SCARTATA {name}  (quasi-duplicato di {keeper}, distanza pHash {dist})")
    if rescued:
        log_lines += ["", f"## Salvate dall'esclusione per detection COCO troppo diverse ({len(rescued)})", ""]
        for name, keeper, dist, sim in sorted(rescued):
            log_lines.append(f"SALVATA {name}  (pHash vicino a {keeper}, distanza {dist}, "
                              f"ma similarità COCO {sim:.2f} < {args.coco_similarity_threshold})")
    log_path.write_text("\n".join(log_lines) + "\n")
    print(f"Log scritto in {log_path}")


if __name__ == "__main__":
    main()
