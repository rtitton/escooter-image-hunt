#!/usr/bin/env python3
"""Copia le immagini candidate (di norma data/selected_images.txt) in una
cartella di destinazione (di norma il dataset di unione).

Per ogni immagine candidata copia il file immagine e scrive un file di label
con le sole istanze escooter, rimappate alla classe 80 — le altre classi dei
dataset sorgente vengono scartate: le classi COCO saranno annotate in un
passo successivo con un modello Ultralytics pretrained di grandi dimensioni
(vedi README.md).

I file di destinazione sono prefissati con l'id del dataset sorgente per
evitare collisioni di nome tra dataset diversi.

Con --candidates-file si può puntare a un elenco diverso, per esempio
data/flagged_rider_contamination.txt (immagini in cui una bbox escooter
include probabilmente il conducente, prodotto da select_images.py), da
copiare in una cartella separata per la correzione manuale invece che nel
dataset di unione vero e proprio — le bbox originali (comprese quelle
"sbagliate") vengono mantenute così come sono, come base di partenza per la
correzione.
"""
import argparse
import shutil
from pathlib import Path

import yaml

import config
import dataset_index
import select_images  # riusa escooter_class_indices() e read_boxes()

DATA_ROOT = config.DATA_ROOT
CANDIDATES_PATH = config.CANDIDATES_PATH
ESCOOTER_CLASS_ID = config.ESCOOTER_CLASS_ID


def load_candidates(path: Path, limit: int | None = None) -> list[str]:
    lines = [l for l in path.read_text().splitlines() if l.strip() and not l.startswith("#")]
    return lines[:limit] if limit else lines


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates-file", type=Path, default=CANDIDATES_PATH,
                         help="Elenco delle immagini da copiare (default: data/selected_images.txt)")
    parser.add_argument("--out-dir", type=Path, default=DATA_ROOT / "processed" / "union",
                         help="Cartella di destinazione (default: data/processed/union)")
    parser.add_argument("--limit", type=int, default=None,
                         help="Copia solo le prime N candidate (per test su campione)")
    args = parser.parse_args()

    log_path = DATA_ROOT / "logs" / f"build_union_dataset-{args.candidates_file.stem}.log"
    candidates = load_candidates(args.candidates_file, args.limit)
    entries = {e["id"]: e for e in dataset_index.load_index()}

    out_images = args.out_dir / "images"
    out_labels = args.out_dir / "labels"
    
    # cancella eventuali images/labels già presenti
    
    if out_images.exists():
        shutil.rmtree(out_images)
    else:
        out_images.mkdir(parents=True, exist_ok=True)
        
    if out_labels.exists():
        shutil.rmtree(out_labels)
    else:
        out_labels.mkdir(parents=True, exist_ok=True)

    names_cache: dict[str, list[str]] = {}
    skipped_no_escooter = []
    copied = 0

    for rel in candidates:
        dataset_id, split, _, filename = rel.split("/")
        entry = entries[dataset_id]
        dedup_dir = DATA_ROOT.parent / entry["dedup_dir"]
        img_path = dedup_dir / split / "images" / filename
        lbl_path = dedup_dir / split / "labels" / (Path(filename).stem + ".txt")

        if dataset_id not in names_cache:
            data_yaml = yaml.safe_load((dedup_dir / "data.yaml").read_text())
            names_cache[dataset_id] = data_yaml.get("names", [])
        escooter_idx = select_images.escooter_class_indices(entry, names_cache[dataset_id])

        escooter_boxes = [b for b in select_images.read_boxes(lbl_path) if b[0] in escooter_idx]
        if not escooter_boxes:
            skipped_no_escooter.append(rel)
            continue

        dest_stem = f"{dataset_id}__{Path(filename).stem}"
        shutil.copy2(img_path, out_images / f"{dest_stem}{Path(filename).suffix}")
        label_lines = [
            f"{ESCOOTER_CLASS_ID} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"
            for _, xc, yc, w, h in escooter_boxes
        ]
        (out_labels / f"{dest_stem}.txt").write_text("\n".join(label_lines) + "\n")
        copied += 1

    print(f"{copied}/{len(candidates)} immagini copiate in {args.out_dir}")
    if skipped_no_escooter:
        print(f"ATTENZIONE: {len(skipped_no_escooter)} candidate senza istanze escooter nel label "
              f"(inatteso: select_images.py le avrebbe dovute scartare a monte).")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        f"# Log build_union_dataset ({copied}/{len(candidates)} copiate da {args.candidates_file} in {args.out_dir})\n\n"
        f"## Candidate senza istanze escooter, saltate ({len(skipped_no_escooter)})\n"
        + "\n".join(skipped_no_escooter) + "\n"
    )
    print(f"Log scritto in {log_path}")


if __name__ == "__main__":
    main()
