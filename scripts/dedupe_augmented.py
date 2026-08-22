#!/usr/bin/env python3
"""Rimuove le copie generate da augmentation, tenendo un'immagine per nome-base.

Roboflow esporta ogni immagine (originale o aumentata) come
    <nome-base>.rf.<hash>.<ext>
con lo stesso nome-base condiviso tra le varianti augmentate di una stessa
immagine sorgente. Questo script raggruppa per nome-base e tiene solo il
primo file (in ordine alfabetico) di ogni gruppo, copiando la coppia
immagine+label corrispondente in una nuova cartella.
"""
import argparse
import re
import shutil
from pathlib import Path

import dataset_index

RF_SUFFIX = re.compile(r"\.rf\.[a-f0-9]+\.(jpg|jpeg|png)$", re.IGNORECASE)
SPLITS = ("train", "valid", "test")


def base_name(filename: str) -> str:
    return RF_SUFFIX.sub("", filename)


def dedupe_split(src_split: Path, dst_split: Path) -> tuple[int, int]:
    img_dir = src_split / "images"
    lbl_dir = src_split / "labels"
    if not img_dir.exists():
        return 0, 0

    groups: dict[str, list[Path]] = {}
    for img_path in sorted(img_dir.iterdir()):
        groups.setdefault(base_name(img_path.name), []).append(img_path)

    dst_img_dir = dst_split / "images"
    dst_lbl_dir = dst_split / "labels"
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    for group in groups.values():
        chosen = group[0]
        shutil.copy2(chosen, dst_img_dir / chosen.name)
        label_path = lbl_dir / (chosen.stem + ".txt")
        if label_path.exists():
            shutil.copy2(label_path, dst_lbl_dir / label_path.name)

    total_files = sum(len(g) for g in groups.values())
    return len(groups), total_files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, type=Path, help="Cartella del dataset scaricato (es. data/raw/electric-scooter-cd7hw-v1)")
    parser.add_argument("--out-dir", type=Path, default=None, help="Cartella di output (default: data/interim/<nome>-dedup)")
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    out_dir = args.out_dir or dataset_dir.parents[1] / "interim" / f"{dataset_dir.name}-dedup"

    data_yaml = dataset_dir / "data.yaml"
    if data_yaml.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(data_yaml, out_dir / "data.yaml")

    for split in SPLITS:
        kept, total = dedupe_split(dataset_dir / split, out_dir / split)
        if total:
            print(f"{split}: {kept}/{total} immagini tenute ({total - kept} scartate come varianti augmentate)")

    print(f"Dataset deduplicato in {out_dir}")

    index = dataset_index.load_index()
    matches = [e for e in index if e["id"] == dataset_dir.name]
    if matches:
        matches[0]["images_dedup"] = dataset_index.image_counts(out_dir)
        dataset_index.save_index(index)
        print(f"Indice aggiornato: {dataset_index.README_PATH}")
    else:
        print(f"Nota: nessuna voce in {dataset_index.INDEX_PATH} per '{dataset_dir.name}', indice non aggiornato "
              f"(esegui download_dataset.py per registrare prima il dataset raw).")


if __name__ == "__main__":
    main()
