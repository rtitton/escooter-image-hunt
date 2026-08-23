#!/usr/bin/env python3
"""Costruisce una versione deduplicata di un dataset scaricato, senza toccare il raw.

Roboflow esporta ogni immagine (originale o aumentata) come
    <nome-base>.rf.<hash>.<ext>
con lo stesso nome-base condiviso tra le varianti augmentate di una stessa
immagine sorgente. Questo script raggruppa per nome-base e, per ogni gruppo,
copia in data/interim/<nome>-dedup/ un solo file (preferendo uno che non
mostri segni di rotazione, vedi looks_rotated). Le annotazioni a poligono
vengono convertite nel bounding box minimo che le contiene. Il dataset in
data/raw/ resta invariato.

Per ogni esecuzione viene scritto un log in data/logs/<nome>.log con
l'elenco delle immagini scartate e delle annotazioni convertite.
"""
import argparse
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

import bbox_convert
import dataset_index

RF_SUFFIX = re.compile(r"\.rf\.[a-f0-9]+\.(jpg|jpeg|png)$", re.IGNORECASE)
SPLITS = ("train", "valid", "test")

BLACK_THRESHOLD = 10  # valore di grigio sotto il quale un pixel è considerato "nero"
EDGE_SAMPLE = 30  # righe/colonne campionate a partire da ogni bordo


def base_name(filename: str) -> str:
    return RF_SUFFIX.sub("", filename)


def looks_rotated(image_path: Path) -> bool:
    """Rileva il padding a cuneo tipico della rotazione, distinguendolo dal
    letterboxing (barre nere a larghezza costante), che invece è presente
    indistintamente su originali e augmentate e non va usato come segnale.

    Per ciascuno dei 4 bordi, controlliamo se il conteggio di pixel neri si
    assottiglia allontanandosi dal bordo (cuneo, tipico della rotazione)
    invece di restare costante (barra piena, letterboxing) o nullo (nessun
    padding).
    """
    with Image.open(image_path) as im:
        gray = np.asarray(im.convert("L"))
    h, w = gray.shape
    n = min(EDGE_SAMPLE, h // 2, w // 2)
    if n < 5:
        return False

    edges = {
        "top": [(gray[i] < BLACK_THRESHOLD).sum() for i in range(n)],
        "bottom": [(gray[h - 1 - i] < BLACK_THRESHOLD).sum() for i in range(n)],
        "left": [(gray[:, i] < BLACK_THRESHOLD).sum() for i in range(n)],
        "right": [(gray[:, w - 1 - i] < BLACK_THRESHOLD).sum() for i in range(n)],
    }
    for edge, counts in edges.items():
        size = w if edge in ("top", "bottom") else h
        first, last = counts[0], counts[-1]
        if size * 0.15 < first < size * 0.95 and last < first * 0.7:
            return True
    return False


def choose_keeper(group: list[Path]) -> Path:
    """Sceglie il file da tenere per un gruppo di varianti augmentate:
    preferisce il primo (in ordine alfabetico) che non sembra ruotato;
    se tutti sembrano ruotati, tiene comunque il primo (nessun'altra base
    per decidere)."""
    for candidate in group:
        if not looks_rotated(candidate):
            return candidate
    return group[0]


def dedupe_split(split: str, src_split: Path, dst_split: Path, discarded_log: list, converted_log: list) -> tuple[int, int, int]:
    """Deduplica uno split, appendendo le voci di log a discarded_log/converted_log.
    Ritorna (immagini tenute, immagini totali, annotazioni convertite da poligono)."""
    img_dir = src_split / "images"
    lbl_dir = src_split / "labels"
    if not img_dir.exists():
        return 0, 0, 0

    groups: dict[str, list[Path]] = {}
    for img_path in sorted(img_dir.iterdir()):
        groups.setdefault(base_name(img_path.name), []).append(img_path)

    dst_img_dir = dst_split / "images"
    dst_lbl_dir = dst_split / "labels"
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    converted_total = 0
    for base, group in groups.items():
        keeper = choose_keeper(group)
        shutil.copy2(keeper, dst_img_dir / keeper.name)

        for extra in group:
            if extra != keeper:
                discarded_log.append(
                    f"{split}/images/{extra.name}  (gruppo={base}, tenuta={keeper.name})"
                )

        label_path = lbl_dir / (keeper.stem + ".txt")
        if label_path.exists():
            original_text = label_path.read_text()
            n_converted = sum(
                1 for line in original_text.splitlines()
                if line.strip() and len(line.split()) > bbox_convert.BBOX_FIELDS
            )
            (dst_lbl_dir / label_path.name).write_text(bbox_convert.convert_label_file(original_text))
            if n_converted:
                converted_log.append(
                    f"{split}/labels/{label_path.name}  ({n_converted} annotazioni poligono->bbox)"
                )
            converted_total += n_converted

    total_files = sum(len(g) for g in groups.values())
    return len(groups), total_files, converted_total


def dedupe_dataset(dataset_dir: Path, out_dir: Path | None = None) -> dict:
    """Deduplica un dataset scaricato, scrive il log e aggiorna l'indice.
    Ritorna un riepilogo con out_dir e il numero di annotazioni convertite."""
    dataset_dir = dataset_dir.resolve()
    data_root = dataset_dir.parents[1]
    out_dir = out_dir or data_root / "interim" / f"{dataset_dir.name}-dedup"

    data_yaml = dataset_dir / "data.yaml"
    if data_yaml.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(data_yaml, out_dir / "data.yaml")

    discarded_log: list = []
    converted_log: list = []
    total_converted = 0
    for split in SPLITS:
        kept, total, converted = dedupe_split(split, dataset_dir / split, out_dir / split, discarded_log, converted_log)
        if total:
            print(f"{split}: {kept}/{total} immagini copiate in {out_dir / split} "
                  f"({total - kept} scartate come varianti augmentate, {converted} annotazioni poligono->bbox)")
        total_converted += converted

    print(f"Dataset deduplicato in {out_dir}")

    log_dir = data_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{dataset_dir.name}.log"
    log_lines = [
        f"# Log deduplica: {dataset_dir.name}",
        f"# generato: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"## Immagini scartate come varianti augmentate ({len(discarded_log)})",
        *discarded_log,
        "",
        f"## Annotazioni convertite da poligono a bounding box ({total_converted})",
        *converted_log,
        "",
    ]
    log_path.write_text("\n".join(log_lines))
    print(f"Log scritto in {log_path}")

    index = dataset_index.load_index()
    matches = [e for e in index if e["id"] == dataset_dir.name]
    if matches:
        matches[0]["images_dedup"] = dataset_index.image_counts(out_dir)
        matches[0]["dedup_dir"] = str(out_dir.relative_to(data_root.parent))
        matches[0]["converted_polygon_annotations"] = total_converted
        dataset_index.save_index(index)
        print(f"Indice aggiornato: {dataset_index.README_PATH}")
    else:
        print(f"Nota: nessuna voce in {dataset_index.INDEX_PATH} per '{dataset_dir.name}', indice non aggiornato "
              f"(esegui download_dataset.py per registrare prima il dataset raw).")

    return {"out_dir": out_dir, "converted_polygon_annotations": total_converted}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, type=Path, help="Cartella del dataset scaricato (es. data/raw/electric-scooter-cd7hw-v1)")
    parser.add_argument("--out-dir", type=Path, default=None, help="Cartella di output (default: data/interim/<nome>-dedup)")
    args = parser.parse_args()

    dedupe_dataset(args.dataset_dir, args.out_dir)


if __name__ == "__main__":
    main()
