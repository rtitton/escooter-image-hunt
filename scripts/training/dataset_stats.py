#!/usr/bin/env python3
"""Statistiche a console di un dataset YOLO, con esportazione CSV opzionale.

Avvio:
    python3 scripts/training/dataset_stats.py data/processed/coco_escooter_merge_rep5/data.yaml
        [--sort frequency|id|name] [--csv statistiche.csv]

Supporta train/val/test come cartelle ricorsive, liste .txt o liste di sorgenti.
Il path relativo del dataset e le sorgenti relative partono dalla cartella dello
YAML. Nelle liste .txt, ./ parte dalla cartella della lista; gli altri percorsi
relativi partono dalla directory corrente, come in Ultralytics.
Una riga di label non vuota conta come un'istanza. Non modifica dati o cache.
Il CSV contiene riepilogo e classi, distinti dalla colonna tipo; le colonne
non pertinenti al tipo di riga restano vuote.
"""
import argparse
import csv
import math
import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import yaml

from materialize_dataset import label_path


IMAGE_EXTENSIONS = {
    ".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff",
    ".webp", ".pfm", ".heic",
}


def absolute(path: Path) -> Path:
    return Path(os.path.abspath(path))


def load_dataset(data_yaml: Path) -> tuple[dict[int, str], dict[str, list[Path]]]:
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{data_yaml}: attesa una mappa YAML")
    raw_names = data.get("names")
    if isinstance(raw_names, list):
        names = dict(enumerate(raw_names))
    elif isinstance(raw_names, dict):
        if any(isinstance(key, bool) or not isinstance(key, (int, str))
               or not str(key).isdigit() for key in raw_names):
            raise ValueError(f"{data_yaml}: ID di names non validi")
        names = {int(key): value for key, value in raw_names.items()}
        if len(names) != len(raw_names):
            raise ValueError(f"{data_yaml}: ID di names duplicati")
    else:
        raise ValueError(f"{data_yaml}: names deve essere una lista o una mappa")
    if not names or sorted(names) != list(range(len(names))):
        raise ValueError(f"{data_yaml}: gli ID di names devono essere consecutivi da 0")
    if any(not isinstance(name, str) or not name for name in names.values()):
        raise ValueError(f"{data_yaml}: nomi di classe non validi")
    if data.get("nc", len(names)) != len(names):
        raise ValueError(f"{data_yaml}: nc non corrisponde a names")
    root_value = data.get("path")
    if root_value is not None and not isinstance(root_value, str):
        raise ValueError(f"{data_yaml}: path deve essere una stringa")
    root = Path(root_value or data_yaml.parent)
    if not root.is_absolute():
        root = data_yaml.parent / root
    splits = {}
    for split in ("train", "val", "test"):
        source = data.get(split)
        if source is None or source == "":
            continue
        sources = source if isinstance(source, list) else [source]
        images = []
        for entry in sources:
            if not isinstance(entry, str) or not entry:
                raise ValueError(f"{data_yaml}: sorgente {split} non valida: {entry!r}")
            path = Path(entry)
            images.extend(read_images(path if path.is_absolute() else root / path))
        if not images:
            raise ValueError(f"{data_yaml}: nessuna immagine nello split {split}")
        splits[split] = images
    if not splits:
        raise ValueError(f"{data_yaml}: nessuno split train/val/test")
    return names, splits


def read_images(source: Path) -> list[Path]:
    if source.is_dir():
        return sorted(absolute(p) for p in source.rglob("*")
                      if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file())
    if not source.is_file():
        raise FileNotFoundError(f"Sorgente non trovata: {source}")
    if source.suffix.lower() != ".txt":
        raise ValueError(f"Attesa una cartella o una lista .txt: {source}")
    images = []
    for number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        entry = line.strip()
        if not entry:
            continue
        path = source.parent / entry[2:] if entry.startswith("./") else Path(entry)
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"{source}:{number}: estensione immagine non supportata: {entry}")
        images.append(absolute(path))
    return images


@dataclass
class Annotation:
    instances: Counter[int]
    missing: bool


def read_annotation(image: Path, names: dict[int, str]) -> Annotation:
    if not image.is_file():
        raise FileNotFoundError(f"Immagine non trovata (symlink rotto?): {image}")
    label = label_path(image)
    if not label.is_file():
        if label.exists() or label.is_symlink():
            raise ValueError(f"Label non regolare o symlink rotto: {label}")
        return Annotation(Counter(), True)
    counts: Counter[int] = Counter()
    for number, line in enumerate(label.read_text(encoding="utf-8").splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        try:
            values = [float(part) for part in parts]
        except ValueError as exc:
            raise ValueError(f"{label}:{number}: label non numerica") from exc
        if (len(values) < 5 or not all(math.isfinite(value) for value in values)
                or not values[0].is_integer() or int(values[0]) not in names):
            raise ValueError(f"{label}:{number}: label non valida o classe fuori da names")
        counts[int(values[0])] += 1
    return Annotation(counts, False)


@dataclass
class Statistics:
    images: int
    background: int
    missing: int
    empty: int
    instances: Counter[int]
    class_images: Counter[int]
    max_instances: int


def summarize(images: list[Path], annotations: dict[Path, Annotation]) -> Statistics:
    instances: Counter[int] = Counter()
    class_images: Counter[int] = Counter()
    background = missing = empty = max_instances = 0
    for image in images:
        annotation = annotations[image]
        counts = annotation.instances
        instances.update(counts)
        class_images.update(counts.keys())
        if not counts:
            background += 1
            missing += int(annotation.missing)
            empty += int(not annotation.missing)
        max_instances = max(max_instances, counts.total())
    return Statistics(len(images), background, missing, empty,
                      instances, class_images, max_instances)


def print_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [max(len(row[i]) for row in [headers, *rows]) for i in range(len(headers))]
    for row in [headers, *rows]:
        print("  ".join(value.ljust(width) if i == 0 else value.rjust(width)
                        for i, (value, width) in enumerate(zip(row, widths))))


def report(names: dict[int, str], splits: dict[str, list[Path]],
           annotations: dict[Path, Annotation], sort: str = "frequency",
           csv_path: Path | None = None) -> None:
    stats = {split: summarize(images, annotations) for split, images in splits.items()}
    total = summarize([image for images in splits.values() for image in images], annotations)
    print("\nRiepilogo:")
    summary_headers = ["split", "immagini", "istanze", "sfondo",
                       "label assenti", "label vuote", "ist./img", "max ist./img"]
    summary_rows = []
    for split, stat in [*stats.items(), ("TOTALE", total)]:
        summary_rows.append([split, str(stat.images), str(stat.instances.total()),
                             str(stat.background), str(stat.missing), str(stat.empty),
                             f"{stat.instances.total() / stat.images:.2f}",
                             str(stat.max_instances)])
    print_table(summary_headers, summary_rows)
    if sort == "frequency":
        order = sorted(names, key=lambda cls: (-total.instances[cls], cls))
        description = "istanze totali decrescenti"
    elif sort == "id":
        order = sorted(names)
        description = "ID crescente"
    elif sort == "name":
        order = sorted(names, key=lambda cls: (names[cls].casefold(), cls))
        description = "nome alfabetico"
    else:
        raise ValueError(f"Ordinamento non valido: {sort}")
    print(f"\nClassi ordinate per {description}:")
    rows = []
    for cls in order:
        count = total.instances[cls]
        percentage = 100 * count / total.instances.total() if total.instances else 0
        rows.append([str(cls), names[cls], str(count), f"{percentage:.2f}",
                     str(total.class_images[cls]),
                     *[str(stat.instances[cls]) for stat in stats.values()]])
    class_headers = ["ID", "classe", "istanze", "% istanze", "immagini",
                     *[f"ist. {split}" for split in stats]]
    print_table(class_headers, rows)
    print("\nLabel assenti o vuote sono contate come sfondo; verificare che sia intenzionale.")
    if csv_path is not None:
        headers = ["tipo", *summary_headers,
                   *[header for header in class_headers if header not in summary_headers]]
        with csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=headers)
            writer.writeheader()
            for kind, columns, records in (("riepilogo", summary_headers, summary_rows),
                                           ("classe", class_headers, rows)):
                for row in records:
                    writer.writerow({"tipo": kind, **dict(zip(columns, row))})
        print(f"CSV: {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data_yaml", type=Path, help="File data.yaml del dataset")
    parser.add_argument("--sort", choices=("frequency", "id", "name"), default="frequency",
                        help="Ordine classi: frequenza decrescente (default), ID o nome crescente")
    parser.add_argument("--csv", type=Path, default=None, help="Salva anche riepilogo e classi in CSV")
    args = parser.parse_args()
    try:
        data_yaml = absolute(args.data_yaml)
        names, splits = load_dataset(data_yaml)
        if args.csv is not None and args.csv.suffix.lower() != ".csv":
            raise ValueError("--csv deve indicare un file con estensione .csv")
        annotations = {}
        for split, images in splits.items():
            print(f"Lettura {split}: {len(images)} immagini...", file=sys.stderr, flush=True)
            for image in images:
                if image not in annotations:
                    annotations[image] = read_annotation(image, names)
        print(f"Dataset: {data_yaml}\nClassi dichiarate: {len(names)}")
        report(names, splits, annotations, args.sort, args.csv)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        parser.exit(1, f"Errore: {exc}\n")


if __name__ == "__main__":
    main()
