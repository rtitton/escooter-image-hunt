#!/usr/bin/env python3
"""Download a Roboflow dataset in YOLO format into data/raw/."""
import argparse
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from roboflow import Roboflow

import dataset_index

FORMAT = "yolov8"
DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "raw"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, help="Roboflow workspace id")
    parser.add_argument("--project", required=True, help="Roboflow project id")
    parser.add_argument("--version", type=int, default=None, help="Numero di versione del dataset (default: la più recente senza augmentation, se disponibile)")
    args = parser.parse_args()

    load_dotenv()
    rf = Roboflow(api_key=os.environ["RF_API_KEY"])
    project = rf.workspace(args.workspace).project(args.project)

    versions = project.versions()
    print("Versioni disponibili:")
    for v in versions:
        aug = v.augmentation or {}
        print(f"  v{v.version}: {v.images} immagini, augmentation={'nessuna' if not aug else list(aug.keys())}")

    version_number = args.version
    if version_number is None:
        no_aug_versions = [v for v in versions if not v.augmentation]
        pool = no_aug_versions or versions
        version_number = max(int(v.version) for v in pool)
        print(f"Nessuna versione specificata, uso: {version_number}"
              f"{' (nessuna augmentation)' if no_aug_versions else ' (attenzione: tutte le versioni hanno augmentation)'}")

    version = project.version(version_number)
    out_dir = DATA_ROOT / f"{args.project}-v{version_number}"
    version.download(FORMAT, location=str(out_dir))
    print(f"Dataset scaricato in {out_dir}")

    annotation_format = check_annotation_format(out_dir)
    update_index(out_dir, args.project, version_number, annotation_format)


def check_annotation_format(dataset_dir: Path) -> str:
    """Verifica che le annotazioni siano bounding box semplici (5 campi) e non poligoni."""
    non_bbox = 0
    checked = 0
    for labels_dir in dataset_dir.glob("*/labels"):
        for label_file in labels_dir.glob("*.txt"):
            for line in label_file.read_text().splitlines():
                if not line.strip():
                    continue
                checked += 1
                if len(line.split()) != 5:
                    non_bbox += 1

    if checked == 0:
        print("Attenzione: nessuna annotazione trovata da verificare.")
        return "sconosciuto"
    if non_bbox == 0:
        print(f"Formato annotazioni verificato: {checked}/{checked} righe sono bounding box a 5 campi.")
        return "bbox"
    if non_bbox == checked:
        print(f"ATTENZIONE: {non_bbox}/{checked} righe di annotazione NON sono bounding box a 5 campi "
              f"(probabile formato poligono/segmentazione). Questo dataset è fuori scope per ora.")
        return "poligono"
    print(f"ATTENZIONE: formato misto — {non_bbox}/{checked} righe non sono bounding box a 5 campi. "
          f"Questo dataset è fuori scope per ora.")
    return "misto"


def update_index(dataset_dir: Path, project: str, version_number: int, annotation_format: str) -> None:
    data_yaml = yaml.safe_load((dataset_dir / "data.yaml").read_text())
    rf_meta = data_yaml.get("roboflow", {})

    entry = {
        "id": dataset_dir.name,
        "workspace": rf_meta.get("workspace", ""),
        "project": project,
        "version": version_number,
        "url": rf_meta.get("url", ""),
        "classes": data_yaml.get("names", []),
        "images_raw": dataset_index.image_counts(dataset_dir),
        "annotation_format": annotation_format,
        "in_scope": annotation_format == "bbox",
    }

    index = dataset_index.load_index()
    index = dataset_index.upsert(index, entry)
    dataset_index.save_index(index)
    print(f"Indice aggiornato: {dataset_index.README_PATH}")


if __name__ == "__main__":
    main()
