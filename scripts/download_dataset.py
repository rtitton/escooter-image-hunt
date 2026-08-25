#!/usr/bin/env python3
"""Download a Roboflow dataset in YOLO format into data/raw/."""
import argparse
from pathlib import Path

import yaml
from roboflow import Roboflow

import config
import dataset_index

DOWNLOAD_FORMAT = config.DOWNLOAD_FORMAT
RAW_DIR = config.RAW_DIR


def download_dataset(
    workspace: str,
    project: str,
    version: int | None = None,
    escooter_class_names: list[str] | None = None,
) -> tuple[Path, int, list[str]]:
    """Scarica un dataset Roboflow, verifica il formato annotazioni e (se
    escooter_class_names è fornito) che quei nomi esistano nel data.yaml.
    Aggiorna l'indice. Ritorna (cartella scaricata, versione scaricata,
    nomi classe mancanti)."""
    rf = Roboflow(api_key=config.RF_API_KEY)
    rf_project = rf.workspace(workspace).project(project)

    versions = rf_project.versions()
    if not versions:
        raise RuntimeError(
            f"Il progetto {workspace}/{project} non ha nessuna versione pubblicata su Roboflow "
            f"(il proprietario non ha mai generato una 'version' esportabile)."
        )
    print("Versioni disponibili:")
    for v in versions:
        aug = v.augmentation or {}
        print(f"  v{v.version}: {v.images} immagini, augmentation={'nessuna' if not aug else list(aug.keys())}")

    version_number = version
    if version_number is None:
        no_aug_versions = [v for v in versions if not v.augmentation]
        pool = no_aug_versions or versions
        version_number = max(int(v.version) for v in pool)
        print(f"Nessuna versione specificata, uso: {version_number}"
              f"{' (nessuna augmentation)' if no_aug_versions else ' (attenzione: tutte le versioni hanno augmentation)'}")

    rf_version = rf_project.version(version_number)
    out_dir = RAW_DIR / f"{project}-v{version_number}"
    rf_version.download(DOWNLOAD_FORMAT, location=str(out_dir))
    print(f"Dataset scaricato in {out_dir}")

    annotation_format = check_annotation_format(out_dir)
    missing = validate_class_names(out_dir, escooter_class_names) if escooter_class_names else []
    update_index(out_dir, workspace, project, version_number, annotation_format, escooter_class_names or [], missing)
    return out_dir, version_number, missing


def check_annotation_format(dataset_dir: Path) -> str:
    """Rileva se le annotazioni sono bounding box semplici (5 campi) o poligoni
    (verranno convertite in bbox da dedupe_augmented.py)."""
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
        print(f"Formato annotazioni: {checked}/{checked} righe sono già bounding box a 5 campi.")
        return "bbox"
    if non_bbox == checked:
        print(f"Formato annotazioni: poligono ({non_bbox}/{checked} righe) — verrà convertito in "
              f"bounding box nella versione interim.")
        return "poligono"
    print(f"Formato annotazioni: misto — {non_bbox}/{checked} righe a poligono, il resto bbox. "
          f"Verranno convertite tutte in bounding box nella versione interim.")
    return "misto"


def validate_class_names(dataset_dir: Path, requested: list[str]) -> list[str]:
    """Ritorna i nomi in `requested` che NON sono presenti tra le classi del data.yaml scaricato."""
    data_yaml = yaml.safe_load((dataset_dir / "data.yaml").read_text())
    available = data_yaml.get("names", [])
    missing = [name for name in requested if name not in available]
    if missing:
        print(f"ATTENZIONE: classi escooter non trovate nel data.yaml: {missing} "
              f"(classi disponibili: {available})")
    else:
        print(f"Classi escooter verificate: {requested} presenti nel data.yaml.")
    return missing


def update_index(
    dataset_dir: Path,
    workspace: str,
    project: str,
    version_number: int,
    annotation_format: str,
    escooter_class_names: list[str],
    escooter_class_names_missing: list[str],
) -> None:
    data_yaml = yaml.safe_load((dataset_dir / "data.yaml").read_text())

    # Il data.yaml di Roboflow può riportare workspace/url del progetto
    # sorgente originale (es. per un fork), non di quello effettivamente
    # richiesto: usiamo workspace/project passati esplicitamente, non i
    # metadati embedded, per evitare link morti nell'indice.
    entry = {
        "id": dataset_dir.name,
        "workspace": workspace,
        "project": project,
        "version": version_number,
        "url": f"https://universe.roboflow.com/{workspace}/{project}/dataset/{version_number}",
        "classes": data_yaml.get("names", []),
        "images_raw": dataset_index.image_counts(dataset_dir),
        "annotation_format": annotation_format,
        "in_scope": annotation_format != "sconosciuto",
        "escooter_class_names": escooter_class_names,
        "escooter_class_names_missing": escooter_class_names_missing,
    }

    index = dataset_index.load_index()
    index = dataset_index.upsert(index, entry)
    dataset_index.save_index(index)
    print(f"Indice aggiornato: {dataset_index.README_PATH}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, help="Roboflow workspace id")
    parser.add_argument("--project", required=True, help="Roboflow project id")
    parser.add_argument("--version", type=int, default=None, help="Numero di versione del dataset (default: la più recente senza augmentation, se disponibile)")
    parser.add_argument("--escooter-class-names", default=None, help="Nomi delle classi escooter attese nel data.yaml, separati da '|' (opzionale, per validazione)")
    args = parser.parse_args()

    names = args.escooter_class_names.split("|") if args.escooter_class_names else None
    _, version_number, _ = download_dataset(args.workspace, args.project, args.version, names)
    print(f"Versione scaricata: {version_number}")


if __name__ == "__main__":
    main()
