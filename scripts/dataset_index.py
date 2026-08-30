"""Mantiene aggiornati data/datasets.json (sorgente dati) e data/README.md (vista leggibile).

Usato da download_dataset.py e dedupe_augmented.py — non è pensato per essere
eseguito direttamente.
"""
import csv
import json
from pathlib import Path

import config

DATA_ROOT = config.DATA_ROOT
INDEX_PATH = config.INDEX_PATH
README_PATH = config.README_PATH
CSV_PATH = config.CSV_PATH


def load_index() -> list[dict]:
    if INDEX_PATH.exists():
        return json.loads(INDEX_PATH.read_text())
    return []


def save_index(index: list[dict]) -> None:
    index = sorted(index, key=lambda e: e["id"])
    INDEX_PATH.write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n")
    render_readme(index)


def upsert(index: list[dict], entry: dict) -> list[dict]:
    """Inserisce entry, unendo i campi con una voce esistente stesso id
    (es. non perdere images_dedup se si ri-esegue download_dataset.py dopo
    aver già deduplicato)."""
    existing = next((e for e in index if e["id"] == entry["id"]), None)
    merged = {**existing, **entry} if existing else entry
    index = [e for e in index if e["id"] != entry["id"]]
    index.append(merged)
    return index


def enabled_ids() -> set[str]:
    """Ritorna gli id (`<project_id>-v<version>`, coerenti con quelli
    assegnati da download_dataset.py) delle righe con enabled=1 in
    datasets_to_download.csv. Usato per escludere dai passi di
    elaborazione i dataset disabilitati senza doverli rimuovere dall'indice."""
    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        print(f"Indice CSV dei dataset utilizzato: {CSV_PATH}")
        reader = csv.DictReader(f)
        return {
            f"{row['project_id']}-v{row['version']}"
            for row in reader
            if row["enabled"] == "1" and row["version"]
        }


# Colonne di datasets_to_download.csv che, se valorizzate, sovrascrivono per
# quel dataset il parametro di selezione omonimo di config/.env. Cella vuota o
# "default" -> nessun override.
SELECTION_OVERRIDE_COLUMNS: dict[str, type] = {
    "variety_min_instances": int,
    "closeup_area_threshold": float,
    "faraway_area_threshold": float,
    "phash_distance_threshold": int,
}


def selection_overrides() -> dict[str, dict]:
    """Override per-dataset dei parametri di selezione (v.
    SELECTION_OVERRIDE_COLUMNS), letti dal CSV. Chiave = id dataset
    (`<project_id>-v<version>`); valore = dict dei soli parametri
    effettivamente sovrascritti. Dataset senza override non compaiono."""
    overrides: dict[str, dict] = {}
    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not row.get("version"):
                continue
            ds_id = f"{row['project_id']}-v{row['version']}"
            resolved: dict = {}
            for col, cast in SELECTION_OVERRIDE_COLUMNS.items():
                raw = (row.get(col) or "").strip()
                if not raw or raw.lower() == "default":
                    continue
                try:
                    resolved[col] = cast(raw)
                except ValueError as e:
                    raise ValueError(
                        f"{CSV_PATH}: valore non valido nella colonna '{col}' per {ds_id}: {raw!r}"
                    ) from e
            if resolved:
                overrides[ds_id] = resolved
    return overrides


def count_images(split_dir: Path) -> int:
    img_dir = split_dir / "images"
    return len(list(img_dir.glob("*"))) if img_dir.exists() else 0


def image_counts(dataset_dir: Path) -> dict:
    return {split: count_images(dataset_dir / split) for split in ("train", "valid", "test")}


def render_readme(index: list[dict]) -> None:
    lines = [
        "# Indice dataset scaricati",
        "",
        "Generato automaticamente da `scripts/download_dataset.py` e "
        "`scripts/dedupe_augmented.py` — non modificare a mano.",
        "",
        "| id | sorgente | classi | classi escooter | immagini raw (train/valid/test) | formato | in scope | dedup (train/valid/test) | poligoni convertiti |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for e in index:
        src = f"[{e['workspace']}/{e['project']} v{e['version']}]({e['url']})"
        rc = e["images_raw"]
        raw_str = f"{rc['train']}/{rc['valid']}/{rc['test']}"
        dedup = e.get("images_dedup")
        dedup_str = f"{dedup['train']}/{dedup['valid']}/{dedup['test']}" if dedup else "—"
        scope = "sì" if e["in_scope"] else "no"
        converted = e.get("converted_polygon_annotations")
        converted_str = str(converted) if converted is not None else "—"
        escooter_names = e.get("escooter_class_names") or []
        missing = e.get("escooter_class_names_missing") or []
        escooter_str = ", ".join(escooter_names) if escooter_names else "—"
        if missing:
            escooter_str += f" ⚠️ mancanti: {', '.join(missing)}"
        lines.append(
            f"| `{e['id']}` | {src} | {', '.join(e['classes'])} | {escooter_str} | {raw_str} | "
            f"{e['annotation_format']} | {scope} | {dedup_str} | {converted_str} |"
        )

    lines.append("")
    lines.append(
        "Percorsi: `data/raw/<id>/` contiene il dataset scaricato, invariato. "
        "`scripts/dedupe_augmented.py` raggruppa le varianti augmentate per nome-base e copia "
        "in `data/interim/<id>-dedup/` una sola immagine per gruppo (preferendo, quando "
        "possibile, quella senza segni di rotazione), convertendo eventuali poligoni in "
        "bounding box. Il log dettagliato di ogni esecuzione è in `data/logs/<id>.log`."
    )
    README_PATH.write_text("\n".join(lines) + "\n")
