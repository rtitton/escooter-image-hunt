"""Mantiene aggiornati data/datasets.json (sorgente dati) e data/README.md (vista leggibile).

Usato da download_dataset.py e dedupe_augmented.py — non è pensato per essere
eseguito direttamente.
"""
import json
from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
INDEX_PATH = DATA_ROOT / "datasets.json"
README_PATH = DATA_ROOT / "README.md"


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
