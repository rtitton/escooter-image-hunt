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
    index = [e for e in index if e["id"] != entry["id"]]
    index.append(entry)
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
        "| id | sorgente | classi | immagini raw (train/valid/test) | formato | in scope | dedup (train/valid/test) |",
        "|---|---|---|---|---|---|---|",
    ]
    for e in index:
        src = f"[{e['workspace']}/{e['project']} v{e['version']}]({e['url']})"
        rc = e["images_raw"]
        raw_str = f"{rc['train']}/{rc['valid']}/{rc['test']}"
        dedup = e.get("images_dedup")
        dedup_str = f"{dedup['train']}/{dedup['valid']}/{dedup['test']}" if dedup else "—"
        scope = "sì" if e["in_scope"] else "no"
        lines.append(
            f"| `{e['id']}` | {src} | {', '.join(e['classes'])} | {raw_str} | "
            f"{e['annotation_format']} | {scope} | {dedup_str} |"
        )

    lines.append("")
    lines.append(
        "Percorsi: `data/raw/<id>/` (download originale), "
        "`data/interim/<id>-dedup/` (dopo deduplica augmentation, se presente)."
    )
    README_PATH.write_text("\n".join(lines) + "\n")
