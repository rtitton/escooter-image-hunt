#!/usr/bin/env python3
"""Copia le immagini candidate (di norma data/selected_images.txt) mantenendo
la separazione per dataset sorgente: per ogni dataset crea una cartella
data/interim/<id>-selected/ con la stessa struttura di <id>-dedup/ (split
train/valid/test, ciascuno con images/ e labels/, più data.yaml), contenente
solo le immagini selezionate.

A differenza di build_union_dataset.py — che unisce tutti i dataset in
un'unica cartella e rimappa le istanze escooter alla classe 80 — qui i
dataset restano separati e le label vengono copiate così come sono, senza
alcun remap di classe.
"""
import argparse
import shutil
from pathlib import Path

import config
import dataset_index

DATA_ROOT = config.DATA_ROOT
CANDIDATES_PATH = config.CANDIDATES_PATH
SPLITS = ("train", "valid", "test")
SELECTED_SUFFIX = "-selected"


def load_candidates(path: Path, limit: int | None = None) -> list[str]:
    lines = [l for l in path.read_text().splitlines() if l.strip() and not l.startswith("#")]
    return lines[:limit] if limit else lines


def group_by_dataset(candidates: list[str]) -> dict[str, list[tuple[str, str]]]:
    grouped: dict[str, list[tuple[str, str]]] = {}
    for rel in candidates:
        dataset_id, split, _, filename = rel.split("/")
        grouped.setdefault(dataset_id, []).append((split, filename))
    return grouped


def clear_out_dir(out_dir: Path) -> None:
    for split in SPLITS:
        for sub in ("images", "labels"):
            sub_dir = out_dir / split / sub
            if sub_dir.exists():
                shutil.rmtree(sub_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates-file", type=Path, default=CANDIDATES_PATH,
                         help="Elenco delle immagini da copiare (default: data/selected_images.txt)")
    parser.add_argument("--limit", type=int, default=None,
                         help="Considera solo le prime N candidate (per test su campione)")
    args = parser.parse_args()

    candidates = load_candidates(args.candidates_file, args.limit)
    grouped = group_by_dataset(candidates)
    entries = {e["id"]: e for e in dataset_index.load_index()}

    log_lines = []
    total_copied = 0

    for dataset_id, items in sorted(grouped.items()):
        entry = entries.get(dataset_id)
        if entry is None or not entry.get("dedup_dir"):
            log_lines.append(f"SALTATO dataset {dataset_id}: non trovato nell'indice o privo di dedup_dir")
            print(f"ATTENZIONE: dataset {dataset_id} non trovato nell'indice o privo di dedup_dir, saltato.")
            continue

        dedup_dir = DATA_ROOT.parent / entry["dedup_dir"]
        out_dir = DATA_ROOT / "interim" / f"{dataset_id}{SELECTED_SUFFIX}"
        clear_out_dir(out_dir)

        copied = 0
        missing = []
        for split, filename in items:
            src_img = dedup_dir / split / "images" / filename
            src_lbl = dedup_dir / split / "labels" / (Path(filename).stem + ".txt")
            if not src_img.exists():
                missing.append(f"{dataset_id}/{split}/images/{filename}")
                continue

            out_images = out_dir / split / "images"
            out_labels = out_dir / split / "labels"
            out_images.mkdir(parents=True, exist_ok=True)
            out_labels.mkdir(parents=True, exist_ok=True)

            shutil.copy2(src_img, out_images / filename)
            if src_lbl.exists():
                shutil.copy2(src_lbl, out_labels / src_lbl.name)
            copied += 1

        data_yaml_src = dedup_dir / "data.yaml"
        if data_yaml_src.exists():
            shutil.copy2(data_yaml_src, out_dir / "data.yaml")

        total_copied += copied
        print(f"{dataset_id}: {copied}/{len(items)} immagini copiate in {out_dir}")
        log_lines.append(f"{dataset_id}: {copied}/{len(items)} copiate in {out_dir}")
        for m in missing:
            log_lines.append(f"  MANCANTE {m}")

    print(f"Totale: {total_copied}/{len(candidates)} immagini copiate.")

    log_path = DATA_ROOT / "logs" / f"build_selected_datasets-{args.candidates_file.stem}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        f"# Log build_selected_datasets ({total_copied}/{len(candidates)} copiate da {args.candidates_file})\n\n"
        + "\n".join(log_lines) + "\n"
    )
    print(f"Log scritto in {log_path}")


if __name__ == "__main__":
    main()
