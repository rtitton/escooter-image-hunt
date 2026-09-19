#!/usr/bin/env python3
"""Materializza la versione revisionata del dataset di unione: copia in
--out-dir solo le immagini con decisione "select" nel file di decisioni di
review_app.py, insieme ai rispettivi label (che riflettono già eventuali
modifiche alle bbox fatte durante la review, essendo scritti subito su
disco da review_app.py).

A differenza di build_bydataset_annotated.py (pensato per un controllo
visivo, con bbox disegnate e immagini divise per dataset sorgente), qui
l'output è un dataset YOLO pronto all'uso: stessa struttura images/+labels/
del dataset sorgente, nomi file invariati.

Ad ogni esecuzione la cartella di output viene svuotata e ripopolata.

Avvio:
    python3 scripts/materialize_union_reviewed.py
        [--source-dir DIR] [--decisions-file FILE] [--out-dir DIR] [--limit N]
"""
import argparse
import json
import shutil
from pathlib import Path

import config


def load_selected_names(decisions_file: Path) -> list[str]:
    if not decisions_file.exists():
        raise SystemExit(f"File di decisioni non trovato: {decisions_file}")
    decisions = json.loads(decisions_file.read_text())
    return sorted(name for name, decision in decisions.items() if decision == "select")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-dir", type=Path, default=config.UNION_DIR,
                         help=f"Dataset sorgente, con images/ e labels/ (default: {config.UNION_DIR})")
    parser.add_argument("--decisions-file", type=Path, default=None,
                         help="File JSON di review_app.py (default: <source-dir>/review_decisions.json)")
    parser.add_argument("--out-dir", type=Path, default=config.UNION_REVIEWED_DIR,
                         help=f"Cartella di destinazione (default: {config.UNION_REVIEWED_DIR})")
    parser.add_argument("--limit", type=int, default=None,
                         help="Materializza solo le prime N immagini selezionate (per test su campione)")
    args = parser.parse_args()

    source_dir = args.source_dir.resolve()
    decisions_file = (args.decisions_file or source_dir / "review_decisions.json").resolve()
    src_images = source_dir / "images"
    src_labels = source_dir / "labels"
    if not src_images.is_dir() or not src_labels.is_dir():
        raise SystemExit(f"{source_dir} deve contenere le sottocartelle images/ e labels/")

    all_names = {p.name for p in src_images.iterdir()}
    selected = load_selected_names(decisions_file)
    missing = [n for n in selected if n not in all_names]
    if missing:
        raise SystemExit(
            f"{len(missing)} immagini selezionate nel file di decisioni non sono presenti in {src_images} "
            f"(es. {missing[0]})"
        )
    if args.limit:
        selected = selected[:args.limit]

    todo = sum(1 for n in all_names if n not in json.loads(decisions_file.read_text()))
    if todo:
        print(f"ATTENZIONE: {todo} immagini del dataset sorgente non hanno ancora una decisione "
              f"(non conteggiate qui: la review non è completa).")

    out_images = args.out_dir / "images"
    out_labels = args.out_dir / "labels"
    if args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    total_per_dataset: dict[str, int] = {}
    for name in all_names:
        dataset_id = Path(name).stem.partition("__")[0]
        total_per_dataset[dataset_id] = total_per_dataset.get(dataset_id, 0) + 1

    selected_per_dataset: dict[str, int] = {}
    for name in selected:
        stem = Path(name).stem
        shutil.copy2(src_images / name, out_images / name)
        lbl_src = src_labels / f"{stem}.txt"
        lbl_dst = out_labels / f"{stem}.txt"
        if lbl_src.exists():
            shutil.copy2(lbl_src, lbl_dst)
        else:
            lbl_dst.write_text("")
        dataset_id = stem.partition("__")[0]
        selected_per_dataset[dataset_id] = selected_per_dataset.get(dataset_id, 0) + 1

    for dataset_id in sorted(total_per_dataset):
        n = selected_per_dataset.get(dataset_id, 0)
        m = total_per_dataset[dataset_id]
        print(f"{dataset_id}: {n} / {m} immagini selezionate")
    print(f"Totale: {len(selected)} / {len(all_names)} immagini selezionate, materializzate in {args.out_dir}")


if __name__ == "__main__":
    main()
