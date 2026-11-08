#!/usr/bin/env python3
"""Sceglie un sottoinsieme di COCO (formato YOLO di Ultralytics) per classi e
numero massimo di immagini, e ne scrive la lista dei percorsi.

Logica
  - Un'immagine è idonea se contiene almeno una delle classi scelte (--classes).
  - Delle immagini scelte si tengono TUTTE le etichette, anche quelle delle classi non scelte
    (non si toccano i file di label): togliendole, quegli oggetti diventerebbero sfondo.
  - Il budget --max-images si divide per quote fra le classi, dalla più rara alla più frequente:
    la quota di una classe è (posti rimasti) / (classi ancora da trattare). Le classi con meno immagini
    della quota le prendono tutte e il budget avanzato passa alle successive; le frequenti (person, car)
    vengono tagliate. Le immagini già scelte per una classe contano anche per le altre classi che
    contengono. La scelta dentro la classe è casuale (--seed), quindi riproducibile.
  - --min-per-class K: porta ogni classe ad almeno K immagini (se ne ha) anche oltre il budget.

Uscita
  --out-list  lista di percorsi assoluti, uno per riga (si passa a build_coco_merged_lists.py con
              --coco-train-list), ordinata
  --report    CSV con, per ogni classe COCO, immagini e istanze prima e dopo il taglio

--classes accetta nomi COCO o indici, e il preset "traffic".

Avvio:
    python3 scripts/training/make_coco_subset.py --coco-dir data/coco --split train2017
        --classes traffic --max-images 30000 --out-list data/processed/coco_subset_traffic/train.txt
        [--report ...csv] [--min-per-class K] [--seed 42]
"""
import argparse
import csv
import math
import os
import random
from collections import defaultdict
from pathlib import Path

import sys
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../dataset")))  # moduli di scripts/dataset/
from split_dataset import CLASS_NAMES

COCO_NAMES = CLASS_NAMES[:80]  # nell'ordine di coco.yaml di Ultralytics
PRESETS = {
    "traffic": ["person", "bicycle", "car", "motorcycle", "bus", "train", "truck", "traffic light",
                "fire hydrant", "stop sign", "parking meter", "bench", "backpack", "umbrella", "handbag",
                "suitcase", "skateboard"],
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def parse_classes(specs: list) -> list:
    out = []
    for spec in specs:
        for item in PRESETS.get(spec, [spec]):
            if item.isdigit():
                idx = int(item)
            elif item in COCO_NAMES:
                idx = COCO_NAMES.index(item)
            else:
                raise SystemExit(f"Classe sconosciuta: {item!r} (nomi COCO, indici 0-79 o preset {sorted(PRESETS)})")
            if not 0 <= idx < 80:
                raise SystemExit(f"Indice di classe fuori da 0-79: {idx}")
            if idx not in out:
                out.append(idx)
    return out


def read_counts(label_file: Path) -> dict:
    """{classe: istanze} di un file di label; {} se il file non esiste (immagine senza oggetti)."""
    counts = defaultdict(int)
    if label_file.exists():
        for line in label_file.read_text().splitlines():
            parts = line.split()
            if parts:
                counts[int(float(parts[0]))] += 1
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--coco-dir", type=Path, required=True, help="COCO in formato YOLO (images/ e labels/)")
    parser.add_argument("--split", default="train2017", help="train2017 (default) o val2017")
    parser.add_argument("--classes", nargs="+", required=True, help="Nomi COCO, indici o preset (traffic)")
    parser.add_argument("--max-images", type=int, required=True, help="Numero massimo di immagini del subset")
    parser.add_argument("--min-per-class", type=int, default=0,
                         help="Minimo di immagini per classe, anche oltre il budget (default: 0)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-list", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=None, help="CSV per classe (default: accanto alla lista)")
    args = parser.parse_args()

    classes = parse_classes(args.classes)
    images_dir = args.coco_dir / "images" / args.split
    labels_dir = args.coco_dir / "labels" / args.split
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise SystemExit(f"Layout non trovato: {images_dir} e {labels_dir}")
    print(f"{len(classes)} classi: {', '.join(COCO_NAMES[c] for c in classes)}")

    # Una passata sui label: per ogni immagine le istanze per classe.
    stems = sorted(p.stem for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    counts = {s: read_counts(labels_dir / f"{s}.txt") for s in stems}
    ext = {p.stem: p.suffix for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS}
    by_class = {c: [s for s in stems if c in counts[s]] for c in classes}

    rng = random.Random(args.seed)
    selected = set()
    for i, c in enumerate(sorted(classes, key=lambda c: (len(by_class[c]), c))):
        remaining_classes = len(classes) - i
        quota = math.ceil(max(args.max_images - len(selected), 0) / remaining_classes)
        quota = max(quota, args.min_per_class - sum(1 for s in by_class[c] if s in selected))
        new = [s for s in by_class[c] if s not in selected]
        selected.update(rng.sample(new, min(max(quota, 0), len(new))))
    if args.min_per_class == 0 and len(selected) > args.max_images:
        raise SystemExit("Errore interno: superato il budget")

    paths = sorted(os.path.abspath(images_dir / f"{s}{ext[s]}") for s in selected)
    args.out_list.parent.mkdir(parents=True, exist_ok=True)
    args.out_list.write_text("\n".join(paths) + "\n")

    report = args.report or args.out_list.with_suffix(".csv")
    with report.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["classe", "nome", "scelta", "immagini_prima", "immagini_dopo", "istanze_prima", "istanze_dopo"])
        for c in range(80):
            before = [s for s in stems if c in counts[s]]
            after = [s for s in before if s in selected]
            w.writerow([c, COCO_NAMES[c], int(c in classes), len(before), len(after),
                        sum(counts[s][c] for s in before), sum(counts[s][c] for s in after)])
    print(f"{len(selected)} immagini su {len(stems)} -> {args.out_list}")
    print("Per classe scelta (immagini dopo/prima):")
    for c in classes:
        n_before = len(by_class[c])
        n_after = sum(1 for s in by_class[c] if s in selected)
        print(f"  {COCO_NAMES[c]:15s} {n_after:6d} / {n_before}")
    print(f"Report: {report}")


if __name__ == "__main__":
    main()
