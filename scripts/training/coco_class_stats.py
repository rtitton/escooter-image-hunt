#!/usr/bin/env python3
"""Istanze e immagini per classe di COCO (formato YOLO di Ultralytics), in un CSV
ordinato per frequenza (istanze di train, dalla più alta alla più bassa).

Colonne: classe, nome, istanze e immagini per ogni split richiesto (default train2017 e val2017).
Le immagini senza file di label (nessun oggetto) non contano per nessuna classe.

Avvio (dalla radice del repo):
    python3 scripts/training/coco_class_stats.py [--coco-dir data/coco] [--out data/coco/class_stats.csv]
        [--splits train2017 val2017]
"""
import argparse
import csv
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../dataset")))  # moduli di scripts/dataset/
from split_dataset import CLASS_NAMES

COCO_NAMES = CLASS_NAMES[:80]


def split_stats(labels_dir: Path):
    """(istanze per classe, immagini per classe) dai file di label di una cartella."""
    if not labels_dir.is_dir():
        raise SystemExit(f"Cartella non trovata: {labels_dir}")
    instances, images = Counter(), Counter()
    for f in labels_dir.glob("*.txt"):
        classes = [int(float(l.split()[0])) for l in f.read_text().splitlines() if l.split()]
        instances.update(classes)
        images.update(set(classes))
    return instances, images


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--coco-dir", type=Path, default=Path("data/coco"))
    parser.add_argument("--splits", nargs="+", default=["train2017", "val2017"])
    parser.add_argument("--out", type=Path, default=None, help="CSV di uscita (default: <coco-dir>/class_stats.csv)")
    args = parser.parse_args()

    stats = {s: split_stats(args.coco_dir / "labels" / s) for s in args.splits}
    first = args.splits[0]
    order = sorted(range(80), key=lambda c: (-stats[first][0][c], c))
    out = args.out or args.coco_dir / "class_stats.csv"
    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["classe", "nome"] + [f"{k}_{s}" for s in args.splits for k in ("istanze", "immagini")])
        for c in order:
            w.writerow([c, COCO_NAMES[c]] + [stats[s][i][c] for s in args.splits for i in (0, 1)])
    print(f"{out} (ordinato per istanze di {first})")
    for c in order[:5] + order[-5:]:
        print(f"  {COCO_NAMES[c]:15s} {stats[first][0][c]:7d} istanze, {stats[first][1][c]:6d} immagini")


if __name__ == "__main__":
    main()
