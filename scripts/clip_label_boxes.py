#!/usr/bin/env python3
"""Ritaglia ai bordi dell'immagine le bbox dei label YOLO che sporgono
(centro + metà dimensione oltre [0, 1]): box del genere, create trascinando
oltre il bordo in review_app.py (ora impedito), equivalgono a box amodali,
contrarie alla convenzione COCO adottata per le classi COCO. Una bbox
interamente fuori dall'immagine viene eliminata.

Riscrive solo i file con almeno una bbox da correggere, e solo quelle righe
(le altre restano com'erano); soglia di tolleranza 1e-4 per gli arrotondamenti
dei label (6 decimali). Con --dry-run stampa soltanto cosa cambierebbe.

Avvio:
    python3 scripts/clip_label_boxes.py <dataset_dir> [--dry-run]
con <dataset_dir> che contiene la sottocartella labels/ (es. union_reviewed_coco).
"""
import argparse
from pathlib import Path

from review_app import clip_box

TOLERANCE = 1e-4


def sporge(xc: float, yc: float, w: float, h: float) -> bool:
    return (xc - w / 2 < -TOLERANCE or yc - h / 2 < -TOLERANCE
            or xc + w / 2 > 1 + TOLERANCE or yc + h / 2 > 1 + TOLERANCE)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset_dir", type=Path, help="Dataset con la sottocartella labels/")
    parser.add_argument("--dry-run", action="store_true", help="Non scrive: mostra solo cosa cambierebbe")
    args = parser.parse_args()

    labels_dir = args.dataset_dir / "labels"
    if not labels_dir.is_dir():
        raise SystemExit(f"{labels_dir} non esiste")

    files = clipped = removed = 0
    for path in sorted(labels_dir.glob("*.txt")):
        out, changed = [], False
        for line in path.read_text().splitlines():
            parts = line.split()
            if len(parts) == 5 and sporge(*map(float, parts[1:])):
                changed = True
                new = clip_box(*map(float, parts[1:]))
                if new is None:
                    removed += 1
                    continue
                clipped += 1
                line = f"{int(float(parts[0]))} " + " ".join(f"{v:.6f}" for v in new)
            out.append(line)
        if changed:
            files += 1
            if not args.dry_run:
                path.write_text("\n".join(out) + "\n" if out else "")

    print(f"{'Da correggere' if args.dry_run else 'Corretti'}: {clipped} bbox ritagliate, "
          f"{removed} eliminate (interamente fuori), in {files} file")


if __name__ == "__main__":
    main()
