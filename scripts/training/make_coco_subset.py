#!/usr/bin/env python3
"""Sceglie un sottoinsieme di COCO (formato YOLO di Ultralytics) per classi e
numero massimo di immagini, e ne scrive la lista dei percorsi.

Logica
  - Un'immagine è idonea se contiene almeno una delle classi scelte (--classes).
  - Delle immagini scelte (e di quelle del val) si tengono SOLO le etichette delle classi scelte, con gli
    indici invariati: le altre classi non hanno più esempi, e i loro oggetti diventano sfondo (il modello
    impara a non rilevarli). Il subset è una linea di train a sé, non un COCO ridotto.
  - Il budget --max-images si divide per quote fra le classi, dalla più rara alla più frequente:
    la quota di una classe è (posti rimasti) / (classi ancora da trattare). Le classi con meno immagini
    della quota le prendono tutte e il budget avanzato passa alle successive; le frequenti (person, car)
    vengono tagliate. Le immagini già scelte per una classe contano anche per le altre classi che
    contengono. La scelta dentro la classe è casuale (--seed), quindi riproducibile.
  - --min-per-class K: porta ogni classe ad almeno K immagini (se ne ha) anche oltre il budget.

Validation
  Il val si ricava da val2017 con lo stesso criterio del train: le immagini che contengono almeno una
  classe scelta (tutte, senza tetto), con i label filtrati sulle stesse classi.

Etichette filtrate
  Ultralytics ricava i label dal percorso dell'immagine (/images/ -> /labels/), quindi le liste puntano a
  symlink in <out-dir>/train_filtered/images e <out-dir>/val_filtered/images, con accanto i label filtrati in
  .../labels. Le immagini originali non si toccano.

Uscita, in --out-dir
  train.txt, val.txt   percorsi delle immagini (symlink in train_filtered/ e val_filtered/)
  classes.txt          gli indici delle classi scelte (li usa build_coco_merged_lists.py per filtrare allo
                       stesso modo il nostro valid)
  train.csv            per ogni classe COCO, immagini e istanze prima e dopo il taglio (train)
  Si passa a build_coco_merged_lists.py con --subset-dir. Ad ogni esecuzione le cartelle *_filtered/ sono
  rigenerate.

--classes accetta nomi COCO o indici, e il preset "traffic".

Avvio:
    python3 scripts/training/make_coco_subset.py --coco-dir data/coco --classes traffic --max-images 30000
        --out-dir data/processed/coco_subset_30000 [--min-per-class K] [--seed 42]
  Senza --min-per-class mostra le frequenze delle classi scelte e chiede K (invio = 0).
"""
import argparse
import csv
import math
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

import sys
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../dataset")))  # moduli di scripts/dataset/
from split_dataset import CLASS_NAMES

COCO_NAMES = CLASS_NAMES[:80]  # nell'ordine di coco.yaml di Ultralytics
PRESETS = {
    "traffic": ["person", "bicycle", "car", "motorcycle", "bus", "train", "truck", "traffic light",
                "fire hydrant", "stop sign", "parking meter", "bench", "skateboard"],
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


def ask_min_per_class(classes: list, by_class: dict, counts: dict, max_images: int) -> int:
    """Stampa le frequenze delle classi scelte (immagini e istanze nello split) e chiede K, il minimo di
    immagini per classe. Invio = 0. Senza terminale interattivo usa 0 senza chiedere."""
    print(f"\nFrequenze nello split delle {len(classes)} classi scelte (dalla più frequente):")
    print(f"  {'classe':15s} {'immagini':>9s} {'istanze':>9s}")
    for c in sorted(classes, key=lambda c: -len(by_class[c])):
        n_inst = sum(counts[s][c] for s in by_class[c])
        print(f"  {COCO_NAMES[c]:15s} {len(by_class[c]):9d} {n_inst:9d}")
    quota = math.ceil(max_images / len(classes))
    print(f"\nQuota media con --max-images {max_images}: circa {quota} immagini per classe; le classi con meno "
          "immagini le prendono tutte.")
    print("K = minimo di immagini per classe anche oltre --max-images (0 = nessun minimo: il subset non "
          "supera mai --max-images).")
    if not sys.stdin.isatty():
        print("Terminale non interattivo: uso K = 0 (per cambiarlo, --min-per-class).")
        return 0
    while True:
        answer = input("K [default 0]: ").strip()
        if not answer:
            return 0
        if answer.isdigit():
            return int(answer)
        print("Inserire un intero >= 0 (o invio per 0).")


def write_filtered_dataset(pairs: list, out_dir: Path, keep: set) -> int:
    """Scrive in out_dir/images (symlink) e out_dir/labels le coppie (immagine, file di label) con le sole righe
    delle classi in keep; le immagini senza nessuna riga restante sono saltate. Ritorna quante ne scrive."""
    for sub in ("images", "labels"):
        shutil.rmtree(out_dir / sub, ignore_errors=True)
        (out_dir / sub).mkdir(parents=True)
    n = 0
    for image, label in pairs:
        lines = [l for l in (label.read_text().splitlines() if label.exists() else [])
                 if l.split() and int(float(l.split()[0])) in keep]
        if not lines:
            continue
        (out_dir / "images" / image.name).symlink_to(os.path.abspath(image))
        (out_dir / "labels" / f"{image.stem}.txt").write_text("\n".join(lines) + "\n")
        n += 1
    return n


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--coco-dir", type=Path, required=True, help="COCO in formato YOLO (images/ e labels/)")
    parser.add_argument("--classes", nargs="+", required=True, help="Nomi COCO, indici o preset (traffic)")
    parser.add_argument("--max-images", type=int, required=True, help="Numero massimo di immagini del subset")
    parser.add_argument("--min-per-class", type=int, default=None,
                         help="Minimo di immagini per classe, anche oltre il budget (default 0). Se omesso, "
                              "lo script mostra le frequenze delle classi scelte e lo chiede")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, required=True, help="Dove scrivere train.txt, val.txt, ecc.")
    args = parser.parse_args()

    classes = parse_classes(args.classes)
    images_dir = args.coco_dir / "images" / "train2017"
    labels_dir = args.coco_dir / "labels" / "train2017"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise SystemExit(f"Layout non trovato: {images_dir} e {labels_dir}")
    print(f"{len(classes)} classi: {', '.join(COCO_NAMES[c] for c in classes)}")

    # Una passata sui label: per ogni immagine le istanze per classe.
    stems = sorted(p.stem for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    counts = {s: read_counts(labels_dir / f"{s}.txt") for s in stems}
    ext = {p.stem: p.suffix for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS}
    by_class = {c: [s for s in stems if c in counts[s]] for c in classes}

    min_per_class = args.min_per_class
    if min_per_class is None:
        min_per_class = ask_min_per_class(classes, by_class, counts, args.max_images)

    rng = random.Random(args.seed)
    selected = set()
    for i, c in enumerate(sorted(classes, key=lambda c: (len(by_class[c]), c))):
        remaining_classes = len(classes) - i
        quota = math.ceil(max(args.max_images - len(selected), 0) / remaining_classes)
        quota = max(quota, min_per_class - sum(1 for s in by_class[c] if s in selected))
        new = [s for s in by_class[c] if s not in selected]
        selected.update(rng.sample(new, min(max(quota, 0), len(new))))
    if min_per_class == 0 and len(selected) > args.max_images:
        raise SystemExit("Errore interno: superato il budget")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_pairs = [(images_dir / f"{s}{ext[s]}", labels_dir / f"{s}.txt") for s in sorted(selected)]
    write_filtered_dataset(train_pairs, args.out_dir / "train_filtered", set(classes))
    paths = sorted(os.path.abspath(p) for p in (args.out_dir / "train_filtered" / "images").iterdir())
    (args.out_dir / "train.txt").write_text("\n".join(paths) + "\n")

    report = args.out_dir / "train.csv"
    with report.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["classe", "nome", "scelta", "immagini_prima", "immagini_dopo", "istanze_prima", "istanze_dopo"])
        for c in range(80):
            before = [s for s in stems if c in counts[s]]
            after = [s for s in before if s in selected]
            w.writerow([c, COCO_NAMES[c], int(c in classes), len(before), len(after),
                        sum(counts[s][c] for s in before), sum(counts[s][c] for s in after)])
    print(f"{len(selected)} immagini su {len(stems)} -> {args.out_dir / 'train.txt'}")
    print("Per classe scelta (immagini dopo/prima):")
    for c in classes:
        n_before = len(by_class[c])
        n_after = sum(1 for s in by_class[c] if s in selected)
        print(f"  {COCO_NAMES[c]:15s} {n_after:6d} / {n_before}")
    print(f"Report: {report}")

    # Val: stesse classi, da val2017, con i label filtrati (v. docstring)
    val_images = args.coco_dir / "images" / "val2017"
    val_labels = args.coco_dir / "labels" / "val2017"
    pairs = [(p, val_labels / f"{p.stem}.txt") for p in sorted(val_images.iterdir())
             if p.suffix.lower() in IMAGE_EXTENSIONS]
    n = write_filtered_dataset(pairs, args.out_dir / "val_filtered", set(classes))
    val_paths = sorted(os.path.abspath(p) for p in (args.out_dir / "val_filtered" / "images").iterdir())
    (args.out_dir / "val.txt").write_text("\n".join(val_paths) + "\n")
    (args.out_dir / "classes.txt").write_text("\n".join(map(str, classes)) + "\n")
    print(f"Val: {n} immagini su {len(pairs)} di val2017 -> {args.out_dir / 'val.txt'}")


if __name__ == "__main__":
    main()
