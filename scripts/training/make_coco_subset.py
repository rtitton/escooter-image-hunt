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
  percorsi che passano da <out-dir>/train/images e <out-dir>/val/images (un solo symlink
  ciascuno, a images/train2017 e images/val2017 di COCO), con accanto i label filtrati in .../labels. Le
  immagini originali non si toccano. Per avere una copia fisica: scripts/training/materialize_dataset.py.

Uscita, in --out-dir con _<max-images>_<min-per-class> aggiunto al nome (es. coco_subset -> coco_subset_30000_0)
  train.txt, val.txt   percorsi delle immagini (via i symlink di train/ e val/)
  data.yaml            80 classi COCO (indici invariati), train.txt e val.txt
  classes.txt          gli indici delle classi scelte (li usa build_coco_merged_lists.py per filtrare allo
                       stesso modo il nostro valid)
    train.csv, val.csv   per le classi scelte, conteggi e percentuali in COCO e nel subset
  Si passa a build_coco_merged_lists.py con --subset-dir. Ad ogni esecuzione le cartelle train/ e val/ sono
  rigenerate.

--classes accetta nomi COCO o indici, e il preset "traffic".

Avvio:
    python3 scripts/training/make_coco_subset.py --coco-dir data/coco --classes traffic --max-images 30000
        --out-dir data/processed/coco_subset [--min-per-class K] [--seed 42]
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
    """Stampa le frequenze delle classi scelte e chiede K, il minimo di immagini per classe. Invio = 0."""
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


def write_filtered_dataset(pairs: list, source_images: Path, out_dir: Path, keep: set) -> list:
    """Scrive in out_dir/labels i label delle coppie (immagine, file di label) con le sole righe delle classi in
    keep; le immagini senza nessuna riga restante sono saltate. out_dir/images è un solo symlink alla cartella
    delle immagini originali (source_images). Ritorna i percorsi delle immagini scritte, passando da out_dir/images."""
    images_link = out_dir / "images"
    if images_link.is_symlink():
        images_link.unlink()
    else:
        shutil.rmtree(images_link, ignore_errors=True)  # versioni precedenti: cartella di symlink per file
    shutil.rmtree(out_dir / "labels", ignore_errors=True)
    (out_dir / "labels").mkdir(parents=True)
    images_link.symlink_to(os.path.abspath(source_images))
    paths = []
    for image, label in pairs:
        lines = [l for l in (label.read_text().splitlines() if label.exists() else [])
                 if l.split() and int(float(l.split()[0])) in keep]
        if not lines:
            continue
        (out_dir / "labels" / f"{image.stem}.txt").write_text("\n".join(lines) + "\n")
        paths.append(os.path.abspath(images_link / image.name))
    return paths


def write_class_report(report: Path, classes: list, counts: dict, subset: set) -> None:
    coco_total_images = len(counts)
    coco_total_instances = sum(sum(class_counts.values()) for class_counts in counts.values())
    subset_total_images = len(subset)
    subset_total_instances = sum(counts[stem].get(c, 0) for stem in subset for c in classes)

    def percentage(value: int, total: int) -> str:
        return f"{value / total * 100:.2f}".replace(".", ",") if total else "0,00"

    with report.open("w", newline="") as f:
        w = csv.writer(f, delimiter=";")
        w.writerow(["CLASSE", "NOME", "IMMAGINI_COCO", "PERCENTUALE_IMMAGINI_COCO",
                    "ISTANZE_COCO", "PERCENTUALE_ISTANZE_COCO", "IMMAGINI_SUBSET",
                    "PERCENTUALE_IMMAGINI_SUBSET", "ISTANZE_SUBSET", "PERCENTUALE_ISTANZE_SUBSET"])
        for c in classes:
            before = [stem for stem, class_counts in counts.items() if c in class_counts]
            after = [stem for stem in before if stem in subset]
            coco_images = len(before)
            subset_images = len(after)
            coco_instances = sum(counts[stem][c] for stem in before)
            subset_instances = sum(counts[stem][c] for stem in after)
            w.writerow([c, COCO_NAMES[c], coco_images, percentage(coco_images, coco_total_images),
                        coco_instances, percentage(coco_instances, coco_total_instances),
                        subset_images, percentage(subset_images, subset_total_images),
                        subset_instances, percentage(subset_instances, subset_total_instances)])


def print_class_report(report: Path) -> None:
    with report.open(newline="") as f:
        rows = list(csv.reader(f, delimiter=";"))
    print(f"\n{report}:")
    groups = [
        ("COCO", [0, 1, 2, 3, 4, 5], ["ID", "NOME", "IMMAGINI", "% IMG", "ISTANZE", "% IST."]),
        ("SUBSET", [0, 1, 6, 7, 8, 9], ["ID", "NOME", "IMMAGINI", "% IMG", "ISTANZE", "% IST."]),
    ]
    for title, indices, headers in groups:
        table = [headers, *[[row[i] for i in indices] for row in rows[1:]]]
        widths = [max(len(row[i]) for row in table) for i in range(len(headers))]
        print(f"{title}:")
        for row_index, row in enumerate(table):
            cells = []
            for i, value in enumerate(row):
                align = "<" if row_index == 0 or i == 1 else ">"
                cells.append(f"{value:{align}{widths[i]}}")
            print("  ".join(cells))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--coco-dir", type=Path, required=True, help="COCO in formato YOLO (images/ e labels/)")
    parser.add_argument("--classes", nargs="+", required=True, help="Nomi COCO, indici o preset (traffic)")
    parser.add_argument("--max-images", type=int, required=True, help="Numero massimo di immagini del subset")
    parser.add_argument("--min-per-class", type=int, default=None,
                         help="Minimo di immagini per classe, anche oltre il budget (default 0). Se omesso, "
                              "lo script mostra le frequenze delle classi scelte e lo chiede")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, required=True, help="Dove scrivere train.txt, val.txt, ecc.; al nome si aggiunge _<max-images>_<min-per-class>")
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
    args.out_dir = args.out_dir.with_name(f"{args.out_dir.name}_{args.max_images}_{min_per_class}")
    print(f"Cartella di uscita: {args.out_dir}")

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
    paths = sorted(write_filtered_dataset(train_pairs, images_dir, args.out_dir / "train", set(classes)))
    (args.out_dir / "train.txt").write_text("\n".join(paths) + "\n")

    report = args.out_dir / "train.csv"
    write_class_report(report, classes, counts, selected)
    print(f"{len(selected)} immagini su {len(stems)} -> {args.out_dir / 'train.txt'}")
    if len(selected) > args.max_images:
        print(f"ATTENZIONE: {len(selected)} immagini, oltre --max-images {args.max_images}: "
              f"--min-per-class {min_per_class} ha portato le classi al minimo anche oltre il budget.")
    # Val: stesse classi, da val2017, con i label filtrati (v. docstring)
    val_images = args.coco_dir / "images" / "val2017"
    val_labels = args.coco_dir / "labels" / "val2017"
    pairs = [(p, val_labels / f"{p.stem}.txt") for p in sorted(val_images.iterdir())
             if p.suffix.lower() in IMAGE_EXTENSIONS]
    val_counts = {image.stem: read_counts(label) for image, label in pairs}
    val_paths = sorted(write_filtered_dataset(pairs, val_images, args.out_dir / "val", set(classes)))
    n = len(val_paths)
    (args.out_dir / "val.txt").write_text("\n".join(val_paths) + "\n")
    val_report = args.out_dir / "val.csv"
    write_class_report(val_report, classes, val_counts, {Path(path).stem for path in val_paths})
    (args.out_dir / "classes.txt").write_text("\n".join(map(str, classes)) + "\n")
    (args.out_dir / "data.yaml").write_text("\n".join(
        ["train: train.txt", "val: val.txt", f"nc: {len(COCO_NAMES)}", "names:"]
        + [f"  {i}: {n}" for i, n in enumerate(COCO_NAMES)]) + "\n")
    print(f"Val: {n} immagini su {len(pairs)} di val2017 -> {args.out_dir / 'val.txt'}")
    print_class_report(report)
    print_class_report(val_report)


if __name__ == "__main__":
    main()
