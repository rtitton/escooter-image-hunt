#!/usr/bin/env python3
"""Unisce COCO (nel formato YOLO scaricato da Ultralytics) e il nostro split
train/valid in un unico dataset per Ultralytics, senza copiare né spostare
immagini: scrive due file di lista (train.txt, val.txt, un percorso
assoluto per riga) e un data.yaml a 81 classi (COCO 0-79 + escooter 80)
che li usa.

Layout atteso:
    <coco-dir>/images/train2017/*.jpg   <coco-dir>/labels/train2017/*.txt
    <coco-dir>/images/val2017/*.jpg     <coco-dir>/labels/val2017/*.txt
    <split-dir>/train/{images,labels}   <split-dir>/valid/{images,labels}
(il secondo è l'output di split_dataset.py). Ultralytics ricava il label di
un'immagine sostituendo l'ultimo "/images/" del percorso con "/labels/" e
l'estensione con ".txt": vale per entrambi i layout, quindi i percorsi nelle
liste NON vanno risolti (niente realpath), altrimenti si rompe la
corrispondenza. Le immagini COCO senza file label sono immagini senza
oggetti (sfondo), come nel resto di Ultralytics.

  train.txt = tutte le immagini di COCO train2017 + quelle del nostro train,
              ripetute --repeat volte (default 1; le righe ripetute restano
              duplicate nel dataset di Ultralytics, con augmentation diversa
              a ogni epoca)
  val.txt   = COCO val2017 + il nostro valid (mai ripetuto)

Controlla che i label di COCO usino gli indici 0-79 (come l'originale di
Ultralytics e come il nostro schema): con gli id originali di COCO (1-90) il
merge sarebbe sbagliato.

Avvio:
    python3 scripts/build_coco_merged_lists.py --coco-dir DIR --split-dir DIR
        --out-dir DIR [--repeat N] [--coco-train-list LISTA]
"""
import argparse
import os
import random
from pathlib import Path

from split_dataset import CLASS_NAMES

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SANITY_SAMPLE = 300  # label di COCO controllati per verificare che gli indici siano 0-79


def list_images(images_dir: Path) -> list:
    """Percorsi assoluti (non risolti) delle immagini di una cartella, ordinati."""
    if not images_dir.is_dir():
        raise SystemExit(f"Cartella non trovata: {images_dir}")
    paths = [os.path.abspath(os.path.join(images_dir, n)) for n in os.listdir(images_dir)
             if os.path.splitext(n)[1].lower() in IMAGE_EXTENSIONS]
    if not paths:
        raise SystemExit(f"Nessuna immagine in {images_dir}")
    return sorted(paths)


def label_path(image_path: str) -> str:
    """Come Ultralytics: ultimo /images/ -> /labels/, estensione -> .txt."""
    head, sep, tail = image_path.rpartition(f"{os.sep}images{os.sep}")
    if not sep:
        raise SystemExit(f"Il percorso non contiene /images/: {image_path}")
    return os.path.splitext(f"{head}{os.sep}labels{os.sep}{tail}")[0] + ".txt"


def check_coco_labels(images: list, seed: int = 0) -> None:
    """Su un campione di label COCO: esistono quasi tutti (ma non per forza: le immagini senza
    oggetti non hanno file) e gli indici di classe sono 0-79."""
    sample = random.Random(seed).sample(images, min(SANITY_SAMPLE, len(images)))
    found = max_cls = 0
    for img in sample:
        lbl = label_path(img)
        if not os.path.exists(lbl):
            continue
        found += 1
        for line in Path(lbl).read_text().splitlines():
            parts = line.split()
            if parts:
                max_cls = max(max_cls, int(float(parts[0])))
    if found == 0:
        raise SystemExit("Nessun label trovato per il campione di immagini COCO: controllare il layout "
                         "(<coco-dir>/labels/train2017/*.txt accanto a images/train2017)")
    if max_cls > 79:
        raise SystemExit(f"Nei label COCO c'è la classe {max_cls}: gli indici devono essere 0-79 "
                         "(formato Ultralytics, non gli id originali di COCO)")
    print(f"  controllo label COCO: {found}/{len(sample)} del campione hanno un file, classe massima {max_cls}")


def write_lines(path: Path, lines: list) -> None:
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--coco-dir", type=Path, required=True, help="COCO in formato YOLO (images/ e labels/)")
    parser.add_argument("--split-dir", type=Path, required=True, help="Il nostro split (output di split_dataset.py)")
    parser.add_argument("--out-dir", type=Path, required=True, help="Dove scrivere train.txt, val.txt, data.yaml")
    parser.add_argument("--coco-train-list", type=Path, default=None,
                         help="Lista di percorsi (es. da make_coco_subset.py) da usare al posto di tutto train2017")
    parser.add_argument("--repeat", type=int, default=1,
                         help="Quante volte inserire le nostre immagini di train (default: 1)")
    args = parser.parse_args()
    if args.repeat < 1:
        raise SystemExit("--repeat deve essere almeno 1")

    if args.coco_train_list:
        coco_train = [l for l in args.coco_train_list.read_text().splitlines() if l.strip()]
    else:
        coco_train = list_images(args.coco_dir / "images" / "train2017")
    coco_val = list_images(args.coco_dir / "images" / "val2017")
    our_train = list_images(args.split_dir / "train" / "images")
    our_valid = list_images(args.split_dir / "valid" / "images")
    print(f"COCO: {len(coco_train)} train, {len(coco_val)} val; nostro split: {len(our_train)} train, {len(our_valid)} valid")
    check_coco_labels(coco_train)

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    train = coco_train + our_train * args.repeat
    val = coco_val + our_valid
    write_lines(out_dir / "train.txt", train)
    write_lines(out_dir / "val.txt", val)
    yaml_lines = [f"path: {out_dir}", "train: train.txt", "val: val.txt", f"nc: {len(CLASS_NAMES)}", "names:"]
    yaml_lines += [f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES)]
    write_lines(out_dir / "data.yaml", yaml_lines)

    print(f"train.txt: {len(train)} righe ({len(coco_train)} COCO + {len(our_train)} nostre x {args.repeat})")
    print(f"val.txt:   {len(val)} righe ({len(coco_val)} COCO + {len(our_valid)} nostre)")
    print(f"Scritto in {out_dir} (data.yaml incluso)")


if __name__ == "__main__":
    main()
