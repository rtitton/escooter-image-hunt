#!/usr/bin/env python3
"""Trasforma un dataset "virtuale" (liste train.txt / val.txt di percorsi di immagini, con i label accanto
secondo la convenzione di Ultralytics: /images/ -> /labels/, estensione .txt) in un dataset fisico: copia tutte
le immagini e tutte le label in <out-dir>/{train,val}/{images,labels} e scrive <out-dir>/data.yaml.

  - I symlink vengono seguiti: si copia il contenuto del file puntato, non il link.
  - Un'immagine senza file di label (sfondo) viene copiata e le si crea un label vuoto.
  - Righe ripetute nella lista (es. il --repeat di build_coco_merged_lists.py) vengono copiate una volta sola:
    la copia fisica non ha le ripetizioni (si segnala quante). Due immagini diverse con lo stesso nome fanno
    fermare lo script (si sovrascriverebbero).
  - data.yaml: nc e names sono copiati dal data.yaml accanto alla lista (quello scritto da make_coco_subset.py e
    da build_coco_merged_lists.py), o da --data-yaml. train e val puntano a train/images e val/images, senza
    "path": Ultralytics usa la cartella del data.yaml, quindi la cartella si può spostare.

Avvio:
    python3 scripts/training/materialize_dataset.py --train data/processed/coco_subset_30000/train.txt
        --val data/processed/coco_subset_30000/val.txt --out-dir data/processed/coco_subset_30000_physical
"""
import argparse
import shutil
from pathlib import Path

import yaml


def label_path(image: Path) -> Path:
    """Convenzione di Ultralytics: l'ultimo /images/ del percorso diventa /labels/, estensione .txt."""
    parts = image.parts
    for i in range(len(parts) - 2, -1, -1):
        if parts[i] == "images":
            return Path(*parts[:i], "labels", *parts[i + 1:]).with_suffix(".txt")
    raise SystemExit(f"Nel percorso non c'è una cartella images/: {image}")


def materialize(list_file: Path, out_dir: Path) -> int:
    images = [Path(l.strip()) for l in list_file.read_text().splitlines() if l.strip()]
    unique = list(dict.fromkeys(images))
    by_name = {}
    for image in unique:
        if by_name.setdefault(image.name, image) != image:
            raise SystemExit(f"{list_file}: due immagini diverse si chiamano {image.name} "
                             f"({by_name[image.name]}, {image}), la copia piatta le sovrascriverebbe")
    if len(unique) != len(images):
        print(f"  {list_file.name}: {len(images) - len(unique)} righe ripetute, copiate una volta sola")
    images = unique
    for sub in ("images", "labels"):
        shutil.rmtree(out_dir / sub, ignore_errors=True)
        (out_dir / sub).mkdir(parents=True)
    for i, image in enumerate(images, 1):
        if not image.exists():
            raise SystemExit(f"Immagine non trovata (symlink rotto?): {image}")
        shutil.copyfile(image, out_dir / "images" / image.name)  # copyfile segue i symlink
        label = label_path(image)
        target = out_dir / "labels" / f"{image.stem}.txt"
        if label.exists():
            shutil.copyfile(label, target)
        else:
            target.write_text("")
        if i % 5000 == 0:
            print(f"  {i}/{len(images)}")
    return len(images)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", type=Path, help="Lista delle immagini di train (train.txt)")
    parser.add_argument("--val", type=Path, help="Lista delle immagini di val (val.txt)")
    parser.add_argument("--data-yaml", type=Path,
                         help="data.yaml da cui copiare nc e names (default: data.yaml accanto alla lista)")
    parser.add_argument("--out-dir", type=Path, required=True, help="Dove scrivere train/ e val/")
    args = parser.parse_args()
    if not args.train and not args.val:
        raise SystemExit("Serve almeno --train o --val")

    data_yaml = args.data_yaml or (args.train or args.val).parent / "data.yaml"
    if not data_yaml.is_file():
        raise SystemExit(f"data.yaml di origine non trovato: {data_yaml} (serve per nc e names: usare --data-yaml)")
    source = yaml.safe_load(data_yaml.read_text())

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_yaml = {}
    for split, list_file in (("train", args.train), ("val", args.val)):
        if not list_file:
            continue
        if not list_file.is_file():
            raise SystemExit(f"Lista non trovata: {list_file}")
        n = materialize(list_file, args.out_dir / split)
        print(f"{split}: {n} immagini -> {args.out_dir / split}")
        out_yaml[split] = f"{split}/images"
    out_yaml.update(nc=source["nc"], names=source["names"])
    (args.out_dir / "data.yaml").write_text(yaml.safe_dump(out_yaml, sort_keys=False, allow_unicode=True))
    print(f"data.yaml -> {args.out_dir / 'data.yaml'}")


if __name__ == "__main__":
    main()
