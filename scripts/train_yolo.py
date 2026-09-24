#!/usr/bin/env python3
"""Avvia un training Ultralytics YOLO su un dataset train/valid (es. quello
prodotto da split_dataset.py), con i parametri principali esposti da riga di
comando.

Con --classes si allena su un sottoinsieme delle classi del data.yaml
sorgente invece che su tutte: utile, per esempio, per un primo training
sulla sola classe escooter (80) mentre il resto del dataset (annotazione
COCO, revisione manuale) è ancora in corso. Ultralytics non ha un parametro
di training per "usa solo queste classi": --single-cls tratterebbe invece
ogni bbox esistente come un'unica classe, mescolando escooter con auto,
persone ecc. Con --classes questo script prepara perciò un dataset filtrato
a parte: per ogni split (train/valid), tiene solo le righe label con classe
in --classes e le rimappa a 0..N-1 nell'ordine dato (--classes 80 -> la
sola classe 80 diventa la classe 0), scrive un nuovo data.yaml con nc/names
coerenti, e allena su quello. Le immagini non vengono duplicate (sono
identiche, cambiano solo le label): sono linkate nella cartella filtrata,
con fallback a una copia se il filesystem non supporta i link simbolici.

La cartella filtrata viene rigenerata a ogni esecuzione (--filtered-dir,
default accanto al data.yaml sorgente).

Avvio:
    python3 scripts/train_yolo.py --data DIR/data.yaml --model yolo11l.pt
        --epochs 100 --batch 16 [--freeze N] [--imgsz 640] [--device cpu|0]
        [--classes 80] [--filtered-dir DIR] [--project DIR] [--name NAME]
        [--patience N] [--workers N] [--seed N] [--resume]
"""
import argparse
import shutil
from pathlib import Path

import yaml


def load_data_yaml(path: Path) -> dict:
    data = yaml.safe_load(path.read_text())
    base = Path(data.get("path", path.parent))
    if not base.is_absolute():
        base = (path.parent / base).resolve()
    return {
        "base": base,
        "train": (base / data["train"]).resolve(),
        "val": (base / data[key]).resolve() if (key := next((k for k in ("val", "valid") if k in data), None)) else None,
        "names": {int(k): v for k, v in data["names"].items()} if isinstance(data["names"], dict)
                 else dict(enumerate(data["names"])),
    }


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    try:
        dst.symlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def filter_split(images_dir: Path, classes: list, out_dir: Path) -> int:
    """Filtra e rimappa le label di uno split (images_dir/../labels) sulle
    sole classi in classes (rimappate a 0..N-1 nell'ordine dato), scrivendo
    immagini (link) e label in out_dir/images, out_dir/labels. Ritorna il
    numero di immagini copiate (tutte: anche quelle senza istanze delle
    classi scelte restano, con label vuoto, come nel resto della pipeline)."""
    labels_dir = images_dir.parent / "labels"
    remap = {cls: i for i, cls in enumerate(classes)}
    out_images, out_labels = out_dir / "images", out_dir / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    names = sorted(p.name for p in images_dir.iterdir())
    for name in names:
        link_or_copy(images_dir / name, out_images / name)
        src_label = labels_dir / f"{Path(name).stem}.txt"
        kept_lines = []
        if src_label.exists():
            for line in src_label.read_text().splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                cls = int(float(parts[0]))
                if cls in remap:
                    kept_lines.append(f"{remap[cls]} {' '.join(parts[1:])}")
        dst_label = out_labels / f"{Path(name).stem}.txt"
        dst_label.write_text("\n".join(kept_lines) + "\n" if kept_lines else "")
    return len(names)


def build_filtered_dataset(source_yaml: Path, classes: list, filtered_dir: Path) -> Path:
    data = load_data_yaml(source_yaml)
    unknown = [c for c in classes if c not in data["names"]]
    if unknown:
        raise SystemExit(f"Classi non presenti in {source_yaml}: {unknown} (disponibili: {sorted(data['names'])})")

    if filtered_dir.exists():
        shutil.rmtree(filtered_dir)

    n_train = filter_split(data["train"], classes, filtered_dir / "train")
    print(f"Train: {n_train} immagini filtrate su classi {classes} -> {filtered_dir / 'train'}")
    has_val = data["val"] is not None and data["val"].is_dir()
    if has_val:
        n_val = filter_split(data["val"], classes, filtered_dir / "valid")
        print(f"Valid: {n_val} immagini filtrate su classi {classes} -> {filtered_dir / 'valid'}")

    filtered_yaml = filtered_dir / "data.yaml"
    lines = [
        f"path: {filtered_dir.resolve()}",
        "train: train/images",
    ]
    if has_val:
        lines.append("val: valid/images")
    lines += [f"nc: {len(classes)}", "names:"]
    lines += [f"  {i}: {data['names'].get(cls, str(cls))}" for i, cls in enumerate(classes)]
    filtered_yaml.write_text("\n".join(lines) + "\n")
    print(f"data.yaml filtrato scritto in {filtered_yaml}")
    return filtered_yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True, help="data.yaml del dataset (es. train/valid di split_dataset.py)")
    parser.add_argument("--model", default="yolo11l.pt", help="Modello pretrained Ultralytics di partenza (default: yolo11l.pt)")
    parser.add_argument("--epochs", type=int, default=100, help="Numero di epoche (default: 100)")
    parser.add_argument("--batch", type=int, default=-1, help="Dimensione batch (default: -1)")
    parser.add_argument("--freeze", type=int, default=None,
                         help="Congela i primi N layer del modello pretrained (default: nessun freeze)")
    parser.add_argument("--imgsz", type=int, default=640, help="Dimensione immagine di training (default: 640)")
    parser.add_argument("--device", default=None,
                         help="Device Ultralytics (es. 0, 0,1, cpu); default: scelto automaticamente da Ultralytics")
    parser.add_argument("--classes", type=int, nargs="+", default=None,
                         help="Se dato, allena solo su queste classi del data.yaml sorgente, rimappate a 0..N-1 "
                              "nell'ordine dato (es. --classes 80 per la sola classe escooter)")
    parser.add_argument("--filtered-dir", type=Path, default=None,
                         help="Cartella per il dataset filtrato quando --classes è dato "
                              "(default: <cartella data.yaml>_classes_<id1-id2-...>)")
    parser.add_argument("--project", default="data/runs", help="Cartella dei risultati Ultralytics (default: data/runs)")
    parser.add_argument("--name", default=None, help="Nome della run (default: scelto da Ultralytics, train/train2/...)")
    parser.add_argument("--patience", type=int, default=None, help="Epoche di pazienza per l'early stopping (default: Ultralytics, 100)")
    parser.add_argument("--workers", type=int, default=None, help="Worker del dataloader (default: Ultralytics, 8)")
    parser.add_argument("--seed", type=int, default=None, help="Seed di training (default: Ultralytics, 0)")
    parser.add_argument("--resume", action="store_true", help="Riprende l'ultimo training interrotto in --project/--name")
    args = parser.parse_args()

    data_yaml = args.data.resolve()
    if not data_yaml.is_file():
        raise SystemExit(f"data.yaml non trovato: {data_yaml}")

    if args.classes:
        filtered_dir = (args.filtered_dir or
                         data_yaml.parent.parent / f"{data_yaml.parent.name}_classes_{'-'.join(map(str, args.classes))}").resolve()
        train_yaml = build_filtered_dataset(data_yaml, args.classes, filtered_dir)
    else:
        train_yaml = data_yaml

    from ultralytics import YOLO

    model = YOLO(args.model)
    train_kwargs = dict(
        data=str(train_yaml), epochs=args.epochs, batch=args.batch, imgsz=args.imgsz, project=args.project,
    )
    if args.freeze is not None:
        train_kwargs["freeze"] = args.freeze
    if args.device is not None:
        train_kwargs["device"] = args.device
    if args.name is not None:
        train_kwargs["name"] = args.name
    if args.patience is not None:
        train_kwargs["patience"] = args.patience
    if args.workers is not None:
        train_kwargs["workers"] = args.workers
    if args.seed is not None:
        train_kwargs["seed"] = args.seed
    if args.resume:
        train_kwargs["resume"] = True

    print(f"Training {args.model} su {train_yaml}")
    print(f"Parametri: { {k: v for k, v in train_kwargs.items() if k != 'data'} }")
    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
