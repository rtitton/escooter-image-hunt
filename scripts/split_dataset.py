#!/usr/bin/env python3
"""Divide un dataset YOLO (images/ + labels/) in train/valid, in vista del
training: copia i file in <out-dir>/train/{images,labels} e
<out-dir>/valid/{images,labels}, più un data.yaml pronto per Ultralytics
(classi COCO 0-79 + escooter 80, lo schema fisso di questo progetto).

Lo split è stratificato per dataset sorgente (dedotto dal prefisso
"<dataset_id>__" nel nome file, come nel resto della pipeline): fatto a
livello di intero dataset, un dataset sorgente piccolo potrebbe finire per
caso quasi tutto in un solo split. All'interno di ogni dataset sorgente, le
immagini che sembrano frame consecutivi della stessa ripresa (nomi tipo
frame_00010, stessa euristica di clip_and_frame() in select_images.py,
duplicata qui per non dipendere da quel modulo) vengono tenute nello stesso
split invece che assegnate una per una: frame vicini nel tempo sono
quasi-duplicati, e separarli fra train e valid farebbe trapelare
informazione (data leakage), gonfiando artificialmente le metriche di
validazione. Nomi non numerati restano ciascuno il proprio gruppo.

Split deterministico (--seed, default 42): a differenza del campionamento
senza seme di build_visual_check_sample.py, qui la riproducibilità conta —
per confrontare run di training diversi sugli stessi identici split.

Ad ogni esecuzione la cartella di output viene svuotata e ripopolata.

Avvio:
    python3 scripts/split_dataset.py <dataset_dir> [--out-dir DIR]
        [--valid-frac F] [--seed N] [--no-clip-grouping]
"""
import argparse
import random
import re
import shutil
from pathlib import Path

COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]
CLASS_NAMES = COCO_NAMES + ["escooter"]  # id 80, coerente con config.ESCOOTER_CLASS_ID

RF_SUFFIX_RE = re.compile(r"_(?:jpe?g|png)\.rf\.[0-9a-f]{32}$", re.IGNORECASE)
CLIP_FRAME_RE = re.compile(r"^(?P<clip>.*?)[ _-]?(?P<idx>\d+)$")
MAX_PLAUSIBLE_FRAME_IDX = 100_000_000  # oltre questo il "numero" nel nome è un timestamp, non un indice di frame


def dataset_of(name: str) -> str:
    i = name.find("__")
    return name[:i] if i != -1 else name


def clip_of(name: str) -> str:
    """Identità del gruppo video-clip di un file (stessa euristica di
    select_images.clip_and_frame, duplicata qui per non dipendere da quel
    modulo): nomi senza un numero finale plausibile restano ciascuno il
    proprio gruppo (il loro stem)."""
    stem = RF_SUFFIX_RE.sub("", Path(name).stem)
    m = CLIP_FRAME_RE.match(stem)
    if not m:
        return stem
    idx = int(m.group("idx"))
    if idx > MAX_PLAUSIBLE_FRAME_IDX:
        return stem
    return m.group("clip") or "_"


def copy_pair(name: str, src_images: Path, src_labels: Path, dst_images: Path, dst_labels: Path) -> None:
    shutil.copy2(src_images / name, dst_images / name)
    lbl_src = src_labels / f"{Path(name).stem}.txt"
    lbl_dst = dst_labels / f"{Path(name).stem}.txt"
    if lbl_src.exists():
        shutil.copy2(lbl_src, lbl_dst)
    else:
        lbl_dst.write_text("")


def split_names(names: list, valid_frac: float, seed: int, group_by_clip: bool) -> tuple[list, list]:
    """Ritorna (train, valid): split stratificato per dataset sorgente, a
    livello di gruppo (dataset, clip) cosi che una stessa clip video non
    venga spezzata fra i due split.

    Gruppi assegnati con un bilanciamento greedy per deficit (euristica LPT,
    "longest processing time first"): ordinati dal più grande al più
    piccolo, ogni gruppo va allo split che ne ha più bisogno per avvicinarsi
    al proprio target (target - assegnate finora). Necessario perché diversi
    dataset sorgente sono quasi interamente un'unica clip lunga (centinaia
    di frame consecutivi): un riempimento ingenuo "valid finché non
    raggiunge la soglia" può far finire un intero gruppo enorme (e quindi
    l'intero dataset) tutto in un solo split, sbilanciando pesantemente la
    proporzione complessiva. Con questa euristica i gruppi più grandi
    finiscono di norma in train (che parte con un target maggiore), lasciando
    valid composto da un mix di gruppi più piccoli e vari — comunque
    l'unico esito possibile, senza leakage, quando un dataset è di fatto
    una sola clip: va tutto da una parte."""
    rng = random.Random(seed)

    groups: dict[tuple, list] = {}
    for name in names:
        key = (dataset_of(name), name if not group_by_clip else clip_of(name))
        groups.setdefault(key, []).append(name)

    by_dataset: dict[str, list] = {}
    for (ds, _clip), members in groups.items():
        by_dataset.setdefault(ds, []).append(members)

    train, valid = [], []
    per_dataset = {}
    for ds in sorted(by_dataset):
        group_lists = by_dataset[ds]
        rng.shuffle(group_lists)
        group_lists.sort(key=len, reverse=True)  # LPT: i più grandi assegnati per primi
        n_images = sum(len(g) for g in group_lists)
        target_valid = round(n_images * valid_frac)
        target_train = n_images - target_valid
        ds_train, ds_valid = [], []
        for g in group_lists:
            if (target_valid - len(ds_valid)) > (target_train - len(ds_train)):
                ds_valid.extend(g)
            else:
                ds_train.extend(g)
        train.extend(ds_train)
        valid.extend(ds_valid)
        per_dataset[ds] = (len(ds_train), len(ds_valid))
    return train, valid, per_dataset


def write_data_yaml(out_dir: Path) -> None:
    lines = [
        f"path: {out_dir}",
        "train: train/images",
        "val: valid/images",
        f"nc: {len(CLASS_NAMES)}",
        "names:",
    ] + [f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES)]
    (out_dir / "data.yaml").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset_dir", type=Path, help="Dataset sorgente, con le sottocartelle images/ e labels/")
    parser.add_argument("--out-dir", type=Path, default=None,
                         help="Cartella di destinazione (default: <dataset_dir>_split)")
    parser.add_argument("--valid-frac", type=float, default=0.15,
                         help="Frazione di immagini (per dataset sorgente) destinata a valid (default: 0.15)")
    parser.add_argument("--seed", type=int, default=42,
                         help="Seed per lo split, per risultati riproducibili (default: 42)")
    parser.add_argument("--no-clip-grouping", action="store_true",
                         help="Non tenere insieme le immagini della stessa clip video: split immagine per immagine")
    args = parser.parse_args()

    if not (0.0 < args.valid_frac < 1.0):
        raise SystemExit("--valid-frac deve essere tra 0 e 1")

    source_dir = args.dataset_dir.resolve()
    src_images, src_labels = source_dir / "images", source_dir / "labels"
    if not src_images.is_dir() or not src_labels.is_dir():
        raise SystemExit(f"{source_dir} deve contenere le sottocartelle images/ e labels/")

    names = sorted(p.name for p in src_images.iterdir())
    if not names:
        raise SystemExit(f"Nessuna immagine trovata in {src_images}")

    train_names, valid_names, per_dataset = split_names(
        names, args.valid_frac, args.seed, group_by_clip=not args.no_clip_grouping,
    )

    out_dir = (args.out_dir or Path(f"{source_dir}_split")).resolve()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    for split, split_names_ in (("train", train_names), ("valid", valid_names)):
        (out_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (out_dir / split / "labels").mkdir(parents=True, exist_ok=True)
        for name in split_names_:
            copy_pair(name, src_images, src_labels, out_dir / split / "images", out_dir / split / "labels")

    write_data_yaml(out_dir)

    print(f"Sorgente: {source_dir} ({len(names)} immagini)")
    print(f"Train: {len(train_names)}   Valid: {len(valid_names)} "
          f"({len(valid_names) / len(names) * 100:.1f}%)")
    print()
    for ds in sorted(per_dataset):
        t, v = per_dataset[ds]
        print(f"  {ds}: {t} train / {v} valid")
    print(f"\nScritto in {out_dir} (data.yaml incluso)")


if __name__ == "__main__":
    main()
