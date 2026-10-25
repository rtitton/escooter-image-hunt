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

I dataset Roboflow però condividono foto fra loro (stessa foto con prefisso
dataset diverso) e includono copie capovolte ("<nome>_flip"): la
stratificazione per dataset, da sola, le spedirebbe in split diversi. Per
questo i gruppi vengono uniti anche fra dataset diversi quando hanno lo
stesso "nome-base" (nome senza prefisso dataset, senza suffisso Roboflow e
senza "_flip", v. base_key()), e l'intero gruppo unito va nello stesso
split. È un'euristica sul nome: non vede copie con nome diverso (per quelle
c'è split_review_app.py).

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
FLIP_SUFFIX_RE = re.compile(r"[ _-]?flip$", re.IGNORECASE)
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


def base_key(name: str) -> str:
    """Nome-base di un file per riconoscere la stessa foto in dataset
    diversi: senza prefisso "<dataset_id>__", senza estensione e suffisso
    Roboflow, senza "_flip" finale, minuscolo."""
    stem = RF_SUFFIX_RE.sub("", Path(name).stem)
    i = stem.find("__")
    if i != -1:
        stem = stem[i + 2:]
    return FLIP_SUFFIX_RE.sub("", stem).lower()


def copy_pair(name: str, src_images: Path, src_labels: Path, dst_images: Path, dst_labels: Path) -> None:
    shutil.copy2(src_images / name, dst_images / name)
    lbl_src = src_labels / f"{Path(name).stem}.txt"
    lbl_dst = dst_labels / f"{Path(name).stem}.txt"
    if lbl_src.exists():
        shutil.copy2(lbl_src, lbl_dst)
    else:
        lbl_dst.write_text("")


def split_names(names: list, valid_frac: float, seed: int, group_by_clip: bool) -> tuple[list, list, dict]:
    """Ritorna (train, valid, per_dataset): split stratificato per dataset
    sorgente, a livello di gruppo, cosi che né una clip video né copie della
    stessa foto (stesso base_key, anche fra dataset diversi) vengano spezzate
    fra i due split. Con group_by_clip=False si divide immagine per immagine.

    I gruppi (union-find: stessa clip nello stesso dataset, oppure stesso
    base_key in qualunque dataset) sono assegnati dataset per dataset con un
    bilanciamento greedy per deficit (euristica LPT, "longest processing time
    first"): ordinati dal più grande al più piccolo, ogni gruppo va allo split
    che ne ha più bisogno per avvicinarsi al proprio target (target -
    assegnate finora). Necessario perché diversi dataset sorgente sono quasi
    interamente un'unica clip lunga (centinaia di frame consecutivi): un
    riempimento ingenuo "valid finché non raggiunge la soglia" può far finire
    un intero gruppo enorme (e quindi l'intero dataset) tutto in un solo
    split. Un gruppo già assegnato mentre si elaborava un dataset precedente
    resta dov'è: per i dataset successivi conta come già assegnato nei loro
    target (quindi i target per dataset sono rispettati solo
    approssimativamente quando ci sono gruppi che attraversano i dataset)."""
    rng = random.Random(seed)

    parent = {n: n for n in names}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union_by(key_fn):
        first = {}
        for n in names:
            k = key_fn(n)
            if k in first:
                parent[find(n)] = find(first[k])
            else:
                first[k] = n

    if group_by_clip:
        union_by(lambda n: ("clip", dataset_of(n), clip_of(n)))
        union_by(lambda n: ("base", base_key(n)))

    members_of: dict[str, list] = {}
    for n in names:
        members_of.setdefault(find(n), []).append(n)

    by_dataset: dict[str, dict] = {}  # ds -> {root: [nomi del gruppo in questo dataset]}
    for n in names:
        by_dataset.setdefault(dataset_of(n), {}).setdefault(find(n), []).append(n)

    side: dict[str, str] = {}  # root -> "train" | "valid"
    per_dataset = {}
    for ds in sorted(by_dataset):
        group_lists = list(by_dataset[ds].items())
        rng.shuffle(group_lists)
        group_lists.sort(key=lambda kv: len(kv[1]), reverse=True)  # LPT: i più grandi per primi
        n_images = sum(len(g) for _, g in group_lists)
        target_valid = round(n_images * valid_frac)
        target_train = n_images - target_valid
        n_train = sum(len(g) for r, g in group_lists if side.get(r) == "train")
        n_valid = sum(len(g) for r, g in group_lists if side.get(r) == "valid")
        for root, g in group_lists:
            if root in side:
                continue
            if (target_valid - n_valid) > (target_train - n_train):
                side[root] = "valid"
                n_valid += len(g)
            else:
                side[root] = "train"
                n_train += len(g)
        per_dataset[ds] = (n_train, n_valid)

    train = [n for r, ms in members_of.items() if side[r] == "train" for n in ms]
    valid = [n for r, ms in members_of.items() if side[r] == "valid" for n in ms]
    return sorted(train), sorted(valid), per_dataset


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
