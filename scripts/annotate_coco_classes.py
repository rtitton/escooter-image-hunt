#!/usr/bin/env python3
"""Annota tutte le classi COCO (0-79) sul dataset finale, in aggiunta alla
classe escooter (80) già presente nelle label.

Usa lo stesso modello Ultralytics pretrained di grandi dimensioni già usato
dal filtro varietà di select_images.py (config.COCO_MODEL, di norma
yolo11l.pt), ma su una cache dedicata: a differenza di quella del filtro
varietà, qui si salva anche la confidence di ogni detection (non solo classe
e bbox), a una soglia di cattura bassa (COCO_ANNOTATION_CAPTURE_CONF). Questo
separa il costo dell'inferenza (fatta una volta, in cache) dalla soglia di
confidenza effettiva (COCO_ANNOTATION_CONF_THRESHOLD, applicata in scrittura
e ritarabile a piacere senza ricalcolare nulla) — importante perché qui le
detection diventano etichette di training permanenti: un falso positivo pesa
più di un mancato rilevamento, quindi la soglia effettiva è più
conservativa del default Ultralytics (0.25).

Una detection di classe "sosia" del monopattino (COCO_ESCOOTER_LOOKALIKE_
CLASSES: bici, moto, colonnina/parchimetro, skateboard, snowboard) viene
scartata se la sua bbox è coperta per più di COCO_ESCOOTER_OVERLAP_THRESHOLD
dall'unione delle bbox escooter dell'immagine (frazione di area della
detection, non IoU simmetrico, non containment contro una singola bbox — v.
union_containment_ratio()/select_coco_boxes()): in quel caso è
quasi certamente lo stesso oggetto fisico del monopattino, tenuto per
intero o in parte (es. quando il modello rileva solo il pianale,
"skateboard"/"snowboard", o solo lo stelo/manubrio, "parking meter" — un
oggetto verticale sottile scambiato per un altro), e tenerla creerebbe due
etichette contraddittorie sulla stessa area. Il
controllo è limitato a quelle classi apposta: esteso a tutte scarterebbe
anche oggetti reali chiaramente distinti (un'auto, una borsa, una panchina
sullo sfondo...) che ricadono per intero nella bbox escooter solo per
prospettiva/profondità, non perché coincidano fisicamente con essa — errore
osservato in pratica nella prima versione di questo script, con containment
non ristretto alle classi sosia. Nessuna classe è mai scartata per questo
motivo se non è nella lista, "persona" compresa: il conducente è una
detection legittima da annotare, non un duplicato.

Anche ristretto alle classi sosia, il criterio resta geometrico e non
infallibile: un oggetto sosia reale (es. una bici vera) parcheggiato
proprio dietro/accanto al monopattino può avere la stessa containment di un
vero doppione, e verrebbe scartato per errore — non c'è una soglia che
separi in modo pulito i due casi (verificato empiricamente: i falsi scarti
non si concentrano a containment basso). Ogni detection scartata per questo
motivo viene perciò segnalata, non solo buttata via, in FLAGGED_COCO_
OVERLAP_PATH (default data/flagged_coco_overlap.txt), con classe,
confidence e containment. Con --export-flagged-sample N si esporta un
campione casuale con le bbox disegnate (escooter in rosso, scartata in
arancione) per la revisione visiva; le immagini per cui si decide di
recuperare le detection sosia vanno aggiunte (un nome per riga) a
COCO_OVERLAP_RESCUE_PATH (default data/coco_overlap_rescue.txt) — al run
successivo lo scarto per overlap non viene applicato per quelle immagini
(rilancio istantaneo, riusa la cache, non serve rifare l'inferenza).

Indipendentemente da confidenza e overlap, le detection di una classe
"implausibile" in una scena esterna (COCO_IMPLAUSIBLE_CLASSES, default
oggetti da interno come "toilet", "couch", "tv", "sink": v. scripts/.env)
vengono sempre scartate: sono quasi sempre un errore di classificazione ad
alta confidenza del modello (es. un cestino/scatola letto come "toilet"),
non intercettabile alzando la sola soglia generale senza perdere molto
recall altrove. Lista modificabile in scripts/.env (COCO_IMPLAUSIBLE_
CLASSES, id separati da virgola) o disattivabile per una singola esecuzione
con --no-implausible-filter (stringa vuota in .env per disattivarla sempre).

Solo l'orientazione nativa dell'immagine (le immagini di union_reviewed_
phashdedup non sono ruotate/corrette). Le righe classe 80 già presenti nelle
label vengono riscritte tali e quali (rilette dal file sorgente, non dalla
cache); eventuali righe di classi COCO da un'esecuzione precedente di questo
script vengono ignorate e ricalcolate, per idempotenza.

Ad ogni esecuzione la cartella di output viene svuotata e ripopolata.

Avvio:
    python3 scripts/annotate_coco_classes.py
        [--source-dir DIR] [--out-dir DIR] [--cache-path FILE] [--model NAME]
        [--batch-size N] [--capture-conf F] [--conf-threshold F]
        [--overlap-threshold F] [--no-implausible-filter] [--refresh-cache]
        [--limit N] [--sample N] [--sample-dir DIR] [--flagged-path FILE]
        [--rescue-path FILE] [--export-flagged-sample N] [--flagged-sample-dir DIR]
"""
import argparse
import json
import random
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

import config

ESCOOTER_CLASS_ID = config.ESCOOTER_CLASS_ID
PERSON_CLASS_ID = config.PERSON_CLASS_ID
COCO_ESCOOTER_LOOKALIKE_CLASSES = config.COCO_ESCOOTER_LOOKALIKE_CLASSES
BOX_COLOR = config.BOX_COLOR
PERSON_BOX_COLOR = config.PERSON_BOX_COLOR
COCO_BOX_COLOR = config.COCO_BOX_COLOR
FLAGGED_BOX_COLOR = config.COCO_FLAGGED_BOX_COLOR
BOX_WIDTH = config.BOX_WIDTH
LABEL_FONT = ImageFont.load_default(size=20)


def load_json_cache(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def save_json_cache(cache: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2))


def cache_entry_is_valid(entry: dict, image_path: Path, capture_conf: float) -> bool:
    """Invalida la entry se l'immagine è cambiata (mtime/size) o se è stata
    calcolata con una capture conf più alta di quella richiesta ora (in tal
    caso in cache potrebbero mancare detection sotto quella soglia più alta,
    ora richieste)."""
    stat = image_path.stat()
    if entry.get("mtime") != stat.st_mtime or entry.get("size") != stat.st_size:
        return False
    return entry.get("capture_conf", 1.0) <= capture_conf


def detections_from_result(result) -> list:
    """[classe, confidence, x1, y1, x2, y2] per ogni detection, JSON-serializzabili."""
    if result.boxes is None:
        return []
    return [
        [int(cls), conf, *xyxy]
        for cls, conf, xyxy in zip(
            result.boxes.cls.tolist(), result.boxes.conf.tolist(), result.boxes.xyxy.tolist()
        )
    ]


def build_cache(
    names: list[str], images_dir: Path, cache_path: Path, model_name: str, batch_size: int,
    capture_conf: float, refresh_cache: bool,
) -> dict:
    """Ritorna (cache, class_names). class_names (int -> nome) è preso dal
    modello quando viene caricato per inferenza, e altrimenti da una entry
    "__class_names__" salvata in cache alla prima esecuzione: così i nomi
    restano disponibili per i log anche in una run successiva interamente
    servita dalla cache, senza dover ricaricare il modello solo per quello."""
    cache = load_json_cache(cache_path)

    if refresh_cache:
        to_infer = names
    else:
        to_infer = [
            n for n in names
            if n not in cache or not cache_entry_is_valid(cache[n], images_dir / n, capture_conf)
        ]

    if to_infer:
        from tqdm import tqdm
        from ultralytics import YOLO

        print(f"Annotazione COCO: {len(names) - len(to_infer)}/{len(names)} immagini già in cache, "
              f"inferenza YOLO ({model_name}) su {len(to_infer)}.")
        model = YOLO(model_name)
        try:
            for batch_i, start in enumerate(
                tqdm(range(0, len(to_infer), batch_size), desc="annotazione COCO (inferenza)", unit="batch")
            ):
                batch = to_infer[start:start + batch_size]
                images = [Image.open(images_dir / n).convert("RGB") for n in batch]
                results = model.predict(images, conf=capture_conf, verbose=False)
                for im in images:
                    im.close()

                for name, result in zip(batch, results):
                    stat = (images_dir / name).stat()
                    h_px, w_px = result.orig_shape
                    cache[name] = {
                        "mtime": stat.st_mtime,
                        "size": stat.st_size,
                        "capture_conf": capture_conf,
                        "orig_shape": [h_px, w_px],
                        "detections": detections_from_result(result),
                    }

                if (batch_i + 1) % config.COCO_ANNOTATION_CACHE_SAVE_EVERY == 0:
                    save_json_cache(cache, cache_path)
        finally:
            cache["__class_names__"] = {str(k): v for k, v in model.names.items()}
            save_json_cache(cache, cache_path)
        return cache, model.names

    class_names = {int(k): v for k, v in cache.get("__class_names__", {}).items()}
    return cache, class_names


def escooter_lines_and_boxes(label_path: Path, w_px: int, h_px: int) -> tuple[list[str], list[tuple]]:
    """Righe originali (testuali) di classe escooter da preservare tali e
    quali, e le stesse bbox convertite in pixel xyxy per il controllo di
    overlap. Ignora eventuali righe di classi COCO scritte da un'esecuzione
    precedente di questo script."""
    if not label_path.exists():
        return [], []
    lines, boxes = [], []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 5 or int(float(parts[0])) != ESCOOTER_CLASS_ID:
            continue
        lines.append(line)
        xc, yc, bw, bh = (float(p) * s for p, s in zip(parts[1:], (w_px, h_px, w_px, h_px)))
        boxes.append((xc - bw / 2, yc - bh / 2, xc + bw / 2, yc + bh / 2))
    return lines, boxes


def containment_ratio(box_xyxy: tuple, other_xyxy: tuple) -> float:
    """Frazione dell'area di box_xyxy coperta dall'intersezione con
    other_xyxy (stessa definizione di select_images.containment_ratio)."""
    x1, y1 = max(box_xyxy[0], other_xyxy[0]), max(box_xyxy[1], other_xyxy[1])
    x2, y2 = min(box_xyxy[2], other_xyxy[2]), min(box_xyxy[3], other_xyxy[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area = (box_xyxy[2] - box_xyxy[0]) * (box_xyxy[3] - box_xyxy[1])
    return inter / area if area > 0 else 0.0


def union_containment_ratio(box_xyxy: tuple, others: list, grid: int = 32) -> float:
    """Frazione dell'area di box_xyxy coperta dall'UNIONE di others, non dal
    singolo che la copre di più (containment_ratio prende il massimo su un
    solo other alla volta). In una fila di monopattini ravvicinati (dock di
    sharing, parcheggi fitti) una detection sosia può ricadere a cavallo di
    due bbox escooter adiacenti: contro ciascuna singolarmente la
    containment resta bassa (l'area si divide fra le due), pur essendo
    l'intera detection "dentro qualche escooter" — caso osservato in
    pratica, che containment_ratio da solo non cattura. Calcolata per
    campionamento su una griglia grid x grid nella bbox (esatta a meno
    dell'errore di discretizzazione, ~1/grid): sufficiente per una
    decisione di soglia, evitando la geometria esatta dell'unione di
    rettangoli per un numero di box comunque piccolo per immagine."""
    x1, y1, x2, y2 = box_xyxy
    if x2 <= x1 or y2 <= y1 or not others:
        return 0.0
    relevant = [o for o in others if not (o[2] <= x1 or o[0] >= x2 or o[3] <= y1 or o[1] >= y2)]
    if not relevant:
        return 0.0
    hits = 0
    for i in range(grid):
        px = x1 + (i + 0.5) * (x2 - x1) / grid
        for j in range(grid):
            py = y1 + (j + 0.5) * (y2 - y1) / grid
            if any(o[0] <= px <= o[2] and o[1] <= py <= o[3] for o in relevant):
                hits += 1
    return hits / (grid * grid)


def select_coco_boxes(
    detections: list, escooter_boxes: list, conf_threshold: float, overlap_threshold: float,
    implausible_classes: set, rescue: bool = False,
) -> tuple[list, list, int]:
    """Detection [classe, confidence, xyxy] sopra soglia di confidenza,
    scartando:
    - quelle di classe "sosia" del monopattino (COCO_ESCOOTER_LOOKALIKE_
      CLASSES: bici, moto, colonnina/parchimetro, skateboard, snowboard) quasi
      interamente contenute in una bbox escooter — probabile stesso oggetto
      fisico misclassificato, es. il solo pianale del monopattino letto come
      "skateboard", o lo stelo/manubrio letto come "parking meter". Si usa la
      frazione di area della detection COCO coperta dall'UNIONE delle bbox
      escooter dell'immagine (non l'IoU simmetrico, non il containment
      contro una singola bbox — v. union_containment_ratio()): in una fila
      di monopattini ravvicinati una detection sosia può ricadere a cavallo
      di due bbox escooter adiacenti, con containment basso contro ciascuna
      singolarmente pur essendo quasi interamente coperta da qualche
      escooter nel complesso — caso osservato in pratica, da cui il calcolo
      sull'unione invece che sul singolo migliore. Il controllo è limitato a
      quelle classi apposta: esteso a tutte scarterebbe anche oggetti reali
      chiaramente distinti (es. un'auto o una borsa sullo sfondo) che
      ricadono per intero nella bbox escooter solo per prospettiva, non
      perché coincidano fisicamente con essa. Non è comunque un criterio
      geometrico infallibile (un oggetto sosia reale parcheggiato proprio
      dietro/accanto al monopattino può avere la stessa containment di un
      vero doppione), da qui il meccanismo di rescue sotto;
    - quelle di classe implausibile in una scena esterna (implausible_
      classes, es. "toilet", "couch", "tv": oggetti da interno), a
      prescindere dalla confidenza: un errore di classificazione ad alta
      confidenza non è intercettabile alzando la sola soglia generale senza
      perdere molto recall altrove (v. COCO_IMPLAUSIBLE_CLASSES).

    Se rescue è True (immagine presente in COCO_OVERLAP_RESCUE_PATH, dopo
    revisione manuale) lo scarto per overlap sosia non viene applicato — non
    tocca il filtro implausibilità, indipendente da questo problema.

    Ritorna (detection tenute, detection scartate per overlap con dettaglio
    [classe, confidence, box, containment] per il log/campione di revisione,
    conteggio scartate per implausibilità)."""
    kept, discarded_overlap, discarded_implausible = [], [], 0
    for cls, conf, x1, y1, x2, y2 in detections:
        if conf < conf_threshold:
            continue
        if cls in implausible_classes:
            discarded_implausible += 1
            continue
        box = (x1, y1, x2, y2)
        if cls in COCO_ESCOOTER_LOOKALIKE_CLASSES and escooter_boxes:
            containment = union_containment_ratio(box, escooter_boxes)
            if containment > overlap_threshold:
                if not rescue:
                    discarded_overlap.append((cls, conf, box, containment))
                    continue
        kept.append((cls, conf, box))
    return kept, discarded_overlap, discarded_implausible


def to_label_line(cls: int, box: tuple, w_px: int, h_px: int) -> str:
    x1, y1, x2, y2 = box
    xc, yc = (x1 + x2) / 2 / w_px, (y1 + y2) / 2 / h_px
    bw, bh = (x2 - x1) / w_px, (y2 - y1) / h_px
    return f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"


def draw_sample(image_path: Path, escooter_boxes: list, kept: list, names: dict, dest: Path) -> None:
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        draw = ImageDraw.Draw(im)
        for box in escooter_boxes:
            draw.rectangle(box, outline=BOX_COLOR, width=BOX_WIDTH)
        for cls, conf, box in kept:
            color = PERSON_BOX_COLOR if cls == PERSON_CLASS_ID else COCO_BOX_COLOR
            draw.rectangle(box, outline=color, width=BOX_WIDTH)
            label = f"{names.get(cls, cls)} {conf:.2f}"
            draw.text((box[0], box[1]), label, font=LABEL_FONT, fill=color, anchor="lb")
        im.save(dest)


def load_rescue_list(path: Path) -> set:
    """Nomi immagine (COCO_OVERLAP_RESCUE_PATH) per cui, dopo revisione
    manuale di FLAGGED_COCO_OVERLAP_PATH, lo scarto per overlap sosia va
    disattivato. Un nome per riga, righe vuote o che iniziano per # ignorate."""
    if not path.exists():
        return set()
    return {
        line.strip() for line in path.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    }


def draw_flagged(image_path: Path, escooter_boxes: list, flagged: list, names: dict, dest: Path) -> None:
    """Come draw_sample, ma per una detection scartata per overlap sosia (v.
    select_coco_boxes): utile per il campione di revisione manuale, disegna
    la bbox escooter (rossa) e quella scartata (arancione) con classe,
    confidence e containment, cosi da decidere a colpo d'occhio se si tratta
    di un vero doppione o di un oggetto sosia reale scartato per errore."""
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        draw = ImageDraw.Draw(im)
        for box in escooter_boxes:
            draw.rectangle(box, outline=BOX_COLOR, width=BOX_WIDTH)
        for cls, conf, box, containment in flagged:
            draw.rectangle(box, outline=FLAGGED_BOX_COLOR, width=BOX_WIDTH)
            label = f"{names.get(cls, cls)} conf={conf:.2f} containment={containment:.2f}"
            draw.text((box[0], box[1]), label, font=LABEL_FONT, fill=FLAGGED_BOX_COLOR, anchor="lb")
        im.save(dest)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-dir", type=Path, default=config.UNION_REVIEWED_PHASHDEDUP_DIR,
                         help=f"Dataset sorgente, con images/ e labels/ (default: {config.UNION_REVIEWED_PHASHDEDUP_DIR})")
    parser.add_argument("--out-dir", type=Path, default=config.UNION_REVIEWED_COCO_DIR,
                         help=f"Cartella di destinazione (default: {config.UNION_REVIEWED_COCO_DIR})")
    parser.add_argument("--cache-path", type=Path, default=config.COCO_ANNOTATION_CACHE_PATH,
                         help=f"Cache delle detection YOLO (default: {config.COCO_ANNOTATION_CACHE_PATH})")
    parser.add_argument("--model", default=config.COCO_MODEL,
                         help=f"Modello Ultralytics pretrained COCO (default: {config.COCO_MODEL})")
    parser.add_argument("--batch-size", type=int, default=config.COCO_BATCH_SIZE)
    parser.add_argument("--capture-conf", type=float, default=config.COCO_ANNOTATION_CAPTURE_CONF,
                         help="Soglia di confidenza usata (e salvata) in fase di inferenza "
                              f"(default: {config.COCO_ANNOTATION_CAPTURE_CONF})")
    parser.add_argument("--conf-threshold", type=float, default=config.COCO_ANNOTATION_CONF_THRESHOLD,
                         help="Soglia di confidenza effettiva applicata in scrittura, deve essere >= "
                              f"--capture-conf (default: {config.COCO_ANNOTATION_CONF_THRESHOLD})")
    parser.add_argument("--overlap-threshold", type=float, default=config.COCO_ESCOOTER_OVERLAP_THRESHOLD,
                         help="Frazione di area di una detection (non persona) coperta da una bbox escooter "
                              f"oltre la quale viene scartata (default: {config.COCO_ESCOOTER_OVERLAP_THRESHOLD})")
    parser.add_argument("--no-implausible-filter", action="store_true",
                         help="Disattiva lo scarto delle classi implausibili in una scena esterna "
                              "(config.COCO_IMPLAUSIBLE_CLASSES), a prescindere dalla confidenza")
    parser.add_argument("--refresh-cache", action="store_true",
                         help="Ignora la cache e ricalcola l'inferenza per tutte le immagini")
    parser.add_argument("--limit", type=int, default=None,
                         help="Elabora solo le prime N immagini (per test su campione)")
    parser.add_argument("--sample", type=int, default=0,
                         help="Dopo l'annotazione, esporta un campione casuale di N immagini con le bbox "
                              "disegnate (escooter, persona, altre classi COCO), per verifica visiva")
    parser.add_argument("--sample-dir", type=Path, default=None,
                         help="Cartella per il campione di verifica visiva (default: <out-dir>_sample)")
    parser.add_argument("--flagged-path", type=Path, default=config.FLAGGED_COCO_OVERLAP_PATH,
                         help="File con l'elenco delle detection scartate per overlap sosia, per revisione "
                              f"manuale (default: {config.FLAGGED_COCO_OVERLAP_PATH})")
    parser.add_argument("--rescue-path", type=Path, default=config.COCO_OVERLAP_RESCUE_PATH,
                         help="File con i nomi immagine (uno per riga) per cui, dopo revisione manuale di "
                              "--flagged-path, disattivare lo scarto per overlap sosia "
                              f"(default: {config.COCO_OVERLAP_RESCUE_PATH})")
    parser.add_argument("--export-flagged-sample", type=int, default=0,
                         help="Esporta un campione casuale di N detection scartate per overlap sosia, con le "
                              "bbox disegnate (escooter in rosso, scartata in arancione), per decidere quali "
                              "aggiungere a --rescue-path")
    parser.add_argument("--flagged-sample-dir", type=Path, default=None,
                         help="Cartella per il campione di revisione (default: <out-dir>_flagged_sample)")
    args = parser.parse_args()

    if args.conf_threshold < args.capture_conf:
        raise SystemExit(f"--conf-threshold ({args.conf_threshold}) deve essere >= --capture-conf ({args.capture_conf})")

    source_dir = args.source_dir.resolve()
    src_images, src_labels = source_dir / "images", source_dir / "labels"
    if not src_images.is_dir() or not src_labels.is_dir():
        raise SystemExit(f"{source_dir} deve contenere le sottocartelle images/ e labels/")

    names_list = sorted(p.name for p in src_images.iterdir())
    if args.limit:
        names_list = names_list[:args.limit]
    if not names_list:
        raise SystemExit(f"Nessuna immagine trovata in {src_images}")

    cache, class_names = build_cache(
        names_list, src_images, args.cache_path.resolve(), args.model, args.batch_size,
        args.capture_conf, args.refresh_cache,
    )
    class_names = class_names or {}
    implausible_classes = set() if args.no_implausible_filter else config.COCO_IMPLAUSIBLE_CLASSES
    if implausible_classes:
        labels = ", ".join(class_names.get(c, str(c)) for c in sorted(implausible_classes))
        print(f"Classi implausibili scartate a prescindere dalla confidenza: {labels}")

    rescued_names = load_rescue_list(args.rescue_path.resolve())
    if rescued_names:
        print(f"Immagini in {args.rescue_path} (overlap sosia disattivato per queste): {len(rescued_names)}")

    out_dir = args.out_dir.resolve()
    out_images, out_labels = out_dir / "images", out_dir / "labels"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    class_counts: dict[int, int] = {}
    total_discarded_overlap = 0
    total_discarded_implausible = 0
    total_kept = 0
    log_lines = []
    per_image_kept: dict[str, tuple[list, list]] = {}  # per --sample, se richiesto
    per_image_flagged: dict[str, tuple[list, list]] = {}  # (escooter_boxes, flagged) per immagine con scarti overlap
    all_flagged = []  # (name, cls, conf, containment) per il log di revisione

    for name in names_list:
        entry = cache[name]
        h_px, w_px = entry["orig_shape"]
        stem = Path(name).stem
        label_path = src_labels / f"{stem}.txt"

        escooter_lines, escooter_boxes = escooter_lines_and_boxes(label_path, w_px, h_px)
        kept, discarded_overlap, discarded_implausible = select_coco_boxes(
            entry["detections"], escooter_boxes, args.conf_threshold, args.overlap_threshold,
            implausible_classes, rescue=name in rescued_names,
        )
        total_discarded_overlap += len(discarded_overlap)
        total_discarded_implausible += discarded_implausible
        total_kept += len(kept)
        for cls, _conf, _box in kept:
            class_counts[cls] = class_counts.get(cls, 0) + 1
        if discarded_overlap:
            log_lines.append(f"{name}: {len(discarded_overlap)} detection scartate per overlap con bbox escooter")
            per_image_flagged[name] = (escooter_boxes, discarded_overlap)
            for cls, conf, box, containment in discarded_overlap:
                all_flagged.append((name, cls, conf, containment, to_label_line(cls, box, w_px, h_px)))
        if discarded_implausible:
            log_lines.append(f"{name}: {discarded_implausible} detection scartate per classe implausibile")

        coco_lines = [to_label_line(cls, box, w_px, h_px) for cls, _conf, box in kept]
        (out_labels / f"{stem}.txt").write_text("\n".join(escooter_lines + coco_lines) + "\n")
        shutil.copy2(src_images / name, out_images / name)

        if args.sample:
            per_image_kept[name] = (escooter_boxes, kept)

    print(f"\n{len(names_list)} immagini annotate in {out_dir}")
    print(f"Istanze COCO annotate: {total_kept} (soglia confidenza {args.conf_threshold})")
    print(f"Detection scartate per overlap con bbox escooter (copertura > {args.overlap_threshold}): {total_discarded_overlap}")
    print(f"Detection scartate per classe implausibile: {total_discarded_implausible}")
    print("\nIstanze per classe:")
    for cls in sorted(class_counts, key=lambda c: -class_counts[c]):
        label = class_names.get(cls, str(cls))
        print(f"  {label}: {class_counts[cls]}")

    log_path = config.DATA_ROOT / "logs" / "annotate_coco_classes.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        f"# Log annotate_coco_classes (conf_threshold={args.conf_threshold}, "
        f"overlap_threshold={args.overlap_threshold}, "
        f"implausible_classes={'disattivato' if not implausible_classes else sorted(implausible_classes)}, "
        f"modello={args.model})\n"
        f"# {len(names_list)} immagini, {total_kept} istanze COCO annotate, "
        f"{total_discarded_overlap} scartate per overlap, "
        f"{total_discarded_implausible} scartate per classe implausibile\n\n"
        + "\n".join(log_lines) + "\n"
    )
    print(f"\nLog scritto in {log_path}")

    flagged_path = args.flagged_path.resolve()
    flagged_path.parent.mkdir(parents=True, exist_ok=True)
    all_flagged.sort(key=lambda f: -f[2])  # confidence decrescente: i casi più "sicuri" del modello prima
    flagged_path.write_text(
        "# Detection di classe sosia del monopattino (bici/moto/parking meter/skateboard/snowboard)\n"
        "# scartate per overlap con una bbox escooter (v. select_coco_boxes in annotate_coco_classes.py):\n"
        "# non è un criterio geometrico infallibile, un oggetto sosia reale parcheggiato vicino/dietro il\n"
        "# monopattino può essere scartato per errore. Rivedi in review_app.py (la bbox scartata è\n"
        "# disegnata tratteggiata quando compare qui) o con --export-flagged-sample, e aggiungi il nome\n"
        "# immagine (una riga per immagine, righe vuote o che iniziano per # ignorate) a "
        f"{args.rescue_path}\n"
        f"# per tenere le sue detection sosia al prossimo run (rilancio istantaneo, riusa la cache).\n"
        f"# Campi: nome immagine, classe, confidence, containment, riga label (classe xc yc w h, come nei\n"
        f"# file label, per disegnare la bbox scartata).\n"
        f"# {len(all_flagged)} detection segnalate, ordinate per confidenza decrescente.\n\n"
        + "\n".join(
            f"{name}\t{class_names.get(cls, cls)}\tconf={conf:.2f}\tcontainment={containment:.2f}\t{label_line}"
            for name, cls, conf, containment, label_line in all_flagged
        ) + "\n"
    )
    print(f"Detection scartate per overlap segnalate per revisione in {flagged_path}")

    if args.sample:
        sample_dir = (args.sample_dir or Path(f"{out_dir}_sample")).resolve()
        if sample_dir.exists():
            shutil.rmtree(sample_dir)
        sample_dir.mkdir(parents=True)
        sample_names = random.sample(names_list, min(args.sample, len(names_list)))
        for name in sample_names:
            escooter_boxes, kept = per_image_kept[name]
            draw_sample(out_images / name, escooter_boxes, kept, class_names, sample_dir / name)
        print(f"Campione di verifica visiva ({len(sample_names)} immagini) in {sample_dir}")

    if args.export_flagged_sample:
        flagged_sample_dir = (args.flagged_sample_dir or Path(f"{out_dir}_flagged_sample")).resolve()
        if flagged_sample_dir.exists():
            shutil.rmtree(flagged_sample_dir)
        flagged_sample_dir.mkdir(parents=True)
        flagged_names = list(per_image_flagged)
        sample_names = random.sample(flagged_names, min(args.export_flagged_sample, len(flagged_names)))
        for name in sample_names:
            escooter_boxes, flagged = per_image_flagged[name]
            draw_flagged(out_images / name, escooter_boxes, flagged, class_names, flagged_sample_dir / name)
        print(f"Campione di revisione overlap sosia ({len(sample_names)} immagini, "
              f"su {len(flagged_names)} con almeno uno scarto) in {flagged_sample_dir}")


if __name__ == "__main__":
    main()
