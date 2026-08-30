#!/usr/bin/env python3
"""Produce un report dettagliato sulle immagini contenute in
data/image_index.json (l'indice prodotto da select_images.py con una voce
per ogni immagine del pool di partenza e l'esito della selezione).

Il report — senza toccare la pipeline né i file su disco — mostra:

  - totale immagini, selezionate vs escluse (con percentuali);

  - IMMAGINI SCARTATE: distribuzione per criterio (stadio + motivo), sia
    aggregata sia scomposta per dataset. I dettagli variabili nei motivi
    (dimensioni, path del duplicato, conteggi) sono normalizzati per
    raggruppare;

  - PER DATASET: quante immagini sono state scartate da ciascun criterio,
    con l'eventuale riga [override CSV: ...] se il dataset ha override dei
    parametri di selezione in datasets_to_download.csv;

  - CONTRIBUTO DI CIASCUN DATASET ALLA SELEZIONE FINALE: immagini
    selezionate e bounding box escooter portate da ogni dataset, con la
    quota sul totale e la media di bbox per immagine;

  - IMMAGINI SELEZIONATE: distribuzione delle dimensioni/risoluzioni,
    numero totale di annotazioni escooter, numero medio e massimo di
    annotazioni per immagine (globale e per dataset). I conteggi delle
    annotazioni sono letti dai file di label dei dataset deduplicati in
    data/interim/ (serve data/datasets.json e le cartelle *-dedup/); se
    non disponibili, la sezione annotazioni viene saltata con un avviso;

  - DIMENSIONE DELLE BBOX ESCOOTER: sulle sole immagini selezionate,
    distribuzione dell'area relativa (w*h in coordinate normalizzate) e
    della dimensione lineare (radice dell'area = frazione del lato immagine)
    di ogni bbox escooter, istogramma a bin log-spaziati fra le soglie
    FARAWAY/CLOSEUP, gruppi (cluster 1D k-means sull'area in scala log, per
    far emergere popolazioni distinte tipo "lontane" vs "vicine") con i
    dataset che li alimentano, e area/dimensione mediana per dataset.

    python3 scripts/report_image_index.py
    python3 scripts/report_image_index.py --no-annotations   # salta la lettura delle label (e la sezione bbox)
    python3 scripts/report_image_index.py --size-clusters 5  # forza il numero di gruppi di dimensione bbox (0 = auto)
    python3 scripts/report_image_index.py --check-files       # verifica anche l'esistenza su disco
    python3 scripts/report_image_index.py --json              # output JSON invece del testo

Esce 0 sempre, salvo che l'indice non sia leggibile.
"""
import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

try:
    import config
    DEFAULT_INDEX_PATH = config.IMAGE_INDEX_PATH
    REPO_ROOT = config.REPO_ROOT
    DATASETS_JSON = config.INDEX_PATH
    FARAWAY_AREA_THRESHOLD = config.FARAWAY_AREA_THRESHOLD
    CLOSEUP_AREA_THRESHOLD = config.CLOSEUP_AREA_THRESHOLD
except Exception:  # config richiede python-dotenv: fallback ai percorsi/valori convenzionali
    REPO_ROOT = Path(__file__).resolve().parent.parent
    DEFAULT_INDEX_PATH = REPO_ROOT / "data" / "image_index.json"
    DATASETS_JSON = REPO_ROOT / "data" / "datasets.json"
    FARAWAY_AREA_THRESHOLD = 0.001
    CLOSEUP_AREA_THRESHOLD = 0.4

try:
    import dataset_index
    SELECTION_OVERRIDES = dataset_index.selection_overrides()
except Exception:  # CSV non leggibile o dataset_index non importabile: nessun override mostrato
    SELECTION_OVERRIDES = {}

STAGE_ORDER = ["cheap", "temporal", "dedup", "variety"]

_NORMALISERS = [
    (re.compile(r"^immagine troppo piccola: \d+x\d+$"), "immagine troppo piccola (sotto MIN_PIXELS)"),
    (re.compile(r"^quasi-duplicato di .+$"), "quasi-duplicato di un'altra immagine"),
    (re.compile(r"^frame ridondante nella sequenza temporale .+$"),
     "frame ridondante in una sequenza temporale (dedup temporale)"),
    (re.compile(r"^\d+ istanze COCO rilevate, minimo richiesto \d+$"),
     "troppe poche istanze COCO (varietà insufficiente)"),
]


def normalise_reason(reason: str | None) -> str:
    if not reason:
        return "(nessun motivo)"
    for pattern, label in _NORMALISERS:
        if pattern.match(reason):
            return label
    return reason


def criterion(record: dict) -> str:
    """Etichetta compatta 'stadio / motivo' per un'immagine esclusa."""
    stage = record.get("exclusion_stage") or "(nessuno stadio)"
    return f"{stage} / {normalise_reason(record.get('exclusion_reason'))}"


def pct(part: int, whole: int) -> str:
    return f"{100.0 * part / whole:.1f}%" if whole else "-"


def _stage_key(label: str) -> tuple:
    stage = label.split(" / ", 1)[0]
    return (STAGE_ORDER.index(stage) if stage in STAGE_ORDER else 99, label)


# --------------------------------------------------------------------------- #
#  Conteggio annotazioni escooter (dai file di label dei dataset dedup)
# --------------------------------------------------------------------------- #
def _escooter_indices(names: list[str], wanted: set[str]) -> set[int]:
    return {i for i, n in enumerate(names) if n in wanted}


def _escooter_box_areas(label_path: Path, escooter_idx: set[int]) -> list[float] | None:
    """Aree relative (w*h in coordinate YOLO normalizzate) di ogni bbox
    escooter nel file di label, o None se il file non esiste."""
    if not label_path.exists():
        return None  # label mancante
    areas: list[float] = []
    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) == 5 and int(parts[0]) in escooter_idx:
            w, h = float(parts[3]), float(parts[4])
            areas.append(w * h)
    return areas


def load_box_area_reader(datasets_json: Path):
    """Ritorna (fn, error). fn(record) -> lista delle aree relative delle bbox
    escooter dell'immagine (lista vuota se nessuna), o None se il file di
    label non è stato trovato."""
    try:
        import yaml
    except ImportError:
        return None, "modulo 'yaml' non disponibile"
    if not datasets_json.exists():
        return None, f"{datasets_json} non trovato"

    entries = {e["id"]: e for e in json.loads(datasets_json.read_text())}
    yaml_cache: dict[str, set[int]] = {}

    def escooter_idx_for(dataset_id: str) -> set[int] | None:
        if dataset_id in yaml_cache:
            return yaml_cache[dataset_id]
        entry = entries.get(dataset_id)
        if not entry or not entry.get("dedup_dir"):
            yaml_cache[dataset_id] = None
            return None
        data_yaml = REPO_ROOT / entry["dedup_dir"] / "data.yaml"
        names = yaml.safe_load(data_yaml.read_text()).get("names", []) if data_yaml.exists() else []
        idx = _escooter_indices(names, set(entry.get("escooter_class_names") or []))
        yaml_cache[dataset_id] = idx
        return idx

    def read(record: dict) -> list[float] | None:
        entry = entries.get(record["dataset_id"])
        idx = escooter_idx_for(record["dataset_id"])
        if not entry or idx is None:
            return None
        stem = Path(record["image_path"]).stem
        label_path = REPO_ROOT / entry["dedup_dir"] / record["split"] / "labels" / f"{stem}.txt"
        return _escooter_box_areas(label_path, idx)

    return read, None


# --------------------------------------------------------------------------- #
#  Costruzione del report
# --------------------------------------------------------------------------- #
def _dim_stats(records: list[dict]) -> dict:
    widths = [r["width"] for r in records if r.get("width")]
    heights = [r["height"] for r in records if r.get("height")]
    resolutions = Counter(f"{r['width']}x{r['height']}" for r in records if r.get("width") and r.get("height"))
    out = {
        "distinct_resolutions": len(resolutions),
        "resolutions": resolutions.most_common(),
    }
    if widths:
        out["width"] = {"min": min(widths), "median": median(widths), "max": max(widths)}
        out["height"] = {"min": min(heights), "median": median(heights), "max": max(heights)}
    return out


def _quantile(sorted_xs: list[float], q: float) -> float:
    if not sorted_xs:
        return 0.0
    if len(sorted_xs) == 1:
        return sorted_xs[0]
    pos = q * (len(sorted_xs) - 1)
    lo = int(pos)
    frac = pos - lo
    hi = min(lo + 1, len(sorted_xs) - 1)
    return sorted_xs[lo] * (1 - frac) + sorted_xs[hi] * frac


def _kmeans_1d(values: list[float], k: int, iters: int = 100) -> list[list[float]]:
    """k-means 1D deterministico (init sui quantili). Ritorna i gruppi non
    vuoti, ordinati per centroide crescente."""
    xs = sorted(values)
    if k <= 1 or len(xs) <= k:
        return [xs] if xs else []
    centroids = [_quantile(xs, (i + 0.5) / k) for i in range(k)]
    groups: list[list[float]] = [[] for _ in range(k)]
    for _ in range(iters):
        groups = [[] for _ in range(k)]
        for x in xs:
            j = min(range(k), key=lambda c: abs(x - centroids[c]))
            groups[j].append(x)
        new = [sum(g) / len(g) if g else centroids[i] for i, g in enumerate(groups)]
        if new == centroids:
            break
        centroids = new
    return [g for g in groups if g]


def _wcss(groups: list[list[float]]) -> float:
    """Somma degli scarti quadratici entro i gruppi."""
    total = 0.0
    for g in groups:
        if not g:
            continue
        c = sum(g) / len(g)
        total += sum((x - c) ** 2 for x in g)
    return total


def _auto_k(values: list[float], k_max: int = 5) -> int:
    """Sceglie il numero di gruppi col metodo del gomito: il k in cui il
    guadagno marginale sulla WCSS decelera di più (massima differenza
    seconda). Ritorna 1 se i dati non bastano a formare gruppi."""
    distinct = len(set(values))
    if distinct < 3:
        return 1
    k_max = min(k_max, distinct)
    wcss = [0.0] + [_wcss(_kmeans_1d(values, k)) for k in range(1, k_max + 1)]
    if wcss[1] <= 0 or k_max < 3:
        return min(2, k_max)
    imp = {k: wcss[k - 1] - wcss[k] for k in range(2, k_max + 1)}
    best_k, best_decel = 2, -1.0
    for k in range(2, k_max):
        decel = imp[k] - imp.get(k + 1, 0.0)
        if decel > best_decel:
            best_decel, best_k = decel, k
    # se l'ultimo passo taglia ancora oltre il 15% della WCSS iniziale, tienilo
    if imp[k_max] / wcss[1] > 0.15:
        best_k = k_max
    return best_k


def _box_size_stats(boxes: list[tuple[float, str]], n_clusters: int) -> dict:
    """boxes: (area_relativa, dataset_id) per ogni bbox escooter delle immagini
    selezionate. Distribuzione, istogramma log-spaziato e gruppi k-means."""
    if not boxes:
        return {"n_boxes": 0}

    areas = sorted(a for a, _ in boxes)
    linear = [a ** 0.5 for a in areas]

    def band(xs: list[float]) -> dict:
        return {
            "min": xs[0], "p10": _quantile(xs, 0.10), "median": _quantile(xs, 0.50),
            "mean": mean(xs), "p90": _quantile(xs, 0.90), "max": xs[-1],
        }

    # --- istogramma a bin log-spaziati fra le soglie FARAWAY e CLOSEUP ---
    lo, hi = FARAWAY_AREA_THRESHOLD, CLOSEUP_AREA_THRESHOLD
    n_bins = 10
    edges = [lo * (hi / lo) ** (i / n_bins) for i in range(n_bins + 1)]
    edges[0], edges[-1] = 0.0, float("inf")
    hist = []
    for i in range(n_bins):
        a, b = edges[i], edges[i + 1]
        c = sum(1 for x in areas if a <= x < b)
        hist.append({
            "area_from": a, "area_to": b,
            "linear_from": a ** 0.5, "linear_to": (b ** 0.5 if math.isfinite(b) else float("inf")),
            "count": c,
        })

    # --- gruppi: k-means 1D sull'area in scala log ---
    # triple (log10 area, area, dataset) ordinate: i gruppi k-means 1D sono
    # partizioni contigue dei valori ordinati, quindi si affettano per lunghezza
    triples = sorted((math.log10(a), a, ds) for a, ds in boxes if a > 0)
    logs = [t[0] for t in triples]
    k = n_clusters if n_clusters > 0 else _auto_k(logs)
    clusters = []
    pos = 0
    for g in _kmeans_1d(logs, k):
        seg = triples[pos:pos + len(g)]
        pos += len(g)
        if not seg:
            continue
        ds_counts = Counter(ds for _, _, ds in seg)
        center = 10 ** (sum(t[0] for t in seg) / len(seg))
        clusters.append({
            "n": len(seg),
            "area_min": seg[0][1], "area_max": seg[-1][1], "area_center": center,
            "linear_min": seg[0][1] ** 0.5, "linear_max": seg[-1][1] ** 0.5, "linear_center": center ** 0.5,
            "top_datasets": ds_counts.most_common(5),
        })

    # --- per dataset ---
    by_ds: dict[str, list[float]] = defaultdict(list)
    for a, ds in boxes:
        by_ds[ds].append(a)
    per_dataset = {}
    for ds, lst in by_ds.items():
        lst.sort()
        med = _quantile(lst, 0.50)
        per_dataset[ds] = {
            "n_boxes": len(lst),
            "area_median": med, "linear_median": med ** 0.5,
            "area_p10": _quantile(lst, 0.10), "area_p90": _quantile(lst, 0.90),
        }

    return {
        "n_boxes": len(boxes),
        "k_clusters": k,
        "area": band(areas),
        "linear": band(linear),
        "histogram": hist,
        "clusters": clusters,
        "per_dataset": {ds: per_dataset[ds] for ds in sorted(per_dataset)},
    }


def build_report(records: list[dict], box_area_reader, check_files: bool, size_clusters: int = 0) -> dict:
    total = len(records)
    excluded = [r for r in records if r.get("excluded")]
    selected = [r for r in records if not r.get("excluded")]

    # --- distribuzione esclusioni (aggregata) ---
    crit_counter = Counter(criterion(r) for r in excluded)
    stage_counter = Counter((r.get("exclusion_stage") or "(nessuno stadio)") for r in excluded)

    # --- per dataset ---
    per_dataset: dict[str, dict] = {}
    for r in records:
        d = per_dataset.setdefault(r["dataset_id"], {
            "total": 0, "selected": 0, "excluded": 0, "exclusions": Counter(),
        })
        d["total"] += 1
        if r.get("excluded"):
            d["excluded"] += 1
            d["exclusions"][criterion(r)] += 1
        else:
            d["selected"] += 1

    per_split = Counter(r["split"] for r in records)
    per_split_selected = Counter(r["split"] for r in selected)

    report = {
        "index_total": total,
        "selected": len(selected),
        "excluded": len(excluded),
        "excluded_distribution": {
            "by_stage": {s: stage_counter[s] for s in
                         sorted(stage_counter, key=lambda s: STAGE_ORDER.index(s) if s in STAGE_ORDER else 99)},
            "by_criterion": {c: crit_counter[c] for c in sorted(crit_counter, key=_stage_key)},
        },
        "per_dataset": {
            k: {
                "total": v["total"], "selected": v["selected"], "excluded": v["excluded"],
                "exclusions": {c: v["exclusions"][c] for c in sorted(v["exclusions"], key=_stage_key)},
                "selection_overrides": SELECTION_OVERRIDES.get(k, {}),
            }
            for k, v in sorted(per_dataset.items())
        },
        "per_split": {s: {"total": per_split[s], "selected": per_split_selected[s]} for s in sorted(per_split)},
        "selected_stats": {"dimensions": _dim_stats(selected)},
    }

    # --- contributo di ciascun dataset alla selezione finale ---
    contribution = {
        ds: {"selected_images": v["selected"], "images_with_labels": 0,
             "escooter_boxes": 0, "mean_boxes_per_image": 0}
        for ds, v in sorted(per_dataset.items()) if v["selected"]
    }
    report["selected_stats"]["contribution_by_dataset"] = contribution

    # --- annotazioni escooter sulle immagini selezionate ---
    if box_area_reader is not None:
        counts_by_dataset: dict[str, list[int]] = defaultdict(list)
        boxes: list[tuple[float, str]] = []  # (area_relativa, dataset_id) per ogni bbox escooter
        missing_labels = 0
        for r in selected:
            areas = box_area_reader(r)
            if areas is None:
                missing_labels += 1
                continue
            counts_by_dataset[r["dataset_id"]].append(len(areas))
            boxes.extend((a, r["dataset_id"]) for a in areas)

        for ds, lst in counts_by_dataset.items():
            if ds in contribution:
                contribution[ds]["images_with_labels"] = len(lst)
                contribution[ds]["escooter_boxes"] = sum(lst)
                contribution[ds]["mean_boxes_per_image"] = round(mean(lst), 3) if lst else 0

        all_counts = [n for lst in counts_by_dataset.values() for n in lst]
        ann = {
            "images_with_labels": len(all_counts),
            "missing_labels": missing_labels,
            "total_annotations": sum(all_counts),
            "mean_per_image": round(mean(all_counts), 3) if all_counts else 0,
            "max_per_image": max(all_counts) if all_counts else 0,
            "per_image_histogram": dict(sorted(Counter(all_counts).items())),
            "per_dataset": {
                ds: {
                    "images": len(lst),
                    "total_annotations": sum(lst),
                    "mean_per_image": round(mean(lst), 3) if lst else 0,
                    "max_per_image": max(lst) if lst else 0,
                }
                for ds, lst in sorted(counts_by_dataset.items())
            },
        }
        report["selected_stats"]["annotations"] = ann
        report["selected_stats"]["box_sizes"] = _box_size_stats(boxes, size_clusters)

    if check_files:
        report["missing_on_disk"] = [r["image_path"] for r in records if not Path(r["image_path"]).is_file()]

    return report


# --------------------------------------------------------------------------- #
#  Output testuale
# --------------------------------------------------------------------------- #
def print_text(report: dict, index_path: Path, annotation_note: str | None) -> None:
    total = report["index_total"]
    excl = report["excluded"]
    sel = report["selected"]

    print(f"Indice: {index_path}")
    print(f"Immagini nell'indice: {total}\n")
    print(f"  selezionate : {sel:>7}  ({pct(sel, total)})")
    print(f"  escluse     : {excl:>7}  ({pct(excl, total)})\n")

    print("=" * 72)
    print("IMMAGINI SCARTATE — distribuzione per criterio")
    print("=" * 72)
    dist = report["excluded_distribution"]
    print("\n  per stadio:")
    for stage, n in dist["by_stage"].items():
        print(f"      {n:>7}  ({pct(n, excl):>6})  {stage}")
    print("\n  per criterio (stadio / motivo):")
    for crit, n in dist["by_criterion"].items():
        print(f"      {n:>7}  ({pct(n, excl):>6})  {crit}")
    print()

    print("=" * 72)
    print("PER DATASET — scartate per ciascun criterio")
    print("=" * 72)
    for ds, v in sorted(report["per_dataset"].items(), key=lambda kv: -kv[1]["total"]):
        print(f"\n  {ds}   (tot {v['total']}, selezionate {v['selected']}, "
              f"escluse {v['excluded']} — {pct(v['excluded'], v['total'])})")
        if v.get("selection_overrides"):
            ov = ", ".join(f"{k}={val}" for k, val in v["selection_overrides"].items())
            print(f"      [override CSV: {ov}]")
        if not v["exclusions"]:
            print("      (nessuna esclusione)")
        for crit, n in v["exclusions"].items():
            print(f"      {n:>7}  ({pct(n, v['excluded']):>6})  {crit}")
    print()

    print("=" * 72)
    print("PER SPLIT")
    print("=" * 72)
    for split, v in report["per_split"].items():
        print(f"  {split:<8} tot {v['total']:>7}   selezionate {v['selected']:>7}  ({pct(v['selected'], v['total'])})")
    print()

    contrib = report["selected_stats"].get("contribution_by_dataset", {})
    if contrib:
        print("=" * 72)
        print("CONTRIBUTO DI CIASCUN DATASET ALLA SELEZIONE FINALE")
        print("=" * 72)
        tot_img = sum(c["selected_images"] for c in contrib.values())
        tot_box = sum(c["escooter_boxes"] for c in contrib.values())
        has_boxes = any(c["images_with_labels"] for c in contrib.values())
        header = f"\n      {'dataset_id':<34} {'img':>6} {'quota':>7}"
        print(header + (f"  {'bbox':>7} {'quota':>7} {'bbox/img':>9}" if has_boxes else ""))
        for ds, c in sorted(contrib.items(), key=lambda kv: -kv[1]["selected_images"]):
            missing = c["selected_images"] - c["images_with_labels"]
            note = f"  ({missing} senza label)" if has_boxes and missing else ""
            if has_boxes:
                print(f"      {ds:<34} {c['selected_images']:>6} {pct(c['selected_images'], tot_img):>7}  "
                      f"{c['escooter_boxes']:>7} {pct(c['escooter_boxes'], tot_box):>7} "
                      f"{c['mean_boxes_per_image']:>9}{note}")
            else:
                print(f"      {ds:<34} {c['selected_images']:>6} {pct(c['selected_images'], tot_img):>7}")
        if has_boxes:
            mean_all = round(tot_box / tot_img, 3) if tot_img else 0
            print(f"      {'TOTALE':<34} {tot_img:>6} {'100.0%':>7}  {tot_box:>7} {'100.0%':>7} {mean_all:>9}")
        else:
            print(f"      {'TOTALE':<34} {tot_img:>6} {'100.0%':>7}")
            print("      (bbox non calcolate: --no-annotations o label non trovate)")
        print()

    print("=" * 72)
    print("IMMAGINI SELEZIONATE")
    print("=" * 72)
    dim = report["selected_stats"]["dimensions"]
    print(f"\n  dimensioni ({sel} immagini):")
    if dim.get("width"):
        w, h = dim["width"], dim["height"]
        print(f"      larghezza: min {w['min']}  mediana {w['median']:.0f}  max {w['max']}")
        print(f"      altezza  : min {h['min']}  mediana {h['median']:.0f}  max {h['max']}")
    print(f"      risoluzioni distinte: {dim['distinct_resolutions']}")
    for res, n in dim["resolutions"]:
        print(f"          {n:>7}  ({pct(n, sel):>6})  {res}")

    ann = report["selected_stats"].get("annotations")
    if ann is None:
        print(f"\n  annotazioni escooter: non calcolate ({annotation_note})")
    else:
        print("\n  annotazioni escooter (dai file di label dei dataset dedup):")
        print(f"      immagini con label trovate : {ann['images_with_labels']}")
        if ann["missing_labels"]:
            print(f"      immagini senza label trovate: {ann['missing_labels']}  (escluse dai conteggi)")
        print(f"      annotazioni totali          : {ann['total_annotations']}")
        print(f"      media per immagine          : {ann['mean_per_image']}")
        print(f"      massimo per immagine        : {ann['max_per_image']}")
        print("      distribuzione (n. annotazioni -> n. immagini):")
        for k, v in ann["per_image_histogram"].items():
            print(f"          {k:>3} ann.  {v:>7} img")
        print("\n      per dataset:")
        print(f"      {'dataset_id':<34} {'img':>6} {'ann.tot':>9} {'media':>7} {'max':>5}")
        for ds, s in sorted(ann["per_dataset"].items(), key=lambda kv: -kv[1]["total_annotations"]):
            print(f"      {ds:<34} {s['images']:>6} {s['total_annotations']:>9} "
                  f"{s['mean_per_image']:>7} {s['max_per_image']:>5}")
    print()

    bs = report["selected_stats"].get("box_sizes")
    if bs is not None and bs.get("n_boxes"):
        print("=" * 72)
        print("DIMENSIONE DELLE BBOX ESCOOTER (immagini selezionate)")
        print("=" * 72)

        def _a(x):  # area relativa -> percentuale
            return "inf" if x == float("inf") else f"{x * 100:.3f}%"

        def _l(x):  # dimensione lineare -> percentuale del lato immagine
            return "inf" if x == float("inf") else f"{x * 100:.1f}%"

        a, lin = bs["area"], bs["linear"]
        print(f"\n  n. bbox escooter: {bs['n_boxes']}")
        print(f"\n  area relativa (w*h, % dell'area immagine):")
        print(f"      min {_a(a['min'])}   p10 {_a(a['p10'])}   mediana {_a(a['median'])}   "
              f"media {_a(a['mean'])}   p90 {_a(a['p90'])}   max {_a(a['max'])}")
        print(f"  dimensione lineare (radice dell'area, % del lato immagine):")
        print(f"      min {_l(lin['min'])}   p10 {_l(lin['p10'])}   mediana {_l(lin['median'])}   "
              f"media {_l(lin['mean'])}   p90 {_l(lin['p90'])}   max {_l(lin['max'])}")

        hist = bs["histogram"]
        hmax = max((h["count"] for h in hist), default=0)
        print("\n  istogramma (area relativa -> n. bbox):")
        for h in hist:
            bar = "#" * round(40 * h["count"] / hmax) if hmax else ""
            print(f"      [{_a(h['area_from']):>8}, {_a(h['area_to']):>8})  "
                  f"(lato {_l(h['linear_from']):>6}..{_l(h['linear_to']):>6})  "
                  f"{h['count']:>7}  ({pct(h['count'], bs['n_boxes']):>6})  {bar}")

        print(f"\n  gruppi (k-means 1D su log-area, k={bs['k_clusters']}):")
        for i, c in enumerate(bs["clusters"], 1):
            ds = ", ".join(f"{name}({n})" for name, n in c["top_datasets"])
            print(f"      #{i}  n={c['n']:>6} ({pct(c['n'], bs['n_boxes']):>6})  "
                  f"area {_a(c['area_min'])}..{_a(c['area_max'])}  "
                  f"(lato {_l(c['linear_min'])}..{_l(c['linear_max'])}), centro {_a(c['area_center'])}")
            print(f"           dataset: {ds}")

        print("\n  per dataset (area mediana crescente):")
        print(f"      {'dataset_id':<34} {'bbox':>7} {'area med':>10} {'lato med':>9} {'area p10..p90':>20}")
        for ds, s in sorted(bs["per_dataset"].items(), key=lambda kv: kv[1]["area_median"]):
            print(f"      {ds:<34} {s['n_boxes']:>7} {_a(s['area_median']):>10} {_l(s['linear_median']):>9} "
                  f"{_a(s['area_p10']) + '..' + _a(s['area_p90']):>20}")
        print()
    elif bs is not None:
        print("DIMENSIONE DELLE BBOX ESCOOTER: nessuna bbox con label trovata.\n")

    if "missing_on_disk" in report:
        missing = report["missing_on_disk"]
        if missing:
            print(f"FILE MANCANTI SU DISCO ({len(missing)}):")
            for p in missing[:50]:
                print(f"      {p}")
            if len(missing) > 50:
                print(f"      ... e altri {len(missing) - 50}")
        else:
            print("FILE SU DISCO: tutti presenti.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX_PATH,
                        help=f"Percorso dell'indice immagini (default: {DEFAULT_INDEX_PATH})")
    parser.add_argument("--datasets-json", type=Path, default=DATASETS_JSON,
                        help=f"Indice dataset per risalire ai file di label (default: {DATASETS_JSON})")
    parser.add_argument("--no-annotations", action="store_true",
                        help="Non leggere i file di label: salta le statistiche su annotazioni e dimensione bbox")
    parser.add_argument("--size-clusters", type=int, default=0,
                        help="Numero di gruppi per la dimensione delle bbox escooter (0 = scelto automaticamente)")
    parser.add_argument("--check-files", action="store_true",
                        help="Verifica anche che ogni image_path esista su disco (più lento)")
    parser.add_argument("--json", action="store_true", help="Stampa il report in JSON")
    args = parser.parse_args()

    try:
        records = json.loads(args.index.read_text())
    except FileNotFoundError:
        print(f"Indice non trovato: {args.index}", file=sys.stderr)
        return 2
    except json.JSONDecodeError as e:
        print(f"Indice non valido ({args.index}): {e}", file=sys.stderr)
        return 2
    if not isinstance(records, list):
        print(f"Formato inatteso: l'indice non è una lista ({args.index})", file=sys.stderr)
        return 2

    box_area_reader, annotation_note = (None, "disabilitato con --no-annotations")
    if not args.no_annotations:
        box_area_reader, annotation_note = load_box_area_reader(args.datasets_json)

    report = build_report(records, box_area_reader, check_files=args.check_files,
                          size_clusters=args.size_clusters)

    if args.json:
        if box_area_reader is None and not args.no_annotations:
            report["selected_stats"]["annotations_note"] = annotation_note
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print_text(report, args.index, annotation_note)

    return 0


if __name__ == "__main__":
    sys.exit(main())
