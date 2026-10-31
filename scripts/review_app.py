#!/usr/bin/env python3
"""Applicazione locale per la revisione manuale di un dataset in formato
YOLO, organizzato in due sottocartelle images/ e labels/ (un file .txt per
immagine, righe "classe xc yc w h" normalizzate 0..1).

Mostra un'immagine alla volta con le bbox disegnate (canvas HTML) — colorate
per classe con etichetta testuale (nomi COCO standard 0-79, l'escooter
sempre in rosso), utile anche su dataset multi-classe come
union_reviewed_coco e non solo sul dataset di unione a classe singola — e
registra la decisione dell'utente con un tasto:
    s = seleziona
    n = scarta
    freccia sinistra/destra = torna indietro / salta avanti senza decidere
    end = salta alla prima immagine non revisionata nell'elenco corrente (rispetta i filtri)
    campo "vai a immagine" (in alto): salta alla N-esima immagine dell'elenco corrente
    backspace = cancella la decisione sull'immagine corrente
    b = alterna fra modalità singola e Browse (v. sotto)

Modalità Browse (bottone "Singola"/"Browse" in alto): una griglia di 12
miniature per pagina (4 x 3) con le bbox disegnate, per scorrere il dataset
a colpo d'occhio invece di un'immagine alla volta. Lavora sullo stesso
elenco filtrato/ordinato della modalità singola; il bordo della miniatura
indica la decisione (verde = selezionata, rosso = scartata), frecce
sinistra/destra cambiano pagina, un clic su una miniatura la apre in
modalità singola, dove si decide e si modificano le bbox.

Il campo "cerca" (barra dei filtri) tiene solo le immagini il cui nome file
contiene il testo digitato, senza distinguere maiuscole/minuscole; si
combina con gli altri filtri e vale in entrambe le modalità.

Se l'immagine corrente compare in --flagged-path (default il
flagged_coco_overlap.txt di annotate_coco_classes.py: detection di classe
sosia del monopattino scartate per overlap, v. PIPELINE.md sezione 9), lo
segnala con una fascia arancione sopra l'immagine, con classe/confidence/
containment di ogni detection scartata — utile per decidere quali
recuperare in coco_overlap_rescue.txt senza dover incrociare a mano il file
di testo. Nessun effetto se il file non esiste (es. su dataset diversi da
union_reviewed_coco).

Sulla sinistra, un pannello fisso mostra le prime 40 classi del dataset
(id, nome, numero di immagini che le contengono, numero di istanze
totali), ordinate per istanze decrescenti: si aggiorna dopo ogni modifica
alle bbox, utile per tenere d'occhio la composizione del dataset mentre lo
si revisiona.

Filtri e ordinamento (barra sotto la testata): si può filtrare l'elenco
per dataset di origine (dedotto dal prefisso "<dataset_id>__" nel nome
file), per classe (solo le immagini con almeno una bbox di quella classe
nel label ATTUALE, cioè già rivisto — non nella pseudo-label di partenza:
si aggiorna a ogni modifica salvata) e per stato (selezionata / scartata /
non revisionata), e ordinarlo per nome file, dataset, stato o pHash, con
un criterio primario e uno secondario (per pareggiare i casi di parità del
primo).

Le bbox sono modificabili direttamente sul canvas:
    trascina un'area vuota per crearne una nuova, con la classe del menu
        "classe" (sotto il canvas, escooter o una qualsiasi classe COCO):
        utile per colmare a mano i falsi negativi di una pseudo-annotazione
    clic su una bbox per selezionarla, trascina per spostarla; il menu
        "classe" mostra la sua classe e la si può riassegnare da lì
    trascina i quadratini della bbox selezionata per ridimensionarla
    x oppure canc = elimina la bbox selezionata
    r = ripristina le bbox originali (prima di qualsiasi modifica)
    h = mostra/nascondi le bbox (utile con istanze ammucchiate)
    e = mostra/nascondi il pallino rosso a metà del bordo superiore delle
        bbox modificate (nuove, spostate, ridimensionate o con classe
        cambiata rispetto al backup in review_label_backups); vale anche
        nelle miniature di Browse. Senza backup nessuna bbox è marcata.
    + = zoom in (ogni pressione ingrandisce ulteriormente, fino a 5x); con
        l'immagine ingrandita si scorre con lo scroll del browser e si
        continua ad annotare come a zoom 1x (utile per gli oggetti piccoli
        o le bbox da riallargare sulla parte occlusa)
    - = torna alla dimensione originale (non zoom out graduale)
    esc = deseleziona
Ogni modifica viene salvata subito nel file label corrispondente. La
prima volta che un'immagine viene modificata, il suo file label originale
viene copiato nella cartella di backup (--backup-dir) come rete di
sicurezza: il tasto r ripristina le bbox da quella copia.

Le decisioni sono scritte subito nel file delle decisioni (--decisions-file)
ad ogni tasto premuto: la sessione si può interrompere e riprendere quando
si vuole, riparte dalla prima immagine ancora senza decisione.

Avvio:
    python3 scripts/review_app.py <dataset_dir> [--port 8765]
        [--decisions-file FILE] [--backup-dir DIR] [--flagged-path FILE]

<dataset_dir> deve contenere le sottocartelle images/ e labels/. Se non
specificati, --decisions-file e --backup-dir vengono creati dentro
<dataset_dir> (rispettivamente review_decisions.json e
review_label_backups/).

poi apri http://localhost:<port> nel browser.
"""
import argparse
import json
import mimetypes
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

import imagehash
from PIL import Image

import config


def load_decisions(decisions_path: Path) -> dict:
    if decisions_path.exists():
        return json.loads(decisions_path.read_text())
    return {}


def save_decisions(decisions_path: Path, decisions: dict) -> None:
    decisions_path.write_text(json.dumps(decisions, indent=2, ensure_ascii=False) + "\n")


def backup_label_once(backup_dir: Path, name: str, lbl_path: Path) -> None:
    """Copia il file label originale in backup_dir, solo la prima volta
    che viene modificato in questa (o in una precedente) sessione."""
    backup_path = backup_dir / f"{Path(name).stem}.txt"
    if backup_path.exists():
        return
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path.write_text(lbl_path.read_text() if lbl_path.exists() else "")


def read_boxes_file(lbl_path: Path) -> list:
    boxes = []
    if lbl_path.exists():
        for line in lbl_path.read_text().splitlines():
            parts = line.split()
            if len(parts) == 5:
                cls = int(float(parts[0]))
                boxes.append([cls] + [float(v) for v in parts[1:]])
    return boxes


def boxes_modified(lbl_path: Path, backup_path: Path) -> bool:
    """True se esiste un backup (cioè l'immagine è stata modificata almeno una
    volta) e le bbox attuali differiscono ancora da quelle originali (es. non
    è stato fatto un ripristino nel frattempo)."""
    if not backup_path.exists():
        return False
    return read_boxes_file(lbl_path) != read_boxes_file(backup_path)


def clip_box(xc: float, yc: float, w: float, h: float):
    """Ritaglia una bbox YOLO normalizzata (xc, yc, w, h) ai bordi
    dell'immagine [0, 1] e ne ricalcola centro e dimensioni; None se dopo il
    ritaglio non resta area (bbox interamente fuori). Una bbox che sporge dal
    bordo equivarrebbe a una bbox amodale, contraria alla convenzione COCO."""
    x1, x2 = max(0.0, xc - w / 2), min(1.0, xc + w / 2)
    y1, y2 = max(0.0, yc - h / 2), min(1.0, yc + h / 2)
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1


def original_boxes(backup_path: Path):
    """Le bbox originali (contenuto del backup) o None se l'immagine non è mai
    stata modificata: serve all'interfaccia per marcare le singole bbox che
    non coincidono con nessuna originale (nuove o ritoccate)."""
    return read_boxes_file(backup_path) if backup_path.exists() else None


def load_flagged(path: Path) -> dict:
    """Nome immagine -> lista di detection scartate per overlap sosia
    (formato scritto da annotate_coco_classes.py in FLAGGED_COCO_OVERLAP_
    PATH: righe "nome\\tclasse\\tconf=X.XX\\tcontainment=X.XX\\tcls xc yc w
    h", righe vuote o che iniziano per # ignorate), ognuna con
    {"label": descrizione testuale, "box": [cls, xc, yc, w, h]} — il box
    serve a disegnarla sul canvas di revisione, tratteggiata, insieme a
    quelle rimaste nel dataset. Dizionario vuoto se il file non esiste
    (dataset diverso da union_reviewed_coco, o mai generato): la
    segnalazione a video resta semplicemente disattivata."""
    if not path.exists():
        return {}
    flagged: dict = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) != 5:
            continue
        name, cls, conf, containment, label_line = parts
        box_parts = label_line.split()
        if len(box_parts) != 5:
            continue
        box = [int(float(box_parts[0]))] + [float(v) for v in box_parts[1:]]
        flagged.setdefault(name, []).append({"label": f"{cls} {conf} {containment}", "box": box})
    return flagged


def class_stats(labels_dir: Path) -> tuple:
    """Un solo passaggio sui label correnti (quelli già salvati, non le
    pseudo-label originali) per due cose:
    - la lista ordinata (istanze decrescenti) di {classe, n_immagini,
      n_istanze}, per il pannello "top classi" della UI
    - {nome immagine: [classi presenti]}, per il filtro per classe — tiene
      conto di qualunque modifica già salvata (box aggiunte/tolte/
      riclassificate), non dell'annotazione di partenza."""
    instances: Counter = Counter()
    images: Counter = Counter()
    by_image: dict = {}
    for lbl in labels_dir.glob("*.txt"):
        seen = set()
        for cls, *_ in read_boxes_file(lbl):
            instances[cls] += 1
            seen.add(cls)
        if seen:
            by_image[lbl.stem] = sorted(seen)
        for cls in seen:
            images[cls] += 1
    stats = sorted(
        ({"cls": cls, "n_images": images[cls], "n_instances": n} for cls, n in instances.items()),
        key=lambda d: -d["n_instances"],
    )
    return stats, by_image


def compute_phashes(dataset_dir: Path, cache_path: Path) -> dict:
    """pHash di ogni immagine del dataset, con cache su disco (invalidata per
    mtime/size, come la cache pHash della pipeline principale in
    select_images.py)."""
    cache = load_decisions(cache_path)  # stesso formato semplice {chiave: valore}, riusabile
    names = sorted(p.name for p in (dataset_dir / "images").iterdir())
    changed = False
    for name in names:
        path = dataset_dir / "images" / name
        stat = path.stat()
        entry = cache.get(name)
        if entry and entry.get("mtime") == stat.st_mtime and entry.get("size") == stat.st_size:
            continue
        with Image.open(path) as im:
            phash_hex = str(imagehash.phash(im))
        cache[name] = {"mtime": stat.st_mtime, "size": stat.st_size, "phash": phash_hex}
        changed = True
    if changed:
        save_decisions(cache_path, cache)
    return {name: cache[name]["phash"] for name in names if name in cache}


PAGE = """<!doctype html>
<html lang="it">
<head>
<meta charset="utf-8">
<title>Revisione dataset union</title>
<style>
  :root { color-scheme: dark; }
  body { margin: 0; background: #111; color: #eee; font-family: system-ui, sans-serif; }
  #bar { display: flex; justify-content: space-between; align-items: center; padding: 10px 16px; background: #1b1b1b; border-bottom: 1px solid #333; flex-wrap: wrap; gap: 8px; }
  #bar .counts span { margin-right: 14px; }
  #filterbar { display: flex; align-items: center; gap: 16px; padding: 8px 16px; background: #161616; border-bottom: 1px solid #333; flex-wrap: wrap; font-size: 13px; }
  #filterbar label { color: #aaa; display: flex; align-items: center; gap: 4px; }
  #filterbar select { background: #222; color: #eee; border: 1px solid #444; border-radius: 4px; padding: 3px 6px; }
  #goto-input { width: 55px; background: #222; color: #eee; border: 1px solid #444; border-radius: 4px; padding: 3px 6px; margin-left: 10px; }
  #goto-btn { background: #333; color: #eee; border: 1px solid #555; border-radius: 4px; padding: 3px 10px; margin-left: 4px; cursor: pointer; }
  #goto-btn:hover { background: #444; }
  #phash-status { color: #888; font-size: 12px; }
  #layout { display: flex; align-items: flex-start; }
  #sidebar { width: 230px; flex: none; box-sizing: border-box; padding: 12px; background: #161616;
    border-right: 1px solid #333; font-size: 12px; height: 100vh; overflow-y: auto; }
  #sidebar h3 { margin: 0 0 10px; font-size: 13px; color: #ccc; font-weight: normal; }
  #sidebar table { width: 100%; border-collapse: collapse; }
  #sidebar th { text-align: left; color: #888; font-weight: normal; padding: 2px 4px; border-bottom: 1px solid #333; }
  #sidebar td { padding: 3px 4px; border-bottom: 1px solid #262626; }
  #sidebar td.num { text-align: right; color: #bbb; }
  #main { flex: 1; min-width: 0; }
  #stage { position: relative; display: flex; align-items: center; justify-content: center; height: calc(100vh - 178px); }
  #stage.zoomed { display: block; overflow: auto; }
  #stage img { max-width: 96vw; max-height: 100%; display: block; }
  #stage.zoomed img { max-width: none; max-height: none; }
  #stage canvas { position: absolute; pointer-events: auto; cursor: crosshair; }
  #zoom-status { color: #888; }
  #name { text-align: center; padding: 6px; font-size: 13px; color: #999; word-break: break-all; }
  #res-info { font-weight: bold; font-size: 17px; color: #eee; }
  #decision { font-weight: bold; }
  .sel { color: #4caf50; }
  .disc { color: #e53935; }
  #help, #help2 { text-align: center; padding: 6px; font-size: 13px; color: #888; }
  #help kbd, #help2 kbd { background: #333; border-radius: 3px; padding: 1px 6px; margin: 0 2px; }
  #box-status { font-weight: bold; color: #4caf50; }
  #modified-badge { color: #ffb300; font-weight: bold; margin-left: 8px; }
  #restore-btn { background: #333; color: #eee; border: 1px solid #555; border-radius: 4px; padding: 2px 10px; margin-left: 8px; cursor: pointer; font-size: 12px; }
  #restore-btn:hover { background: #444; }
  #done { display: none; text-align: center; margin-top: 80px; font-size: 22px; }
  #empty { display: none; text-align: center; margin-top: 80px; font-size: 18px; color: #888; }
  #flagged-banner { display: none; padding: 8px 16px; background: #4a2f00; color: #ffcc66; border-bottom: 1px solid #a66a00; font-size: 13px; font-weight: bold; }
  #f-search { background: #222; color: #eee; border: 1px solid #444; border-radius: 4px; padding: 3px 6px; width: 200px; }
  #modes { display: flex; gap: 4px; }
  #modes button { background: #222; color: #ccc; border: 1px solid #444; border-radius: 4px; padding: 4px 12px; cursor: pointer; }
  #modes button.active { background: #2d5a2d; color: #fff; border-color: #3a7a3a; }
  #browse { padding: 12px 16px; }
  #browse-bar { display: flex; align-items: center; gap: 10px; margin-bottom: 12px; color: #999; font-size: 13px; }
  #browse-bar button { background: #333; color: #eee; border: 1px solid #555; border-radius: 4px; padding: 3px 12px; cursor: pointer; }
  #browse-bar button:hover { background: #444; }
  #browse-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; }
  .card { background: #1b1b1b; border: 2px solid #333; border-radius: 6px; padding: 4px; cursor: pointer; }
  .card:hover { background: #262626; }
  .card.select { border-color: #4caf50; }
  .card.discard { border-color: #e53935; }
  .card.current { box-shadow: 0 0 0 2px #ffd60a; }
  .card .imgwrap { position: relative; width: 100%; aspect-ratio: 4/3; background: #000; border-radius: 3px; overflow: hidden; }
  .card .imgwrap img, .card .imgwrap canvas { position: absolute; left: 0; top: 0; width: 100%; height: 100%; }
  .card .imgwrap img { object-fit: contain; }
  .card .imgwrap canvas { pointer-events: none; }
  .card .meta { font-size: 11px; color: #999; margin-top: 4px; word-break: break-all; }
</style>
</head>
<body>
<div id="layout">
<aside id="sidebar">
  <h3>Top 40 classi del dataset</h3>
  <table>
    <thead><tr><th>id</th><th>classe</th><th>img</th><th>ist.</th></tr></thead>
    <tbody id="class-stats-body"></tbody>
  </table>
</aside>
<div id="main">
<div id="bar">
  <div>Immagine <span id="idx">-</span>/<span id="total">-</span>
    <input type="number" id="goto-input" min="1" placeholder="#" title="Vai all'immagine numero...">
    <button id="goto-btn">vai</button>
  </div>
  <div id="modes">
    <button id="mode-single" class="active" title="Un'immagine alla volta, con modifica delle bbox">Singola</button>
    <button id="mode-browse" title="Scorri le immagini a blocchi di 12 miniature (tasto b)">Browse</button>
  </div>
  <div class="counts">
    <span class="sel">selezionate: <b id="c-select">0</b></span>
    <span class="disc">scartate: <b id="c-discard">0</b></span>
    <span>rimanenti: <b id="c-todo">0</b></span>
  </div>
</div>
<div id="filterbar">
  <label>cerca:
    <input type="search" id="f-search" placeholder="parte del nome file" title="Mostra solo le immagini il cui nome file contiene questo testo (senza distinguere maiuscole)">
  </label>
  <label>dataset:
    <select id="f-dataset"><option value="">tutti</option></select>
  </label>
  <label>classe:
    <select id="f-class" title="Mostra solo le immagini con almeno una bbox di questa classe, nel label ATTUALE (già rivisto), non nella pseudo-label originale">
      <option value="">tutte</option>
    </select>
  </label>
  <label>stato:
    <select id="f-status">
      <option value="">tutti</option>
      <option value="select">selezionate</option>
      <option value="discard">scartate</option>
      <option value="todo">non revisionate</option>
    </select>
  </label>
  <label>ordina per:
    <select id="sort1">
      <option value="">-</option>
      <option value="filename">nome file</option>
      <option value="dataset">dataset</option>
      <option value="status">stato</option>
      <option value="phash">pHash</option>
    </select>
  </label>
  <label>poi per:
    <select id="sort2">
      <option value="">-</option>
      <option value="filename">nome file</option>
      <option value="dataset">dataset</option>
      <option value="status">stato</option>
      <option value="phash">pHash</option>
    </select>
  </label>
  <span id="phash-status"></span>
</div>
<section id="browse" hidden>
  <div id="browse-bar">
    <button id="browse-prev">&larr; precedenti</button>
    <span id="browse-info"></span>
    <button id="browse-next">successive &rarr;</button>
    <span>clic su una miniatura per aprirla in modalità singola &nbsp; <kbd>&larr;</kbd>/<kbd>&rarr;</kbd> pagina &nbsp; <kbd>b</kbd> torna a singola</span>
  </div>
  <div id="browse-grid"></div>
  <div id="browse-empty" hidden style="text-align:center;margin-top:60px;font-size:18px;color:#888">Nessuna immagine corrisponde ai filtri selezionati.</div>
</section>
<div id="single">
<div id="flagged-banner"></div>
<div id="stage">
  <img id="img">
  <canvas id="canvas"></canvas>
</div>
<div id="name"><span id="fname"></span> &nbsp;&mdash;&nbsp; <span id="res-info"></span><span id="modified-badge" hidden>&#9998; bbox modificate</span></div>
<div id="help">
  <kbd>s</kbd> seleziona &nbsp; <kbd>n</kbd> scarta &nbsp;
  <kbd>&larr;</kbd>/<kbd>&rarr;</kbd> naviga &nbsp; <kbd>end</kbd> prima non revisionata &nbsp; <kbd>backspace</kbd> cancella decisione
  &nbsp;&nbsp; decisione corrente: <span id="decision">-</span>
</div>
<div id="help2">
  trascina un'area vuota per creare una bbox &nbsp; clic/trascina su una bbox per selezionarla/spostarla &nbsp;
  trascina i quadratini per ridimensionare &nbsp; <kbd>x</kbd>/<kbd>canc</kbd> elimina selezionata &nbsp; <kbd>esc</kbd> deseleziona &nbsp; <kbd>h</kbd> mostra/nascondi bbox &nbsp; <kbd>e</kbd> mostra/nascondi pallino rosso (bbox modificate)
  &nbsp;&nbsp; classe: <select id="class-picker" title="Classe delle nuove bbox; se una bbox è selezionata, la riassegna subito"></select>
  &nbsp;&nbsp; bbox: <span id="box-status">-</span>
  &nbsp;&nbsp; <kbd>+</kbd>/<kbd>-</kbd> zoom/dimensione originale <span id="zoom-status"></span>
  <button id="restore-btn" title="Ripristina le bbox originali (tasto r)">ripristina originali (r)</button>
</div>
<div id="done">Fine: tutte le immagini sono state revisionate.</div>
<div id="empty">Nessuna immagine corrisponde ai filtri selezionati.</div>
</div>
</div>
</div>

<script>
let names = [];
let decisions = {};
let flagged = {}; // {name: [descrizioni]} detection scartate per overlap sosia, v. load_flagged()
let phashes = null; // caricato pigramente, {name: hex}, solo se serve per l'ordinamento
let classMembership = {}; // {stem immagine: [classi presenti nel label ATTUALE]}, da loadClassStats()
let view = [];       // elenco filtrato/ordinato attualmente in navigazione
let idx = 0;
let renderToken = 0;

const img = document.getElementById("img");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const fDataset = document.getElementById("f-dataset");
const fStatus = document.getElementById("f-status");
const fClass = document.getElementById("f-class");
const fSearch = document.getElementById("f-search");
const sort1 = document.getElementById("sort1");
const sort2 = document.getElementById("sort2");
const gotoInput = document.getElementById("goto-input");
const gotoBtn = document.getElementById("goto-btn");

function datasetOf(name) {
  const i = name.indexOf("__");
  return i === -1 ? name : name.slice(0, i);
}

function stemOf(name) {
  const i = name.lastIndexOf(".");
  return i === -1 ? name : name.slice(0, i);
}

function statusOf(name) {
  const d = decisions[name];
  return d === "select" || d === "discard" ? d : "todo";
}

const STATUS_ORDER = { todo: 0, select: 1, discard: 2 };

async function loadState() {
  const r = await fetch("/api/state");
  const data = await r.json();
  names = data.names;
  decisions = data.decisions;
  flagged = data.flagged;
  populateDatasetFilter();
  computeView();
  idx = view.findIndex(n => !(n in decisions));
  if (idx === -1) idx = 0;
  render();
  loadClassStats();
}

async function loadClassStats() {
  const r = await fetch("/api/class-stats");
  const data = await r.json();
  classMembership = data.by_image;
  document.getElementById("class-stats-body").innerHTML = data.stats.slice(0, 40).map(s => `
    <tr>
      <td>${s.cls}</td>
      <td>${classLabel(s.cls)}</td>
      <td class="num">${s.n_images}</td>
      <td class="num">${s.n_instances}</td>
    </tr>`).join("");
  // il filtro per classe dipende da classMembership, appena aggiornato: se è attivo, l'immagine
  // corrente potrebbe non farne più parte (es. hai appena tolto l'ultima bbox di quella classe) —
  // stesso trattamento già usato altrove quando un'immagine esce dal filtro corrente.
  if (fClass.value !== "" && view.length) {
    const name = view[idx];
    computeView();
    const keep = view.indexOf(name);
    if (keep === -1) {
      idx = Math.min(idx, view.length - 1);
      refresh();
    } else {
      idx = keep;
      updateCounts();
    }
  }
}

function populateDatasetFilter() {
  const datasets = [...new Set(names.map(datasetOf))].sort();
  const current = fDataset.value;
  fDataset.innerHTML = '<option value="">tutti</option>' +
    datasets.map(d => `<option value="${d}">${d}</option>`).join("");
  fDataset.value = datasets.includes(current) ? current : "";
}

function compareByField(field, a, b) {
  if (field === "filename") return a < b ? -1 : a > b ? 1 : 0;
  if (field === "dataset") {
    const da = datasetOf(a), db = datasetOf(b);
    return da < db ? -1 : da > db ? 1 : 0;
  }
  if (field === "status") return STATUS_ORDER[statusOf(a)] - STATUS_ORDER[statusOf(b)];
  if (field === "phash") {
    const ha = (phashes && phashes[a]) || "", hb = (phashes && phashes[b]) || "";
    return ha < hb ? -1 : ha > hb ? 1 : 0;
  }
  return 0;
}

async function ensurePhashesIfNeeded() {
  if (sort1.value !== "phash" && sort2.value !== "phash") return;
  if (phashes) return;
  const statusEl = document.getElementById("phash-status");
  statusEl.textContent = "calcolo pHash in corso...";
  const r = await fetch("/api/phashes");
  const data = await r.json();
  phashes = data.phashes;
  statusEl.textContent = "";
}

function computeView() {
  const wantedDataset = fDataset.value;
  const wantedStatus = fStatus.value;
  const wantedClass = fClass.value === "" ? null : parseInt(fClass.value, 10);
  const query = fSearch.value.trim().toLowerCase();
  let filtered = names.filter(n => {
    if (query && !n.toLowerCase().includes(query)) return false;
    if (wantedDataset && datasetOf(n) !== wantedDataset) return false;
    if (wantedStatus && statusOf(n) !== wantedStatus) return false;
    if (wantedClass !== null && !(classMembership[stemOf(n)] || []).includes(wantedClass)) return false;
    return true;
  });
  const k1 = sort1.value, k2 = sort2.value;
  if (k1 || k2) {
    filtered = filtered.slice().sort((a, b) => {
      if (k1) { const c = compareByField(k1, a, b); if (c !== 0) return c; }
      if (k2) { const c = compareByField(k2, a, b); if (c !== 0) return c; }
      return a < b ? -1 : a > b ? 1 : 0;
    });
  }
  view = filtered;
}

async function applyFiltersAndSort() {
  const currentName = view[idx];
  await ensurePhashesIfNeeded();
  computeView();
  if (currentName === undefined) {
    idx = 0;
  } else {
    const keep = view.indexOf(currentName);
    // se l'immagine corrente è uscita dal filtro (es. dopo una decisione),
    // resta sulla stessa posizione: quella successiva prende il suo posto
    idx = keep !== -1 ? keep : Math.min(idx, view.length - 1);
  }
  if (mode === "browse") { browseOffset = 0; idx = 0; } // elenco cambiato: si riparte dalla prima pagina
  refresh();
}

[fDataset, fStatus, fClass, sort1, sort2].forEach(el => el.addEventListener("change", applyFiltersAndSort));

// ricerca per nome file: ricalcola a ogni battuta, con un breve ritardo per non rifare il lavoro a ogni tasto
let searchTimer = null;
fSearch.addEventListener("input", () => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(applyFiltersAndSort, 200);
});
fSearch.addEventListener("keydown", (e) => {
  if (e.key === "Enter") fSearch.blur(); // rimette i tasti rapidi (s, n, frecce) sull'immagine
});

// Due modalità: "single" (un'immagine alla volta, modifica bbox) e "browse" (griglia di miniature
// a blocchi di BROWSE_PAGE, sola lettura con le bbox disegnate). Entrambe lavorano sullo stesso
// elenco filtrato/ordinato `view`; `idx` resta l'immagine corrente, condivisa fra le due.
const BROWSE_PAGE = 12;
let mode = "single";
let browseOffset = 0;
let browseToken = 0;

function refresh() {
  if (mode === "browse") renderBrowse(); else render();
}

function setMode(newMode) {
  mode = newMode;
  document.getElementById("mode-single").classList.toggle("active", mode === "single");
  document.getElementById("mode-browse").classList.toggle("active", mode === "browse");
  document.getElementById("single").hidden = mode !== "single";
  document.getElementById("browse").hidden = mode !== "browse";
  if (mode === "browse") {
    browseOffset = Math.floor(idx / BROWSE_PAGE) * BROWSE_PAGE; // la pagina che contiene l'immagine corrente
    renderBrowse();
  } else {
    render();
  }
}

function drawThumbBoxes(im) {
  const cv = im.parentElement.querySelector("canvas");
  const cw = im.parentElement.clientWidth, ch = im.parentElement.clientHeight;
  cv.width = cw; cv.height = ch;
  const nw = im.naturalWidth, nh = im.naturalHeight;
  if (!nw || !nh || !im._boxes) return;
  const sc = Math.min(cw / nw, ch / nh), dw = nw * sc, dh = nh * sc;
  const ox = (cw - dw) / 2, oy = (ch - dh) / 2;
  const tctx = cv.getContext("2d");
  const boxes = im._boxes.slice().sort((a, b) => (a[0] === DEFAULT_CLASS) - (b[0] === DEFAULT_CLASS)); // escooter per ultime
  for (const [cls, xc, yc, w, h] of boxes) {
    tctx.strokeStyle = classColor(cls);
    tctx.lineWidth = cls === DEFAULT_CLASS ? 3 : 1.5;
    tctx.strokeRect(ox + (xc - w / 2) * dw, oy + (yc - h / 2) * dh, w * dw, h * dh);
    if (modifiedDotsVisible && isModifiedBox(cls, xc, yc, w, h, im._original)) {
      drawModifiedDot(tctx, ox + xc * dw, oy + (yc - h / 2) * dh, 3.5);
    }
  }
}

async function renderBrowse() {
  const grid = document.getElementById("browse-grid");
  const info = document.getElementById("browse-info");
  const token = ++browseToken;
  document.getElementById("browse-empty").hidden = view.length !== 0;
  if (view.length === 0) {
    grid.replaceChildren();
    info.textContent = "0 immagini";
    return;
  }
  browseOffset = Math.min(Math.max(browseOffset, 0), Math.floor((view.length - 1) / BROWSE_PAGE) * BROWSE_PAGE);
  const page = view.slice(browseOffset, browseOffset + BROWSE_PAGE);
  info.textContent = `${browseOffset + 1}-${browseOffset + page.length} di ${view.length}`;
  updateCounts();
  const cards = page.map((name, i) => {
    const card = document.createElement("div");
    card.className = "card " + statusOf(name) + (browseOffset + i === idx ? " current" : "");
    const wrap = document.createElement("div");
    wrap.className = "imgwrap";
    const im = document.createElement("img");
    im.loading = "lazy";
    im.onload = () => drawThumbBoxes(im);
    im.src = "/api/image/" + encodeURIComponent(name);
    wrap.append(im, document.createElement("canvas"));
    const meta = document.createElement("div");
    meta.className = "meta";
    meta.textContent = `#${browseOffset + i + 1}  ${name}`;
    card.append(wrap, meta);
    card.onclick = () => { idx = browseOffset + i; setMode("single"); };
    return card;
  });
  grid.replaceChildren(...cards);
  const boxesList = await Promise.all(page.map(n => fetch("/api/boxes/" + encodeURIComponent(n)).then(r => r.json())));
  if (token !== browseToken) return; // superata da una navigazione più recente
  cards.forEach((card, i) => {
    const im = card.querySelector("img");
    im._boxes = boxesList[i].boxes;
    im._original = boxesList[i].original;
    if (im.complete) drawThumbBoxes(im);
  });
}

function browseStep(delta) {
  const last = Math.floor((view.length - 1) / BROWSE_PAGE) * BROWSE_PAGE;
  browseOffset = Math.min(Math.max(browseOffset + delta * BROWSE_PAGE, 0), Math.max(last, 0));
  renderBrowse();
}

document.getElementById("mode-single").addEventListener("click", () => setMode("single"));
document.getElementById("mode-browse").addEventListener("click", () => setMode("browse"));
document.getElementById("browse-prev").addEventListener("click", () => browseStep(-1));
document.getElementById("browse-next").addEventListener("click", () => browseStep(1));

function updateCounts() {
  let s=0, n=0;
  for (const name of names) {
    const d = decisions[name];
    if (d === "select") s++;
    else if (d === "discard") n++;
  }
  document.getElementById("c-select").textContent = s;
  document.getElementById("c-discard").textContent = n;
  document.getElementById("c-todo").textContent = names.length - s - n;
  document.getElementById("total").textContent = view.length;
}

const DEFAULT_CLASS = __ESCOOTER_CLASS_ID__;

// Nomi classe COCO standard (id 0-79, ordine Ultralytics) e palette categorica per il
// colore delle bbox: utile sui dataset con più classi (es. union_reviewed_coco), dove le
// bbox non sono tutte escooter. L'escooter (DEFAULT_CLASS) mantiene il rosso di sempre.
const COCO_NAMES = [
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
];
const CLASS_PALETTE = [
  "#4cc9f0", "#4895ef", "#4361ee", "#7209b7", "#b5179e", "#f72585", "#ff9e00", "#ffca3a",
  "#8ac926", "#52b788", "#219ebc", "#fb8500", "#a8dadc", "#ffb4a2", "#cdb4db", "#90e0ef",
  "#ff6d00", "#06d6a0", "#ffd166", "#c77dff",
];

function classLabel(cls) {
  if (cls === DEFAULT_CLASS) return "escooter";
  return COCO_NAMES[cls] || String(cls);
}

function classColor(cls) {
  if (cls === DEFAULT_CLASS) return "#ff2d55";
  return CLASS_PALETTE[((cls % CLASS_PALETTE.length) + CLASS_PALETTE.length) % CLASS_PALETTE.length];
}
const HANDLE_SIZE = 10;
const MIN_BOX_PX = 6;
const CURSOR_FOR_HANDLE = {
  nw: "nwse-resize", se: "nwse-resize",
  ne: "nesw-resize", sw: "nesw-resize",
  n: "ns-resize", s: "ns-resize",
  w: "ew-resize", e: "ew-resize",
};

let currentBoxes = []; // [{cls, xc, yc, w, h}, ...] normalizzati 0..1
let originalBoxes = null; // [[cls, xc, yc, w, h], ...] del backup, null se l'immagine non è mai stata modificata
let modifiedDotsVisible = true; // tasto e: pallino rosso sulle bbox nuove o ritoccate

// Una bbox è "modificata" se esiste un backup e nessuna bbox originale le coincide (stessa classe,
// coordinate entro 1e-5: i label sono scritti con 6 decimali). Vale per bbox nuove, spostate,
// ridimensionate o con classe cambiata; senza backup nessuna lo è.
function isModifiedBox(cls, xc, yc, w, h, original) {
  if (!original) return false;
  return !original.some(o => o[0] === cls && Math.abs(o[1] - xc) < 1e-5 && Math.abs(o[2] - yc) < 1e-5
    && Math.abs(o[3] - w) < 1e-5 && Math.abs(o[4] - h) < 1e-5);
}

// Pallino rosso a metà del bordo superiore, con contorno bianco per leggersi su qualsiasi sfondo.
function drawModifiedDot(c, x, y, radius) {
  c.beginPath();
  c.arc(x, y, radius, 0, Math.PI * 2);
  c.fillStyle = "#e53935";
  c.fill();
  c.lineWidth = 1.5;
  c.strokeStyle = "#ffffff";
  c.stroke();
}
let currentFlagged = []; // [{cls, xc, yc, w, h, label}, ...] scartate per overlap sosia, sola lettura
let selectedIndex = -1;
let drag = null;

// Classe da assegnare alla prossima bbox creata (persiste da un'immagine all'altra, comodo per
// colmare in sequenza più falsi negativi della stessa classe). Se una bbox è selezionata, il
// menu mostra e permette di cambiare la SUA classe (v. syncClassPicker() e il listener "change").
let newBoxClass = DEFAULT_CLASS;

// Popola un <select> con escooter + le 80 classi COCO (id e nome), condiviso fra il menu classe
// delle bbox e il filtro per classe della filterbar.
function populateClassOptions(sel) {
  const opt = (value, label) => { const o = document.createElement("option"); o.value = value; o.textContent = label; return o; };
  sel.appendChild(opt(DEFAULT_CLASS, "escooter"));
  const group = document.createElement("optgroup");
  group.label = "classi COCO";
  COCO_NAMES.forEach((name, i) => group.appendChild(opt(i, `${i} ${name}`)));
  sel.appendChild(group);
}

(function initClassPicker() {
  const sel = document.getElementById("class-picker");
  populateClassOptions(sel);
  sel.value = String(newBoxClass);
  sel.addEventListener("change", () => {
    newBoxClass = parseInt(sel.value, 10);
    if (selectedIndex !== -1) {
      currentBoxes[selectedIndex].cls = newBoxClass;
      drawBoxes(currentBoxes);
      saveBoxes();
    }
  });
})();
(function initClassFilter() {
  populateClassOptions(document.getElementById("f-class"));
})();
let creatingRect = null;
let boxesVisible = true;
const FLAGGED_COLOR = "#ff8c00";

function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

function boxPixelRect(b) {
  const w = b.w * canvas.width, h = b.h * canvas.height;
  const x = b.xc * canvas.width - w / 2;
  const y = b.yc * canvas.height - h / 2;
  return { x, y, w, h };
}

function handlesFor(r) {
  return [
    { name: "nw", x: r.x,         y: r.y,         left: true,  top: true },
    { name: "ne", x: r.x + r.w,   y: r.y,         right: true, top: true },
    { name: "sw", x: r.x,         y: r.y + r.h,   left: true,  bottom: true },
    { name: "se", x: r.x + r.w,   y: r.y + r.h,   right: true, bottom: true },
    { name: "n",  x: r.x + r.w/2, y: r.y,         top: true },
    { name: "s",  x: r.x + r.w/2, y: r.y + r.h,   bottom: true },
    { name: "w",  x: r.x,         y: r.y + r.h/2, left: true },
    { name: "e",  x: r.x + r.w,   y: r.y + r.h/2, right: true },
  ];
}

function handleAt(r, px, py) {
  for (const h of handlesFor(r)) {
    if (Math.abs(px - h.x) <= HANDLE_SIZE / 2 && Math.abs(py - h.y) <= HANDLE_SIZE / 2) return h;
  }
  return null;
}

function getMousePos(e) {
  const rect = canvas.getBoundingClientRect();
  // limitata al canvas: trascinando oltre il bordo dell'immagine la bbox non può sporgere
  return { x: clamp(e.clientX - rect.left, 0, canvas.width), y: clamp(e.clientY - rect.top, 0, canvas.height) };
}

function resizeCanvas() {
  canvas.width = img.clientWidth;
  canvas.height = img.clientHeight;
  canvas.style.width = img.clientWidth + "px";
  canvas.style.height = img.clientHeight + "px";
  // allineamento esplicito invece di affidarsi al posizionamento statico di un elemento
  // absolute dentro un contenitore flex: resta corretto sia centrato (zoom 1x) sia in
  // modalità scroll (zoomato, v. applyZoom()).
  canvas.style.left = img.offsetLeft + "px";
  canvas.style.top = img.offsetTop + "px";
}

// Zoom (tasti + / -): ridimensiona img e canvas ai pixel, non uno scale CSS — così la
// matematica delle bbox (in pixel di canvas) resta identica a zoom 1x, e #stage.zoomed
// passa a overflow:auto per scorrere l'immagine ingrandita. baseWidth/baseHeight sono le
// dimensioni "adattate alla vista" (zoom 1x), ricalcolate a ogni cambio immagine.
let zoomLevel = 1;
let baseWidth = 0, baseHeight = 0;
const ZOOM_STEP = 1.25, ZOOM_MAX = 5;
const stageEl = document.getElementById("stage");

function applyZoom() {
  if (zoomLevel === 1) {
    img.style.width = "";
    img.style.height = "";
    stageEl.classList.remove("zoomed");
    baseWidth = img.clientWidth;
    baseHeight = img.clientHeight;
  } else {
    img.style.width = Math.round(baseWidth * zoomLevel) + "px";
    img.style.height = Math.round(baseHeight * zoomLevel) + "px";
    stageEl.classList.add("zoomed");
  }
  resizeCanvas();
  drawBoxes(currentBoxes);
  document.getElementById("zoom-status").textContent = zoomLevel === 1 ? "" : `(${Math.round(zoomLevel * 100)}%)`;
}

function drawBoxLabel(r, text, color) {
  ctx.font = "12px system-ui, sans-serif";
  const padX = 4, padY = 2;
  const w = ctx.measureText(text).width + padX * 2;
  const h = 14 + padY * 2;
  const x = r.x, y = Math.max(0, r.y - h);
  ctx.fillStyle = "rgba(0, 0, 0, 0.65)";
  ctx.fillRect(x, y, w, h);
  ctx.fillStyle = color;
  ctx.fillText(text, x + padX, y + h - padY - 3);
}

function drawBoxes(boxes) {
  // il menu classe segue la selezione: mostra la classe della bbox selezionata (e la si può
  // riassegnare da lì), o quella pronta per la prossima bbox nuova se nessuna è selezionata.
  document.getElementById("class-picker").value =
    String(selectedIndex !== -1 && boxes[selectedIndex] ? boxes[selectedIndex].cls : newBoxClass);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (boxesVisible) {
    boxes.forEach((b, i) => {
      const r = boxPixelRect(b);
      const selected = i === selectedIndex;
      const color = classColor(b.cls);
      ctx.strokeStyle = selected ? "#ffd60a" : color;
      ctx.lineWidth = selected ? 3 : 2;
      ctx.strokeRect(r.x, r.y, r.w, r.h);
      drawBoxLabel(r, classLabel(b.cls), selected ? "#ffd60a" : color);
      if (modifiedDotsVisible && isModifiedBox(b.cls, b.xc, b.yc, b.w, b.h, originalBoxes)) {
        drawModifiedDot(ctx, r.x + r.w / 2, r.y, 5);
      }
      if (selected) {
        ctx.fillStyle = "#ffd60a";
        for (const h of handlesFor(r)) {
          ctx.fillRect(h.x - HANDLE_SIZE / 2, h.y - HANDLE_SIZE / 2, HANDLE_SIZE, HANDLE_SIZE);
        }
      }
    });
    // detection scartate per overlap sosia (v. flagged_coco_overlap.txt): non sono nel label,
    // quindi non selezionabili/modificabili — solo lettura, tratteggiate per distinguerle da
    // quelle rimaste nel dataset, in un colore dedicato indipendente dalla classe.
    currentFlagged.forEach(b => {
      const r = boxPixelRect(b);
      ctx.save();
      ctx.setLineDash([6, 4]);
      ctx.strokeStyle = FLAGGED_COLOR;
      ctx.lineWidth = 2;
      ctx.strokeRect(r.x, r.y, r.w, r.h);
      ctx.restore();
      drawBoxLabel(r, classLabel(b.cls) + " (scartata)", FLAGGED_COLOR);
    });
  }
  // il rettangolo in corso di creazione resta visibile anche a bbox nascoste:
  // è quello che permette di disegnarne di nuove "alla cieca" sull'ammucchiata.
  if (creatingRect) {
    ctx.strokeStyle = "#4caf50";
    ctx.lineWidth = 2;
    ctx.setLineDash([6, 4]);
    ctx.strokeRect(creatingRect.x, creatingRect.y, creatingRect.w, creatingRect.h);
    ctx.setLineDash([]);
  }
}

function setBoxStatus(text) {
  document.getElementById("box-status").textContent = text;
}

function setModifiedBadge(modified) {
  document.getElementById("modified-badge").hidden = !modified;
}

async function saveBoxes() {
  const name = view[idx];
  const payload = currentBoxes.map(b => [b.cls, b.xc, b.yc, b.w, b.h]);
  setBoxStatus("salvataggio...");
  try {
    const r = await fetch("/api/boxes/" + encodeURIComponent(name), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ boxes: payload }),
    });
    const data = await r.json();
    setModifiedBadge(data.modified);
    originalBoxes = data.original; // al primo salvataggio nasce il backup: da qui in poi i pallini hanno un riferimento
    drawBoxes(currentBoxes);
    setBoxStatus("salvato (" + currentBoxes.length + ")");
    loadClassStats();
  } catch (err) {
    setBoxStatus("errore salvataggio");
  }
}

async function restoreBoxes() {
  if (view.length === 0) return;
  const name = view[idx];
  if (!confirm("Ripristinare le bbox originali per questa immagine? Le modifiche correnti andranno perse.")) return;
  setBoxStatus("ripristino...");
  const r = await fetch("/api/restore-boxes/" + encodeURIComponent(name), { method: "POST" });
  const data = await r.json();
  currentBoxes = data.boxes.map(([cls, xc, yc, w, h]) => ({ cls, xc, yc, w, h }));
  selectedIndex = -1;
  originalBoxes = data.original;
  drawBoxes(currentBoxes);
  setModifiedBadge(data.modified);
  setBoxStatus((data.restored ? "ripristinato (" : "già originale (") + currentBoxes.length + ")");
  loadClassStats();
}

document.getElementById("restore-btn").addEventListener("click", restoreBoxes);

function goToImageNumber() {
  const n = parseInt(gotoInput.value, 10);
  if (!Number.isFinite(n) || view.length === 0) return;
  idx = Math.min(Math.max(n, 1), view.length) - 1;
  render();
}

gotoBtn.addEventListener("click", goToImageNumber);
gotoInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") { goToImageNumber(); gotoInput.blur(); }
});

function toggleBoxesVisibility() {
  boxesVisible = !boxesVisible;
  if (!boxesVisible) selectedIndex = -1; // evita di trascinare/ridimensionare una bbox che non si vede
  drawBoxes(currentBoxes);
  setBoxStatus(boxesVisible
    ? (currentBoxes.length ? String(currentBoxes.length) : "-")
    : "nascoste (" + currentBoxes.length + ") — disegno nuove bbox ancora attivo");
}

function toggleModifiedDots() {
  modifiedDotsVisible = !modifiedDotsVisible;
  if (mode === "browse") document.querySelectorAll("#browse-grid img").forEach(drawThumbBoxes);
  else {
    drawBoxes(currentBoxes);
    if (boxesVisible) setBoxStatus((currentBoxes.length ? String(currentBoxes.length) : "-")
      + (modifiedDotsVisible ? "" : " — pallini rossi nascosti"));
  }
}

async function render() {
  if (view.length === 0) {
    document.getElementById("stage").style.display = "none";
    document.getElementById("name").style.display = "none";
    document.getElementById("empty").style.display = "block";
    document.getElementById("idx").textContent = 0;
    return;
  }
  document.getElementById("stage").style.display = "flex";
  document.getElementById("name").style.display = "block";
  document.getElementById("empty").style.display = "none";
  document.getElementById("done").style.display = "none";
  const token = ++renderToken;
  const name = view[idx];
  document.getElementById("idx").textContent = idx + 1;
  document.getElementById("fname").textContent = name;
  document.getElementById("res-info").textContent = "";
  updateDecisionLabel();
  updateCounts();
  updateFlaggedBanner(name);

  const boxesPromise = fetch("/api/boxes/" + encodeURIComponent(name)).then(r => r.json());
  const imageLoaded = new Promise((resolve) => { img.onload = resolve; });
  img.src = "/api/image/" + encodeURIComponent(name);

  const [boxesData] = await Promise.all([boxesPromise, imageLoaded]);
  if (token !== renderToken) return; // superata da una navigazione più recente
  document.getElementById("res-info").textContent = img.naturalWidth + " × " + img.naturalHeight + " px";
  setModifiedBadge(boxesData.modified);
  originalBoxes = boxesData.original;
  currentBoxes = boxesData.boxes.map(([cls, xc, yc, w, h]) => ({ cls, xc, yc, w, h }));
  currentFlagged = (flagged[name] || []).map(f => {
    const [cls, xc, yc, w, h] = f.box;
    return { cls, xc, yc, w, h };
  });
  selectedIndex = -1;
  drag = null;
  creatingRect = null;
  setBoxStatus(currentBoxes.length ? String(currentBoxes.length) : "-");
  zoomLevel = 1;
  applyZoom(); // ricalcola anche baseWidth/baseHeight per la nuova immagine, poi resize+draw
}

function updateDecisionLabel() {
  const d = decisions[view[idx]];
  const el = document.getElementById("decision");
  el.className = "";
  if (d === "select") { el.textContent = "SELEZIONATA"; el.className = "sel"; }
  else if (d === "discard") { el.textContent = "SCARTATA"; el.className = "disc"; }
  else { el.textContent = "-"; }
}

function updateFlaggedBanner(name) {
  const entries = flagged[name];
  const banner = document.getElementById("flagged-banner");
  if (entries && entries.length) {
    banner.textContent = "⚠ " + entries.length + " detection scartate per overlap con l'escooter, "
      + "disegnate tratteggiate in arancione (valuta il recupero in coco_overlap_rescue.txt): "
      + entries.map(e => e.label).join("  ·  ");
    banner.style.display = "block";
  } else {
    banner.style.display = "none";
  }
}

async function setDecision(decision) {
  if (view.length === 0) return;
  const name = view[idx];
  if (decision === null) delete decisions[name];
  else decisions[name] = decision;
  await fetch("/api/decision", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, decision }),
  });
  // la decisione può cambiare l'ordinamento/filtro (es. ordinati o filtrati per stato):
  // ricalcola la vista, tenendo il riferimento a dove si trovava l'immagine appena decisa.
  computeView();
  const keep = view.indexOf(name);
  if (decision === null) {
    idx = keep !== -1 ? keep : Math.min(idx, view.length - 1);
    render();
    return;
  }
  if (keep !== -1) {
    // ancora in vista (eventualmente spostata dal riordino): avanza da lì
    if (keep < view.length - 1) {
      idx = keep + 1;
      render();
    } else {
      // era l'ultima della vista: resta lì e mostra il messaggio di fine
      // (niente render(), altrimenti azzererebbe subito il messaggio)
      idx = keep;
      updateDecisionLabel();
      updateCounts();
      document.getElementById("done").style.display = "block";
    }
  } else {
    // uscita dal filtro corrente: chi occupava la sua posizione prende il suo posto
    idx = Math.min(idx, view.length - 1);
    render();
  }
}

canvas.addEventListener("mousedown", (e) => {
  const pos = getMousePos(e);

  // a bbox nascoste si ignorano quelle esistenti (non selezionabili/spostabili
  // perché invisibili) e si va sempre in modalità "crea nuova bbox".
  if (boxesVisible) {
    if (selectedIndex !== -1) {
      const h = handleAt(boxPixelRect(currentBoxes[selectedIndex]), pos.x, pos.y);
      if (h) {
        drag = { mode: "resize", index: selectedIndex, handle: h, orig: boxPixelRect(currentBoxes[selectedIndex]) };
        e.preventDefault();
        return;
      }
    }

    let hitIndex = -1, hitArea = Infinity;
    currentBoxes.forEach((b, i) => {
      const r = boxPixelRect(b);
      if (pos.x >= r.x && pos.x <= r.x + r.w && pos.y >= r.y && pos.y <= r.y + r.h) {
        const area = r.w * r.h;
        if (area < hitArea) { hitArea = area; hitIndex = i; }
      }
    });
    if (hitIndex !== -1) {
      selectedIndex = hitIndex;
      drag = { mode: "move", index: hitIndex, startX: pos.x, startY: pos.y, orig: boxPixelRect(currentBoxes[hitIndex]) };
      drawBoxes(currentBoxes);
      return;
    }
  }

  selectedIndex = -1;
  drag = { mode: "create", startX: pos.x, startY: pos.y };
  creatingRect = { x: pos.x, y: pos.y, w: 0, h: 0 };
  drawBoxes(currentBoxes);
});

window.addEventListener("mousemove", (e) => {
  if (!drag) {
    let cursor = "crosshair";
    if (boxesVisible) {
      const pos = getMousePos(e);
      if (selectedIndex !== -1) {
        const h = handleAt(boxPixelRect(currentBoxes[selectedIndex]), pos.x, pos.y);
        if (h) cursor = CURSOR_FOR_HANDLE[h.name];
      }
      if (cursor === "crosshair") {
        for (const b of currentBoxes) {
          const r = boxPixelRect(b);
          if (pos.x >= r.x && pos.x <= r.x + r.w && pos.y >= r.y && pos.y <= r.y + r.h) { cursor = "move"; break; }
        }
      }
    }
    canvas.style.cursor = cursor;
    return;
  }

  const pos = getMousePos(e);
  if (drag.mode === "move") {
    const dx = pos.x - drag.startX, dy = pos.y - drag.startY;
    const nx = clamp(drag.orig.x + dx, 0, canvas.width - drag.orig.w);
    const ny = clamp(drag.orig.y + dy, 0, canvas.height - drag.orig.h);
    const b = currentBoxes[drag.index];
    b.xc = (nx + drag.orig.w / 2) / canvas.width;
    b.yc = (ny + drag.orig.h / 2) / canvas.height;
    drawBoxes(currentBoxes);
  } else if (drag.mode === "resize") {
    const r = drag.orig, h = drag.handle;
    let left = h.left ? pos.x : r.x;
    let right = h.right ? pos.x : r.x + r.w;
    let top = h.top ? pos.y : r.y;
    let bottom = h.bottom ? pos.y : r.y + r.h;
    left = clamp(left, 0, canvas.width);
    right = clamp(right, 0, canvas.width);
    top = clamp(top, 0, canvas.height);
    bottom = clamp(bottom, 0, canvas.height);
    if (right - left < MIN_BOX_PX) { if (h.left) left = right - MIN_BOX_PX; else right = left + MIN_BOX_PX; }
    if (bottom - top < MIN_BOX_PX) { if (h.top) top = bottom - MIN_BOX_PX; else bottom = top + MIN_BOX_PX; }
    // la dimensione minima non deve rimandare il lato fuori dall'immagine
    left = clamp(left, 0, canvas.width); right = clamp(right, 0, canvas.width);
    top = clamp(top, 0, canvas.height); bottom = clamp(bottom, 0, canvas.height);
    const b = currentBoxes[drag.index];
    b.xc = (left + right) / 2 / canvas.width;
    b.yc = (top + bottom) / 2 / canvas.height;
    b.w = (right - left) / canvas.width;
    b.h = (bottom - top) / canvas.height;
    drawBoxes(currentBoxes);
  } else if (drag.mode === "create") {
    creatingRect = {
      x: Math.min(drag.startX, pos.x), y: Math.min(drag.startY, pos.y),
      w: Math.abs(pos.x - drag.startX), h: Math.abs(pos.y - drag.startY),
    };
    drawBoxes(currentBoxes);
  }
});

window.addEventListener("mouseup", () => {
  if (!drag) return;
  if (drag.mode === "create") {
    if (creatingRect.w >= MIN_BOX_PX && creatingRect.h >= MIN_BOX_PX) {
      currentBoxes.push({
        cls: newBoxClass,
        xc: (creatingRect.x + creatingRect.w / 2) / canvas.width,
        yc: (creatingRect.y + creatingRect.h / 2) / canvas.height,
        w: creatingRect.w / canvas.width,
        h: creatingRect.h / canvas.height,
      });
      selectedIndex = currentBoxes.length - 1;
    }
    creatingRect = null;
  }
  drag = null;
  drawBoxes(currentBoxes);
  saveBoxes();
});

document.addEventListener("keydown", (e) => {
  if (document.activeElement && ["SELECT", "INPUT"].includes(document.activeElement.tagName)) return;
  if (e.key === "b") { setMode(mode === "browse" ? "single" : "browse"); return; }
  if (e.key === "e") { toggleModifiedDots(); return; }
  if (mode === "browse") { // in browse le frecce cambiano pagina, gli altri tasti (decisioni, bbox) non agiscono
    if (e.key === "ArrowRight") browseStep(1);
    else if (e.key === "ArrowLeft") browseStep(-1);
    return;
  }
  if (e.key === "s") setDecision("select");
  else if (e.key === "n") setDecision("discard");
  else if (e.key === "r") restoreBoxes();
  else if (e.key === "h") toggleBoxesVisibility();
  else if (e.key === "+" || e.key === "=") { zoomLevel = Math.min(zoomLevel * ZOOM_STEP, ZOOM_MAX); applyZoom(); }
  else if (e.key === "-") { zoomLevel = 1; applyZoom(); }
  else if (e.key === "Backspace") setDecision(null);
  else if (e.key === "ArrowRight") { idx = Math.min(idx + 1, view.length - 1); render(); }
  else if (e.key === "ArrowLeft") { idx = Math.max(idx - 1, 0); render(); }
  else if (e.key === "End") {
    const i = view.findIndex(n => !(n in decisions));
    if (i !== -1) { idx = i; render(); }
  }
  else if ((e.key === "Delete" || e.key === "x") && selectedIndex !== -1) {
    currentBoxes.splice(selectedIndex, 1);
    selectedIndex = -1;
    drawBoxes(currentBoxes);
    saveBoxes();
  } else if (e.key === "Escape" && selectedIndex !== -1) {
    selectedIndex = -1;
    drawBoxes(currentBoxes);
  }
});

window.addEventListener("resize", () => {
  if (mode === "browse") { document.querySelectorAll("#browse-grid img").forEach(drawThumbBoxes); return; }
  if (zoomLevel === 1) applyZoom(); // ricalcola baseWidth/baseHeight per il nuovo viewport
  else { resizeCanvas(); drawBoxes(currentBoxes); } // zoomati: dimensioni esplicite, non dipendono dal viewport
});
loadState();
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    # Impostati su main() prima dell'avvio del server.
    dataset_dir: Path
    decisions_path: Path
    backup_dir: Path
    phash_cache_path: Path
    flagged_path: Path

    def _json(self, obj: dict, status: int = 200) -> None:
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            body = PAGE.replace("__ESCOOTER_CLASS_ID__", str(config.ESCOOTER_CLASS_ID)).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/state":
            names = sorted(p.name for p in (self.dataset_dir / "images").iterdir())
            self._json({
                "names": names, "decisions": load_decisions(self.decisions_path),
                "flagged": load_flagged(self.flagged_path),
            })
        elif self.path.startswith("/api/image/"):
            name = unquote(self.path[len("/api/image/"):])
            path = self.dataset_dir / "images" / name
            if not path.exists():
                self.send_error(404)
                return
            data = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type(name)[0] or "application/octet-stream")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        elif self.path.startswith("/api/boxes/"):
            name = unquote(self.path[len("/api/boxes/"):])
            lbl_path = self.dataset_dir / "labels" / f"{Path(name).stem}.txt"
            backup_path = self.backup_dir / f"{Path(name).stem}.txt"
            self._json({
                "boxes": read_boxes_file(lbl_path), "modified": boxes_modified(lbl_path, backup_path),
                "original": original_boxes(backup_path),
            })
        elif self.path == "/api/phashes":
            self._json({"phashes": compute_phashes(self.dataset_dir, self.phash_cache_path)})
        elif self.path == "/api/class-stats":
            stats, by_image = class_stats(self.dataset_dir / "labels")
            self._json({"stats": stats, "by_image": by_image})
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        if self.path == "/api/decision":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            name, decision = body["name"], body["decision"]
            decisions = load_decisions(self.decisions_path)
            if decision is None:
                decisions.pop(name, None)
            else:
                decisions[name] = decision
            save_decisions(self.decisions_path, decisions)
            self._json({"ok": True})
        elif self.path.startswith("/api/boxes/"):
            name = unquote(self.path[len("/api/boxes/"):])
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            lbl_path = self.dataset_dir / "labels" / f"{Path(name).stem}.txt"
            backup_label_once(self.backup_dir, name, lbl_path)
            lines = []
            for cls, xc, yc, w, h in body["boxes"]:
                clipped = clip_box(float(xc), float(yc), float(w), float(h))
                if clipped is None:
                    continue
                lines.append(f"{int(cls)} " + " ".join(f"{v:.6f}" for v in clipped))
            lbl_path.parent.mkdir(parents=True, exist_ok=True)
            content = "\n".join(lines)
            lbl_path.write_text(content + "\n" if content else "")
            backup_path = self.backup_dir / f"{Path(name).stem}.txt"
            self._json({
                "ok": True, "modified": boxes_modified(lbl_path, backup_path),
                "original": original_boxes(backup_path),
            })
        elif self.path.startswith("/api/restore-boxes/"):
            name = unquote(self.path[len("/api/restore-boxes/"):])
            lbl_path = self.dataset_dir / "labels" / f"{Path(name).stem}.txt"
            backup_path = self.backup_dir / f"{Path(name).stem}.txt"
            restored = False
            if backup_path.exists():
                lbl_path.parent.mkdir(parents=True, exist_ok=True)
                lbl_path.write_text(backup_path.read_text())
                restored = True
            self._json({
                "ok": True, "restored": restored, "boxes": read_boxes_file(lbl_path),
                "modified": boxes_modified(lbl_path, backup_path),
                "original": original_boxes(backup_path),
            })
        else:
            self.send_error(404)

    def log_message(self, format: str, *args) -> None:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "dataset_dir", type=Path,
        help="Cartella del dataset YOLO, con sottocartelle images/ e labels/",
    )
    parser.add_argument("--port", type=int, default=config.REVIEW_APP_PORT)
    parser.add_argument(
        "--decisions-file", type=Path, default=None,
        help="File JSON delle decisioni (default: <dataset_dir>/review_decisions.json)",
    )
    parser.add_argument(
        "--backup-dir", type=Path, default=None,
        help="Cartella di backup dei label originali (default: <dataset_dir>/review_label_backups)",
    )
    parser.add_argument(
        "--flagged-path", type=Path, default=config.FLAGGED_COCO_OVERLAP_PATH,
        help="File con le detection scartate per overlap sosia (annotate_coco_classes.py), segnalate "
             f"a video quando l'immagine corrente vi compare (default: {config.FLAGGED_COCO_OVERLAP_PATH}; "
             "ignorato in silenzio se il file non esiste, es. su dataset diversi da union_reviewed_coco)",
    )
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    if not (dataset_dir / "images").is_dir() or not (dataset_dir / "labels").is_dir():
        parser.error(f"{dataset_dir} deve contenere le sottocartelle images/ e labels/")

    Handler.dataset_dir = dataset_dir
    Handler.decisions_path = (args.decisions_file or dataset_dir / "review_decisions.json").resolve()
    Handler.backup_dir = (args.backup_dir or dataset_dir / "review_label_backups").resolve()
    Handler.phash_cache_path = dataset_dir / "review_phash_cache.json"
    Handler.flagged_path = args.flagged_path.resolve()

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    num_images = sum(1 for _ in (dataset_dir / "images").iterdir())
    num_flagged = len(load_flagged(Handler.flagged_path))
    print(f"Dataset: {dataset_dir}")
    print(f"Immagini trovate: {num_images}")
    print(f"Decisioni: {Handler.decisions_path}")
    print(f"Backup label: {Handler.backup_dir}")
    if num_flagged:
        print(f"Immagini con detection scartate per overlap sosia segnalate a video: {num_flagged} "
              f"(da {Handler.flagged_path})")
    print(f"Apri http://localhost:{args.port} nel browser (Ctrl+C per fermare)")
    server.serve_forever()


if __name__ == "__main__":
    main()
