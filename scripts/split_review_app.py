#!/usr/bin/env python3
"""Applicazione locale per revisionare a mano lo split train/valid di un
dataset YOLO già diviso da split_dataset.py (sottocartelle train/images,
train/labels, valid/images, valid/labels), e spostare immagini da uno
split all'altro con un clic — ogni spostamento sposta subito anche il
file label corrispondente e viene applicato sul dataset reale (non c'è
salvataggio separato: per tornare indietro basta spostare l'immagine di
nuovo dal lato opposto).

Nata per un problema specifico: con centinaia/migliaia di immagini, una
revisione manuale completa non è pensabile, e lo scopo principale non è
"guardarle tutte" ma trovare i casi peggiori di somiglianza fra i due
split (le foto sono spesso duplicate o quasi-duplicate fra dataset
Roboflow diversi, v. dedupe_phash.py — un quasi-duplicato finito per caso
uno in train e uno in valid gonfia le metriche di validazione). Due modalità:

  - "Coppie sospette": per ogni immagine di valid, tutte le immagini di
    train entro la soglia di distanza pHash (Hamming), raggruppate in una
    riga e ordinate dalla più sospetta: un cluster di copie si risolve in
    una passata. Bottoni per spostare o escludere ciascuna, o per ignorare
    la coppia (non è un duplicato) così non ricompare.
  - "Sfoglia": le due cartelle affiancate, poche miniature alla volta
    (default 4 per lato, configurabile), filtrabili per dataset sorgente
    (prefisso "<dataset_id>__" nel nome file) — per una scorsa manuale
    mirata, non esaustiva, quando si vuole ribilanciare a occhio.

Su ogni miniatura sono disegnate le bbox (escooter in rosso, person in
verde, altre classi COCO in blu).

Azioni per immagine (nelle coppie sospette e in Sfoglia):
  - sposta: passa dall'altro split (immagine + label);
  - escludi: sposta immagine + label in <dataset_dir>/excluded/<split>/,
    fuori dal dataset ma senza cancellarle. Sui veri duplicati è la mossa più
    pulita (eliminando la copia in train non si crea nessun duplicato
    interno né a train né a valid);
  - "Annulla ultima azione": ripristina l'ultimo spostamento o l'ultima
    esclusione registrata nel log.

Avvio:
    python3 scripts/split_review_app.py <dataset_dir> [--port 8767]

<dataset_dir> deve contenere le sottocartelle train/{images,labels} e
valid/{images,labels} (es. data/processed/union_reviewed_coco_split).
Poi apri http://localhost:<port> nel browser.
"""
import argparse
import json
import mimetypes
import shutil
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import config
from review_app import compute_phashes, load_decisions, save_decisions, read_boxes_file
from split_dataset import dataset_of

SPLITS = ("train", "valid")
ESCOOTER_CLASS_ID = config.ESCOOTER_CLASS_ID


def other_split(split: str) -> str:
    return "valid" if split == "train" else "train"


def list_images(split_dir: Path) -> list:
    return sorted(p.name for p in (split_dir / "images").iterdir()) if (split_dir / "images").is_dir() else []


def boxes_of(split_dir: Path, name: str) -> list:
    """Tutte le bbox (classe, xc, yc, w, h) del label di un'immagine, per disegnarle."""
    return read_boxes_file(split_dir / "labels" / f"{Path(name).stem}.txt")


def move_files(src_dir: Path, dst_dir: Path, name: str) -> None:
    """Sposta immagine e label (se manca, ne crea uno vuoto a destinazione)
    da src_dir a dst_dir, entrambe cartelle con sottocartelle images/ e
    labels/. Rifiuta di sovrascrivere un'immagine già presente."""
    dst_img = dst_dir / "images" / name
    if dst_img.exists():
        raise FileExistsError(f"{name} è già presente in {dst_dir}")
    (dst_dir / "images").mkdir(parents=True, exist_ok=True)
    (dst_dir / "labels").mkdir(parents=True, exist_ok=True)
    shutil.move(str(src_dir / "images" / name), str(dst_img))
    lbl_src = src_dir / "labels" / f"{Path(name).stem}.txt"
    lbl_dst = dst_dir / "labels" / f"{Path(name).stem}.txt"
    if lbl_src.exists():
        shutil.move(str(lbl_src), str(lbl_dst))
    else:
        lbl_dst.write_text("")


def excluded_dir(dataset_dir: Path, split: str) -> Path:
    return dataset_dir / "excluded" / split


def count_excluded(dataset_dir: Path) -> int:
    return sum(len(list_images(excluded_dir(dataset_dir, s))) for s in SPLITS)


def hex_to_int(h: str) -> int:
    return int(h, 16)


MAX_MATCHES_PER_GROUP = 12  # oltre, la riga mostra "+N altre" per non allungare la pagina


def cross_split_groups(train_phash: dict, valid_phash: dict, dismissed: set, threshold: float) -> list:
    """Per ogni immagine di valid, le immagini di train con distanza di
    Hamming del pHash <= threshold (le più vicine per prime, al massimo
    MAX_MATCHES_PER_GROUP; il resto in "extra"). Le coppie in `dismissed`
    (chiave "valid|train") vengono saltate. I gruppi sono ordinati dal più
    sospetto (distanza minima) e omessi se non hanno nessun match."""
    train_items = [(name, hex_to_int(h)) for name, h in train_phash.items()]
    groups = []
    for vname, vhash in valid_phash.items():
        vhi = hex_to_int(vhash)
        matches = []
        for tname, thi in train_items:
            d = (vhi ^ thi).bit_count()
            if d <= threshold and f"{vname}|{tname}" not in dismissed:
                matches.append({"train": tname, "distance": d})
        if matches:
            matches.sort(key=lambda m: m["distance"])
            groups.append({
                "valid": vname, "matches": matches[:MAX_MATCHES_PER_GROUP],
                "extra": max(0, len(matches) - MAX_MATCHES_PER_GROUP),
            })
    groups.sort(key=lambda g: g["matches"][0]["distance"])
    return groups


PAGE = """<!doctype html>
<html lang="it">
<head>
<meta charset="utf-8">
<title>Revisione split train/valid</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin: 0; background: #111; color: #eee; font-family: system-ui, sans-serif; font-size: 14px; }
  #bar { display: flex; justify-content: space-between; align-items: center; padding: 10px 16px; background: #1b1b1b; border-bottom: 1px solid #333; flex-wrap: wrap; gap: 12px; }
  #bar .counts span { margin-right: 16px; color: #ccc; }
  #tabs { display: flex; gap: 6px; align-items: center; }
  #tabs button { background: #222; color: #ccc; border: 1px solid #444; border-radius: 4px; padding: 6px 14px; cursor: pointer; }
  #tabs button.active { background: #2d5a2d; color: #fff; border-color: #3a7a3a; }
  #tabs button:hover { background: #333; }
  #tabs button#undo { margin-left: 16px; background: #4a3a20; border-color: #7a6030; color: #fff; }
  #tabs button#undo:hover { background: #5a4828; }
  main { padding: 16px; }
  .hidden { display: none !important; }
  select, input[type=number] { background: #222; color: #eee; border: 1px solid #444; border-radius: 4px; padding: 3px 6px; }
  button.small { background: #333; color: #eee; border: 1px solid #555; border-radius: 4px; padding: 3px 10px; cursor: pointer; }
  button.small:hover { background: #444; }
  button.move { background: #2d4d5a; border-color: #3a7a9a; }
  button.move:hover { background: #37627a; }
  button.exclude { background: #5a2d2d; border-color: #9a3a3a; }
  button.exclude:hover { background: #7a3737; }
  button.dismiss { background: #333; border-color: #666; }

  .imgwrap { position: relative; background: #000; border-radius: 4px; overflow: hidden; }
  .imgwrap img { position: absolute; left: 0; top: 0; width: 100%; height: 100%; object-fit: contain; }
  .imgwrap canvas { position: absolute; left: 0; top: 0; width: 100%; height: 100%; pointer-events: none; }

  #browse { display: flex; gap: 20px; }
  .col { flex: 1; min-width: 0; }
  .col h2 { font-size: 15px; margin: 0 0 8px; color: #9c9; }
  .col.valid h2 { color: #9bc; }
  .colbar { display: flex; align-items: center; gap: 8px; margin-bottom: 10px; flex-wrap: wrap; }
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 12px; }
  .card { background: #1b1b1b; border: 1px solid #333; border-radius: 6px; padding: 6px; }
  .card .imgwrap { width: 100%; aspect-ratio: 4/3; }
  .card .meta { font-size: 11px; color: #999; margin: 4px 0; word-break: break-all; }
  .card .meta b { color: #ccc; }
  .card .btns { display: flex; gap: 4px; margin-top: 4px; }
  .card .btns button { flex: 1; }
  .pageinfo { color: #999; font-size: 12px; }

  #pairs table { width: 100%; border-collapse: collapse; }
  #pairs th { text-align: left; color: #999; font-weight: normal; font-size: 12px; padding: 6px; border-bottom: 1px solid #333; }
  #pairs td { padding: 12px 6px; border-bottom: 1px solid #333; vertical-align: top; }
  #pairs .imgwrap { width: 220px; height: 165px; }
  #pairs .name { font-size: 11px; color: #999; width: 220px; word-break: break-all; margin: 4px 0; }
  #pairs .vcell { width: 236px; }
  #pairs .matches { display: flex; flex-wrap: wrap; gap: 14px; }
  #pairs .match { width: 220px; }
  #pairs .match .dist { font-size: 16px; font-weight: bold; margin-bottom: 3px; }
  #pairs .dist.low { color: #e66; }
  #pairs .dist.mid { color: #ea4; }
  #pairs .dist.high { color: #6a6; }
  #pairs .actions { display: flex; flex-direction: column; gap: 4px; width: 220px; }
  #pairs .extra { color: #999; align-self: center; }
  #topbar-pairs { display: flex; align-items: center; gap: 14px; margin-bottom: 12px; flex-wrap: wrap; }
  #legend { color: #999; font-size: 12px; }
  #legend i { display: inline-block; width: 10px; height: 10px; margin: 0 3px 0 10px; vertical-align: middle; }
</style>
</head>
<body>
<div id="bar">
  <div class="counts">
    <span>train: <b id="c-train">-</b></span>
    <span>valid: <b id="c-valid">-</b></span>
    <span>esclusi: <b id="c-excluded">-</b></span>
    <span>azioni in questa sessione: <b id="c-moves">0</b></span>
  </div>
  <div id="tabs">
    <button id="tab-pairs" class="active">Coppie sospette</button>
    <button id="tab-browse">Sfoglia</button>
    <button id="undo" title="Ripristina l'ultimo spostamento o l'ultima esclusione">Annulla ultima azione</button>
  </div>
</div>

<main>
  <section id="pairs">
    <div id="topbar-pairs">
      <label>soglia distanza pHash &le; <input type="number" id="pairs-threshold" value="15" min="0" max="64" style="width:55px"></label>
      <button class="small" id="pairs-reload">Ricalcola</button>
      <span class="pageinfo" id="pairs-count"></span>
      <span id="legend"><i style="background:#ff3b3b"></i>escooter<i style="background:#3fd23f"></i>person<i style="background:#3b9bff"></i>altre classi</span>
    </div>
    <table>
      <thead><tr><th>valid</th><th>candidati duplicati in train (dal più vicino)</th></tr></thead>
      <tbody id="pairs-body"></tbody>
    </table>
  </section>

  <section id="browse" class="hidden">
    <div class="col train">
      <h2>TRAIN</h2>
      <div class="colbar">
        <select id="train-dataset"></select>
        <label>per pagina <input type="number" id="train-pagesize" value="4" min="1" max="50" style="width:50px"></label>
        <button class="small" id="train-prev">&larr;</button>
        <span class="pageinfo" id="train-pageinfo"></span>
        <button class="small" id="train-next">&rarr;</button>
      </div>
      <div class="grid" id="train-grid"></div>
    </div>
    <div class="col valid">
      <h2>VALID</h2>
      <div class="colbar">
        <select id="valid-dataset"></select>
        <label>per pagina <input type="number" id="valid-pagesize" value="4" min="1" max="50" style="width:50px"></label>
        <button class="small" id="valid-prev">&larr;</button>
        <span class="pageinfo" id="valid-pageinfo"></span>
        <button class="small" id="valid-next">&rarr;</button>
      </div>
      <div class="grid" id="valid-grid"></div>
    </div>
  </section>
</main>

<script>
const ESCOOTER_CLASS = __ESCOOTER_CLASS_ID__;
const state = { browseOffset: { train: 0, valid: 0 }, actions: 0 };

function ea(s) { return s.replace(/&/g, "&amp;").replace(/"/g, "&quot;").replace(/</g, "&lt;"); }
function boxColor(cls) { return cls === ESCOOTER_CLASS ? "#ff3b3b" : cls === 0 ? "#3fd23f" : "#3b9bff"; }

function drawBoxes(img) {
  const wrap = img.parentElement, cv = wrap.querySelector("canvas");
  const cw = wrap.clientWidth, ch = wrap.clientHeight;
  cv.width = cw; cv.height = ch;
  const nw = img.naturalWidth, nh = img.naturalHeight;
  if (!nw || !nh) return;
  const sc = Math.min(cw / nw, ch / nh), dw = nw * sc, dh = nh * sc;
  const ox = (cw - dw) / 2, oy = (ch - dh) / 2;
  const boxes = JSON.parse(decodeURIComponent(img.dataset.boxes));
  boxes.sort((a, b) => (a[0] === ESCOOTER_CLASS) - (b[0] === ESCOOTER_CLASS));  // escooter per ultime, sopra le altre
  const ctx = cv.getContext("2d");
  for (const [cls, xc, yc, w, h] of boxes) {
    ctx.strokeStyle = boxColor(cls);
    ctx.lineWidth = cls === ESCOOTER_CLASS ? 3 : 1.5;
    ctx.strokeRect(ox + (xc - w / 2) * dw, oy + (yc - h / 2) * dh, w * dw, h * dh);
  }
}
window.addEventListener("resize", () => document.querySelectorAll("img[data-boxes]").forEach(drawBoxes));

function thumb(split, name, boxes) {
  return `<div class="imgwrap"><img src="/api/image/${split}/${encodeURIComponent(name)}"
    data-boxes="${encodeURIComponent(JSON.stringify(boxes))}" onload="drawBoxes(this)"><canvas></canvas></div>`;
}

async function refreshCounts() {
  const s = await (await fetch("/api/state")).json();
  document.getElementById("c-train").textContent = s.train_count;
  document.getElementById("c-valid").textContent = s.valid_count;
  document.getElementById("c-excluded").textContent = s.excluded_count;
  for (const split of ["train", "valid"]) {
    const sel = document.getElementById(split + "-dataset");
    const current = sel.value;
    sel.innerHTML = '<option value="">tutti i dataset</option>' +
      s.datasets[split].map(d => `<option value="${ea(d)}">${d}</option>`).join("");
    if (s.datasets[split].includes(current)) sel.value = current;
  }
}

function distClass(d) { return d <= 12 ? "low" : d <= 20 ? "mid" : "high"; }
function nEsc(boxes) { return boxes.filter(b => b[0] === ESCOOTER_CLASS).length; }

async function loadPairs() {
  const th = document.getElementById("pairs-threshold").value;
  const data = await (await fetch(`/api/groups?threshold=${th}`)).json();
  const nPairs = data.groups.reduce((n, g) => n + g.matches.length + g.extra, 0);
  document.getElementById("pairs-count").textContent = `${data.groups.length} immagini di valid con candidati, ${nPairs} coppie`;
  document.getElementById("pairs-body").innerHTML = data.groups.map(g => `
    <tr>
      <td class="vcell">
        ${thumb("valid", g.valid, g.valid_boxes)}
        <div class="name">${g.valid}<br>${nEsc(g.valid_boxes)} bbox escooter</div>
        <div class="actions">
          <button class="small exclude" data-act="exclude" data-split="valid" data-name="${ea(g.valid)}">escludi da valid</button>
          <button class="small move" data-act="move" data-split="valid" data-name="${ea(g.valid)}">sposta valid &rarr; train</button>
          <button class="small dismiss" data-act="dismiss" data-valid="${ea(g.valid)}"
            data-trains="${ea(JSON.stringify(g.matches.map(m => m.train)))}">nessuno è un duplicato</button>
        </div>
      </td>
      <td><div class="matches">
        ${g.matches.map(m => `
          <div class="match">
            <div class="dist ${distClass(m.distance)}">distanza ${m.distance}</div>
            ${thumb("train", m.train, m.boxes)}
            <div class="name">${m.train}<br>${nEsc(m.boxes)} bbox escooter</div>
            <div class="actions">
              <button class="small exclude" data-act="exclude" data-split="train" data-name="${ea(m.train)}">escludi da train</button>
              <button class="small move" data-act="move" data-split="train" data-name="${ea(m.train)}">sposta train &rarr; valid</button>
              <button class="small dismiss" data-act="dismiss" data-valid="${ea(g.valid)}" data-train="${ea(m.train)}">non è un duplicato</button>
            </div>
          </div>`).join("")}
        ${g.extra ? `<div class="extra">+${g.extra} altre sotto soglia</div>` : ""}
      </div></td>
    </tr>`).join("");
}

async function loadBrowse(split) {
  const ds = document.getElementById(split + "-dataset").value;
  const limit = parseInt(document.getElementById(split + "-pagesize").value) || 4;
  const offset = state.browseOffset[split];
  const data = await (await fetch(`/api/list?split=${split}&dataset=${encodeURIComponent(ds)}&offset=${offset}&limit=${limit}`)).json();
  document.getElementById(split + "-pageinfo").textContent =
    data.total ? `${offset + 1}-${Math.min(offset + limit, data.total)} di ${data.total}` : "0 immagini";
  const arrow = split === "train" ? "&rarr; valid" : "&larr; train";
  document.getElementById(split + "-grid").innerHTML = data.items.map(it => `
    <div class="card">${thumb(split, it.name, it.boxes)}
      <div class="meta"><b>${it.dataset}</b><br>${it.name}<br>${it.boxes.filter(b => b[0] === ESCOOTER_CLASS).length} bbox escooter</div>
      <div class="btns">
        <button class="small move" data-act="move" data-split="${split}" data-name="${ea(it.name)}">sposta ${arrow}</button>
        <button class="small exclude" data-act="exclude" data-split="${split}" data-name="${ea(it.name)}">escludi</button>
      </div>
    </div>`).join("");
}

async function refreshAll() {
  await refreshCounts();
  await loadPairs();
  await loadBrowse("train");
  await loadBrowse("valid");
}

async function post(path, body) {
  const r = await fetch(path, { method: "POST", body: JSON.stringify(body || {}) });
  return r.json();
}

document.addEventListener("click", async e => {
  const b = e.target.closest("button[data-act]");
  if (!b) return;
  const act = b.dataset.act;
  if (act === "dismiss") {
    await post("/api/dismiss-pair", {
      valid: b.dataset.valid,
      ...(b.dataset.trains ? { trains: JSON.parse(b.dataset.trains) } : { train: b.dataset.train }),
    });
    await loadPairs();
    return;
  }
  const res = await post("/api/" + act, { name: b.dataset.name, from_split: b.dataset.split });
  if (!res.ok) { alert(res.error || "operazione fallita"); return; }
  state.actions++; document.getElementById("c-moves").textContent = state.actions;
  await refreshAll();
});

document.getElementById("undo").onclick = async () => {
  const res = await post("/api/undo");
  if (!res.ok) { alert(res.error || "niente da annullare"); return; }
  state.actions = Math.max(0, state.actions - 1); document.getElementById("c-moves").textContent = state.actions;
  await refreshAll();
};

for (const split of ["train", "valid"]) {
  const pageSize = () => parseInt(document.getElementById(split + "-pagesize").value) || 4;
  document.getElementById(split + "-prev").onclick = () => {
    state.browseOffset[split] = Math.max(0, state.browseOffset[split] - pageSize());
    loadBrowse(split);
  };
  document.getElementById(split + "-next").onclick = () => {
    state.browseOffset[split] += pageSize();
    loadBrowse(split);
  };
  for (const id of ["-dataset", "-pagesize"])
    document.getElementById(split + id).addEventListener("change", () => { state.browseOffset[split] = 0; loadBrowse(split); });
}
document.getElementById("pairs-reload").onclick = loadPairs;

function showTab(name) {
  document.getElementById("tab-pairs").classList.toggle("active", name === "pairs");
  document.getElementById("tab-browse").classList.toggle("active", name === "browse");
  document.getElementById("pairs").classList.toggle("hidden", name !== "pairs");
  document.getElementById("browse").classList.toggle("hidden", name !== "browse");
  if (name === "browse") { loadBrowse("train"); loadBrowse("valid"); } else loadPairs();  // ridisegna le bbox: a tab nascosto i canvas hanno dimensione 0
}
document.getElementById("tab-pairs").onclick = () => showTab("pairs");
document.getElementById("tab-browse").onclick = () => showTab("browse");

refreshCounts().then(loadPairs);
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    # Impostati su main() prima dell'avvio del server.
    dataset_dir: Path
    phash_cache_paths: dict  # {"train": Path, "valid": Path}
    dismissed_path: Path
    move_log_path: Path

    def _json(self, obj, status: int = 200) -> None:
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _phashes(self, split: str) -> dict:
        return compute_phashes(self.dataset_dir / split, self.phash_cache_paths[split])

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)

        if parsed.path in ("/", "/index.html"):
            body = PAGE.replace("__ESCOOTER_CLASS_ID__", str(ESCOOTER_CLASS_ID)).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        elif parsed.path == "/api/state":
            train_names = list_images(self.dataset_dir / "train")
            valid_names = list_images(self.dataset_dir / "valid")
            self._json({
                "train_count": len(train_names), "valid_count": len(valid_names),
                "excluded_count": count_excluded(self.dataset_dir),
                "datasets": {
                    "train": sorted(set(dataset_of(n) for n in train_names)),
                    "valid": sorted(set(dataset_of(n) for n in valid_names)),
                },
            })

        elif parsed.path == "/api/list":
            split = qs.get("split", [""])[0]
            if split not in SPLITS:
                self.send_error(400, "split deve essere train o valid")
                return
            ds_filter = qs.get("dataset", [""])[0]
            offset = int(qs.get("offset", ["0"])[0])
            limit = int(qs.get("limit", ["4"])[0])
            split_dir = self.dataset_dir / split
            names = list_images(split_dir)
            if ds_filter:
                names = [n for n in names if dataset_of(n) == ds_filter]
            page = names[offset:offset + limit]
            items = [{"name": n, "dataset": dataset_of(n), "boxes": boxes_of(split_dir, n)} for n in page]
            self._json({"items": items, "total": len(names)})

        elif parsed.path.startswith("/api/image/"):
            rest = parsed.path[len("/api/image/"):]
            split, _, name = rest.partition("/")
            name = unquote(name)
            if split not in SPLITS:
                self.send_error(404)
                return
            path = self.dataset_dir / split / "images" / name
            if not path.exists():
                self.send_error(404)
                return
            data = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type(name)[0] or "application/octet-stream")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        elif parsed.path == "/api/groups":
            threshold = float(qs.get("threshold", ["15"])[0])
            dismissed = set(load_decisions(self.dismissed_path).keys())
            groups = cross_split_groups(self._phashes("train"), self._phashes("valid"), dismissed, threshold)
            for g in groups:
                g["valid_boxes"] = boxes_of(self.dataset_dir / "valid", g["valid"])
                for m in g["matches"]:
                    m["boxes"] = boxes_of(self.dataset_dir / "train", m["train"])
            self._json({"groups": groups})

        else:
            self.send_error(404)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}

        if self.path in ("/api/move", "/api/exclude"):
            name, from_split = body["name"], body["from_split"]
            if from_split not in SPLITS:
                self._json({"ok": False, "error": "from_split non valido"}, 400)
                return
            action = self.path.rsplit("/", 1)[1]
            dst = self.dataset_dir / other_split(from_split) if action == "move" else excluded_dir(self.dataset_dir, from_split)
            try:
                move_files(self.dataset_dir / from_split, dst, name)
            except (FileNotFoundError, FileExistsError) as e:
                self._json({"ok": False, "error": str(e)}, 400)
                return
            log = load_decisions(self.move_log_path)
            log.setdefault("moves", []).append({"action": action, "name": name, "from": from_split, "ts": time.time()})
            save_decisions(self.move_log_path, log)
            self._json({"ok": True})

        elif self.path == "/api/undo":
            log = load_decisions(self.move_log_path)
            if not log.get("moves"):
                self._json({"ok": False, "error": "niente da annullare"})
                return
            last = log["moves"][-1]
            src = self.dataset_dir / other_split(last["from"]) if last.get("action", "move") == "move"                 else excluded_dir(self.dataset_dir, last["from"])
            try:
                move_files(src, self.dataset_dir / last["from"], last["name"])
            except (FileNotFoundError, FileExistsError) as e:
                self._json({"ok": False, "error": str(e)}, 400)
                return
            log["moves"].pop()
            save_decisions(self.move_log_path, log)
            self._json({"ok": True, "restored": last["name"], "to_split": last["from"]})

        elif self.path == "/api/dismiss-pair":
            dismissed = load_decisions(self.dismissed_path)
            for train_name in body.get("trains") or [body["train"]]:
                dismissed[f"{body['valid']}|{train_name}"] = True
            save_decisions(self.dismissed_path, dismissed)
            self._json({"ok": True})

        else:
            self.send_error(404)

    def log_message(self, format: str, *args) -> None:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset_dir", type=Path,
                         help="Cartella dello split, con sottocartelle train/{images,labels} e valid/{images,labels}")
    parser.add_argument("--port", type=int, default=config.SPLIT_REVIEW_APP_PORT)
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    for split in SPLITS:
        if not (dataset_dir / split / "images").is_dir() or not (dataset_dir / split / "labels").is_dir():
            parser.error(f"{dataset_dir / split} deve contenere le sottocartelle images/ e labels/")

    Handler.dataset_dir = dataset_dir
    Handler.phash_cache_paths = {
        "train": dataset_dir / "split_review_phash_cache_train.json",
        "valid": dataset_dir / "split_review_phash_cache_valid.json",
    }
    Handler.dismissed_path = dataset_dir / "split_review_dismissed.json"
    Handler.move_log_path = dataset_dir / "split_review_moves.json"

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Dataset: {dataset_dir}")
    print(f"train: {len(list_images(dataset_dir / 'train'))} immagini, valid: {len(list_images(dataset_dir / 'valid'))} immagini")
    print(f"Log spostamenti: {Handler.move_log_path}")
    print(f"Apri http://localhost:{args.port} nel browser (Ctrl+C per fermare)")
    server.serve_forever()


if __name__ == "__main__":
    main()
