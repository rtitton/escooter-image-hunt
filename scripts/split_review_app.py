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

  - "Coppie sospette": per ogni immagine di valid, la sua più vicina in
    train per pHash (distanza di Hamming), ordinate dalla più sospetta.
    Bottone per spostare l'una o l'altra, o per ignorare la coppia (non
    è un duplicato) così non ricompare.
  - "Sfoglia": le due cartelle affiancate, poche miniature alla volta
    (default 4 per lato, configurabile), filtrabili per dataset sorgente
    (prefisso "<dataset_id>__" nel nome file) — per una scorsa manuale
    mirata, non esaustiva, quando si vuole ribilanciare a occhio.

Non disegna le bbox sulle miniature (a differenza di review_app.py): qui
l'obiettivo è decidere la collocazione train/valid, non correggere le
annotazioni. Sotto ogni miniatura c'è comunque il conteggio delle bbox
escooter.

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


def n_escooter_boxes(split_dir: Path, name: str) -> int:
    lbl_path = split_dir / "labels" / f"{Path(name).stem}.txt"
    return sum(1 for cls, *_ in read_boxes_file(lbl_path) if cls == ESCOOTER_CLASS_ID)


def move_image(dataset_dir: Path, name: str, from_split: str) -> None:
    to_split = other_split(from_split)
    src_dir, dst_dir = dataset_dir / from_split, dataset_dir / to_split
    dst_img = dst_dir / "images" / name
    if dst_img.exists():
        raise FileExistsError(f"{name} è già presente in {to_split}")
    (dst_dir / "images").mkdir(parents=True, exist_ok=True)
    (dst_dir / "labels").mkdir(parents=True, exist_ok=True)
    shutil.move(str(src_dir / "images" / name), str(dst_img))
    lbl_src = src_dir / "labels" / f"{Path(name).stem}.txt"
    lbl_dst = dst_dir / "labels" / f"{Path(name).stem}.txt"
    if lbl_src.exists():
        shutil.move(str(lbl_src), str(lbl_dst))
    else:
        lbl_dst.write_text("")


def hex_to_int(h: str) -> int:
    return int(h, 16)


def nearest_cross_split_pairs(train_phash: dict, valid_phash: dict, dismissed: set) -> list:
    """Per ogni immagine di valid, la più vicina in train per distanza di
    Hamming del pHash, ordinate dalla più sospetta (distanza minima). Le
    coppie in `dismissed` (chiave "valid|train") vengono escluse."""
    train_items = [(name, hex_to_int(h)) for name, h in train_phash.items()]
    pairs = []
    for vname, vhash in valid_phash.items():
        vhi = hex_to_int(vhash)
        best_name, best_dist = None, 65  # 65 > distanza massima possibile (64 bit)
        for tname, thi in train_items:
            d = (vhi ^ thi).bit_count()
            if d < best_dist:
                best_dist, best_name = d, tname
        if best_name is not None and f"{vname}|{best_name}" not in dismissed:
            pairs.append({"valid": vname, "train": best_name, "distance": best_dist})
    pairs.sort(key=lambda p: p["distance"])
    return pairs


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
  #tabs { display: flex; gap: 6px; }
  #tabs button { background: #222; color: #ccc; border: 1px solid #444; border-radius: 4px; padding: 6px 14px; cursor: pointer; }
  #tabs button.active { background: #2d5a2d; color: #fff; border-color: #3a7a3a; }
  #tabs button:hover { background: #333; }
  main { padding: 16px; }
  .hidden { display: none !important; }
  select, input[type=number] { background: #222; color: #eee; border: 1px solid #444; border-radius: 4px; padding: 3px 6px; }
  button.small { background: #333; color: #eee; border: 1px solid #555; border-radius: 4px; padding: 3px 10px; cursor: pointer; }
  button.small:hover { background: #444; }
  button.move { background: #2d4d5a; border-color: #3a7a9a; }
  button.move:hover { background: #37627a; }
  button.dismiss { background: #4a3030; border-color: #7a4040; }
  button.dismiss:hover { background: #5a3838; }

  #browse { display: flex; gap: 20px; }
  .col { flex: 1; min-width: 0; }
  .col h2 { font-size: 15px; margin: 0 0 8px; color: #9c9; }
  .col.valid h2 { color: #9bc; }
  .colbar { display: flex; align-items: center; gap: 8px; margin-bottom: 10px; flex-wrap: wrap; }
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 10px; }
  .card { background: #1b1b1b; border: 1px solid #333; border-radius: 6px; padding: 6px; }
  .card img { width: 100%; aspect-ratio: 4/3; object-fit: contain; background: #000; border-radius: 4px; display: block; }
  .card .meta { font-size: 11px; color: #999; margin: 4px 0; word-break: break-all; }
  .card .meta b { color: #ccc; }
  .card button { width: 100%; margin-top: 4px; }
  .pageinfo { color: #999; font-size: 12px; }

  #pairs table { width: 100%; border-collapse: collapse; }
  #pairs th { text-align: left; color: #999; font-weight: normal; font-size: 12px; padding: 6px; border-bottom: 1px solid #333; }
  #pairs td { padding: 8px 6px; border-bottom: 1px solid #262626; vertical-align: middle; }
  #pairs img { width: 110px; height: 82px; object-fit: contain; background: #000; border-radius: 4px; display: block; }
  #pairs .imgcell { display: flex; gap: 8px; align-items: center; }
  #pairs .distance { font-size: 18px; font-weight: bold; }
  #pairs .distance.low { color: #e66; }
  #pairs .distance.mid { color: #ea4; }
  #pairs .distance.high { color: #6a6; }
  #pairs .name { font-size: 11px; color: #999; max-width: 160px; word-break: break-all; }
  #pairs .actions { display: flex; flex-direction: column; gap: 4px; }
  #topbar-pairs { display: flex; align-items: center; gap: 14px; margin-bottom: 12px; flex-wrap: wrap; }
</style>
</head>
<body>
<div id="bar">
  <div class="counts">
    <span>train: <b id="c-train">-</b></span>
    <span>valid: <b id="c-valid">-</b></span>
    <span>spostamenti in questa sessione: <b id="c-moves">0</b></span>
  </div>
  <div id="tabs">
    <button id="tab-pairs" class="active">Coppie sospette</button>
    <button id="tab-browse">Sfoglia</button>
  </div>
</div>

<main>
  <section id="pairs">
    <div id="topbar-pairs">
      <label>soglia distanza pHash &le; <input type="number" id="pairs-threshold" value="15" min="0" max="64" style="width:55px"></label>
      <button class="small" id="pairs-reload">Ricalcola</button>
      <span class="pageinfo" id="pairs-count"></span>
    </div>
    <table>
      <thead><tr><th>distanza</th><th>valid</th><th>train</th><th>azioni</th></tr></thead>
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
const state = { browseOffset: { train: 0, valid: 0 }, moves: 0 };

async function refreshCounts() {
  const r = await fetch("/api/state"); const s = await r.json();
  document.getElementById("c-train").textContent = s.train_count;
  document.getElementById("c-valid").textContent = s.valid_count;
  for (const split of ["train", "valid"]) {
    const sel = document.getElementById(split + "-dataset");
    const current = sel.value;
    sel.innerHTML = '<option value="">tutti i dataset</option>' +
      s.datasets[split].map(d => `<option value="${d}">${d}</option>`).join("");
    if (s.datasets[split].includes(current)) sel.value = current;
  }
}

function distClass(d) { return d <= 12 ? "low" : d <= 20 ? "mid" : "high"; }

async function loadPairs() {
  const th = document.getElementById("pairs-threshold").value;
  const r = await fetch(`/api/pairs?threshold=${th}`);
  const data = await r.json();
  document.getElementById("pairs-count").textContent = `${data.pairs.length} coppie sotto soglia`;
  const body = document.getElementById("pairs-body");
  body.innerHTML = data.pairs.map(p => `
    <tr>
      <td><span class="distance ${distClass(p.distance)}">${p.distance}</span></td>
      <td><div class="imgcell">
        <img src="/api/image/valid/${encodeURIComponent(p.valid)}">
        <div class="name">${p.valid}<br>${p.valid_boxes} bbox escooter</div>
      </div></td>
      <td><div class="imgcell">
        <img src="/api/image/train/${encodeURIComponent(p.train)}">
        <div class="name">${p.train}<br>${p.train_boxes} bbox escooter</div>
      </div></td>
      <td><div class="actions">
        <button class="small move" onclick="movePair('${esc(p.valid)}','valid')">sposta valid &rarr; train</button>
        <button class="small move" onclick="movePair('${esc(p.train)}','train')">sposta train &rarr; valid</button>
        <button class="small dismiss" onclick="dismissPair('${esc(p.valid)}','${esc(p.train)}')">non è un duplicato</button>
      </div></td>
    </tr>`).join("");
}
function esc(s) { return s.replace(/'/g, "\\\\'"); }

async function movePair(name, fromSplit) {
  await fetch("/api/move", { method: "POST", body: JSON.stringify({ name, from_split: fromSplit }) });
  state.moves++; document.getElementById("c-moves").textContent = state.moves;
  await refreshCounts(); await loadPairs(); await loadBrowse("train"); await loadBrowse("valid");
}
async function dismissPair(vname, tname) {
  await fetch("/api/dismiss-pair", { method: "POST", body: JSON.stringify({ valid: vname, train: tname }) });
  await loadPairs();
}

async function loadBrowse(split) {
  const ds = document.getElementById(split + "-dataset").value;
  const limit = parseInt(document.getElementById(split + "-pagesize").value) || 4;
  const offset = state.browseOffset[split];
  const r = await fetch(`/api/list?split=${split}&dataset=${encodeURIComponent(ds)}&offset=${offset}&limit=${limit}`);
  const data = await r.json();
  document.getElementById(split + "-pageinfo").textContent =
    data.total ? `${offset + 1}-${Math.min(offset + limit, data.total)} di ${data.total}` : "0 immagini";
  const grid = document.getElementById(split + "-grid");
  const arrow = split === "train" ? "&rarr; valid" : "&larr; train";
  grid.innerHTML = data.items.map(it => `
    <div class="card">
      <img src="/api/image/${split}/${encodeURIComponent(it.name)}">
      <div class="meta"><b>${it.dataset}</b><br>${it.name}<br>${it.n_boxes} bbox escooter</div>
      <button class="small move" onclick="moveBrowse('${esc(it.name)}','${split}')">${arrow}</button>
    </div>`).join("");
}
async function moveBrowse(name, fromSplit) {
  await fetch("/api/move", { method: "POST", body: JSON.stringify({ name, from_split: fromSplit }) });
  state.moves++; document.getElementById("c-moves").textContent = state.moves;
  await refreshCounts(); await loadBrowse("train"); await loadBrowse("valid");
}

for (const split of ["train", "valid"]) {
  document.getElementById(split + "-prev").onclick = () => {
    const limit = parseInt(document.getElementById(split + "-pagesize").value) || 4;
    state.browseOffset[split] = Math.max(0, state.browseOffset[split] - limit);
    loadBrowse(split);
  };
  document.getElementById(split + "-next").onclick = () => {
    const limit = parseInt(document.getElementById(split + "-pagesize").value) || 4;
    state.browseOffset[split] += limit;
    loadBrowse(split);
  };
  document.getElementById(split + "-dataset").addEventListener("change", () => { state.browseOffset[split] = 0; loadBrowse(split); });
  document.getElementById(split + "-pagesize").addEventListener("change", () => { state.browseOffset[split] = 0; loadBrowse(split); });
}
document.getElementById("pairs-reload").onclick = loadPairs;

document.getElementById("tab-pairs").onclick = () => {
  document.getElementById("tab-pairs").classList.add("active");
  document.getElementById("tab-browse").classList.remove("active");
  document.getElementById("pairs").classList.remove("hidden");
  document.getElementById("browse").classList.add("hidden");
};
document.getElementById("tab-browse").onclick = () => {
  document.getElementById("tab-browse").classList.add("active");
  document.getElementById("tab-pairs").classList.remove("active");
  document.getElementById("browse").classList.remove("hidden");
  document.getElementById("pairs").classList.add("hidden");
  loadBrowse("train"); loadBrowse("valid");
};

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
            body = PAGE.encode("utf-8")
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
            items = [{"name": n, "dataset": dataset_of(n), "n_boxes": n_escooter_boxes(split_dir, n)} for n in page]
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

        elif parsed.path == "/api/pairs":
            threshold = float(qs.get("threshold", ["20"])[0])
            dismissed = set(load_decisions(self.dismissed_path).keys())
            train_phash, valid_phash = self._phashes("train"), self._phashes("valid")
            all_pairs = nearest_cross_split_pairs(train_phash, valid_phash, dismissed)
            pairs = [p for p in all_pairs if p["distance"] <= threshold]
            for p in pairs:
                p["valid_boxes"] = n_escooter_boxes(self.dataset_dir / "valid", p["valid"])
                p["train_boxes"] = n_escooter_boxes(self.dataset_dir / "train", p["train"])
            self._json({"pairs": pairs})

        else:
            self.send_error(404)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}

        if self.path == "/api/move":
            name, from_split = body["name"], body["from_split"]
            if from_split not in SPLITS:
                self._json({"ok": False, "error": "from_split non valido"}, 400)
                return
            try:
                move_image(self.dataset_dir, name, from_split)
            except (FileNotFoundError, FileExistsError) as e:
                self._json({"ok": False, "error": str(e)}, 400)
                return
            log = load_decisions(self.move_log_path)
            log.setdefault("moves", []).append(
                {"name": name, "from": from_split, "to": other_split(from_split), "ts": time.time()}
            )
            save_decisions(self.move_log_path, log)
            self._json({"ok": True, "to_split": other_split(from_split)})

        elif self.path == "/api/dismiss-pair":
            key = f"{body['valid']}|{body['train']}"
            dismissed = load_decisions(self.dismissed_path)
            dismissed[key] = True
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
