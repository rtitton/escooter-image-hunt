#!/usr/bin/env python3
"""Applicazione locale per la revisione manuale di un dataset in formato
YOLO, organizzato in due sottocartelle images/ e labels/ (un file .txt per
immagine, righe "classe xc yc w h" normalizzate 0..1).

Mostra un'immagine alla volta con le bbox disegnate (canvas HTML), e
registra la decisione dell'utente con un tasto:
    s = seleziona
    n = scarta
    l = seleziona con riserva
    freccia sinistra/destra = torna indietro / salta avanti senza decidere
    backspace = cancella la decisione sull'immagine corrente

Le bbox sono modificabili direttamente sul canvas:
    trascina un'area vuota per crearne una nuova
    clic su una bbox per selezionarla, trascina per spostarla
    trascina i quadratini della bbox selezionata per ridimensionarla
    x oppure canc = elimina la bbox selezionata
    esc = deseleziona
Ogni modifica viene salvata subito nel file label corrispondente. La
prima volta che un'immagine viene modificata, il suo file label originale
viene copiato nella cartella di backup (--backup-dir) come rete di
sicurezza.

Le decisioni sono scritte subito nel file delle decisioni (--decisions-file)
ad ogni tasto premuto: la sessione si può interrompere e riprendere quando
si vuole, riparte dalla prima immagine ancora senza decisione.

Avvio:
    python3 scripts/review_app.py <dataset_dir> [--port 8765]
        [--decisions-file FILE] [--backup-dir DIR]

<dataset_dir> deve contenere le sottocartelle images/ e labels/. Se non
specificati, --decisions-file e --backup-dir vengono creati dentro
<dataset_dir> (rispettivamente review_decisions.json e
review_label_backups/).

poi apri http://localhost:<port> nel browser.
"""
import argparse
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

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


PAGE = """<!doctype html>
<html lang="it">
<head>
<meta charset="utf-8">
<title>Revisione dataset union</title>
<style>
  :root { color-scheme: dark; }
  body { margin: 0; background: #111; color: #eee; font-family: system-ui, sans-serif; }
  #bar { display: flex; justify-content: space-between; align-items: center; padding: 10px 16px; background: #1b1b1b; border-bottom: 1px solid #333; }
  #bar .counts span { margin-right: 14px; }
  #stage { position: relative; display: flex; align-items: center; justify-content: center; height: calc(100vh - 140px); }
  #stage img { max-width: 96vw; max-height: 100%; display: block; }
  #stage canvas { position: absolute; pointer-events: auto; cursor: crosshair; }
  #name { text-align: center; padding: 6px; font-size: 13px; color: #999; word-break: break-all; }
  #decision { font-weight: bold; }
  .sel { color: #4caf50; }
  .disc { color: #e53935; }
  .res { color: #ffb300; }
  #help, #help2 { text-align: center; padding: 6px; font-size: 13px; color: #888; }
  #help kbd, #help2 kbd { background: #333; border-radius: 3px; padding: 1px 6px; margin: 0 2px; }
  #box-status { font-weight: bold; color: #4caf50; }
  #done { display: none; text-align: center; margin-top: 80px; font-size: 22px; }
</style>
</head>
<body>
<div id="bar">
  <div>Immagine <span id="idx">-</span>/<span id="total">-</span></div>
  <div class="counts">
    <span class="sel">selezionate: <b id="c-select">0</b></span>
    <span class="res">con riserva: <b id="c-reserve">0</b></span>
    <span class="disc">scartate: <b id="c-discard">0</b></span>
    <span>rimanenti: <b id="c-todo">0</b></span>
  </div>
</div>
<div id="stage">
  <img id="img">
  <canvas id="canvas"></canvas>
</div>
<div id="name"></div>
<div id="help">
  <kbd>s</kbd> seleziona &nbsp; <kbd>l</kbd> seleziona con riserva &nbsp; <kbd>n</kbd> scarta &nbsp;
  <kbd>&larr;</kbd>/<kbd>&rarr;</kbd> naviga &nbsp; <kbd>backspace</kbd> cancella decisione
  &nbsp;&nbsp; decisione corrente: <span id="decision">-</span>
</div>
<div id="help2">
  trascina un'area vuota per creare una bbox &nbsp; clic/trascina su una bbox per selezionarla/spostarla &nbsp;
  trascina i quadratini per ridimensionare &nbsp; <kbd>x</kbd>/<kbd>canc</kbd> elimina selezionata &nbsp; <kbd>esc</kbd> deseleziona
  &nbsp;&nbsp; bbox: <span id="box-status">-</span>
</div>
<div id="done">Fine: tutte le immagini sono state revisionate.</div>

<script>
let names = [];
let decisions = {};
let idx = 0;
let renderToken = 0;

const img = document.getElementById("img");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

async function loadState() {
  const r = await fetch("/api/state");
  const data = await r.json();
  names = data.names;
  decisions = data.decisions;
  idx = names.findIndex(n => !(n in decisions));
  if (idx === -1) idx = 0;
  render();
}

function updateCounts() {
  let s=0, l=0, n=0;
  for (const name of names) {
    const d = decisions[name];
    if (d === "select") s++;
    else if (d === "reserve") l++;
    else if (d === "discard") n++;
  }
  document.getElementById("c-select").textContent = s;
  document.getElementById("c-reserve").textContent = l;
  document.getElementById("c-discard").textContent = n;
  document.getElementById("c-todo").textContent = names.length - s - l - n;
  document.getElementById("total").textContent = names.length;
}

const DEFAULT_CLASS = __ESCOOTER_CLASS_ID__;
const HANDLE_SIZE = 10;
const MIN_BOX_PX = 6;
const CURSOR_FOR_HANDLE = {
  nw: "nwse-resize", se: "nwse-resize",
  ne: "nesw-resize", sw: "nesw-resize",
  n: "ns-resize", s: "ns-resize",
  w: "ew-resize", e: "ew-resize",
};

let currentBoxes = []; // [{cls, xc, yc, w, h}, ...] normalizzati 0..1
let selectedIndex = -1;
let drag = null;
let creatingRect = null;

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
  return { x: e.clientX - rect.left, y: e.clientY - rect.top };
}

function resizeCanvas() {
  canvas.width = img.clientWidth;
  canvas.height = img.clientHeight;
  canvas.style.width = img.clientWidth + "px";
  canvas.style.height = img.clientHeight + "px";
}

function drawBoxes(boxes) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  boxes.forEach((b, i) => {
    const r = boxPixelRect(b);
    const selected = i === selectedIndex;
    ctx.strokeStyle = selected ? "#ffd60a" : "#ff2d55";
    ctx.lineWidth = selected ? 3 : 2;
    ctx.strokeRect(r.x, r.y, r.w, r.h);
    if (selected) {
      ctx.fillStyle = "#ffd60a";
      for (const h of handlesFor(r)) {
        ctx.fillRect(h.x - HANDLE_SIZE / 2, h.y - HANDLE_SIZE / 2, HANDLE_SIZE, HANDLE_SIZE);
      }
    }
  });
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

async function saveBoxes() {
  const name = names[idx];
  const payload = currentBoxes.map(b => [b.cls, b.xc, b.yc, b.w, b.h]);
  setBoxStatus("salvataggio...");
  try {
    await fetch("/api/boxes/" + encodeURIComponent(name), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ boxes: payload }),
    });
    setBoxStatus("salvato (" + currentBoxes.length + ")");
  } catch (err) {
    setBoxStatus("errore salvataggio");
  }
}

async function render() {
  if (names.length === 0) return;
  document.getElementById("done").style.display = "none";
  const token = ++renderToken;
  const name = names[idx];
  document.getElementById("idx").textContent = idx + 1;
  document.getElementById("name").textContent = name;
  updateDecisionLabel();
  updateCounts();

  const boxesPromise = fetch("/api/boxes/" + encodeURIComponent(name)).then(r => r.json()).then(d => d.boxes);
  const imageLoaded = new Promise((resolve) => { img.onload = resolve; });
  img.src = "/api/image/" + encodeURIComponent(name);

  const [rawBoxes] = await Promise.all([boxesPromise, imageLoaded]);
  if (token !== renderToken) return; // superata da una navigazione più recente
  currentBoxes = rawBoxes.map(([cls, xc, yc, w, h]) => ({ cls, xc, yc, w, h }));
  selectedIndex = -1;
  drag = null;
  creatingRect = null;
  setBoxStatus(currentBoxes.length ? String(currentBoxes.length) : "-");
  resizeCanvas();
  drawBoxes(currentBoxes);
}

function updateDecisionLabel() {
  const d = decisions[names[idx]];
  const el = document.getElementById("decision");
  el.className = "";
  if (d === "select") { el.textContent = "SELEZIONATA"; el.className = "sel"; }
  else if (d === "reserve") { el.textContent = "CON RISERVA"; el.className = "res"; }
  else if (d === "discard") { el.textContent = "SCARTATA"; el.className = "disc"; }
  else { el.textContent = "-"; }
}

async function setDecision(decision) {
  const name = names[idx];
  if (decision === null) delete decisions[name];
  else decisions[name] = decision;
  await fetch("/api/decision", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, decision }),
  });
  updateCounts();
  if (decision !== null) advance();
  else updateDecisionLabel();
}

function advance() {
  if (idx < names.length - 1) {
    idx++;
    render();
  } else {
    document.getElementById("done").style.display = "block";
  }
}

canvas.addEventListener("mousedown", (e) => {
  const pos = getMousePos(e);

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

  selectedIndex = -1;
  drag = { mode: "create", startX: pos.x, startY: pos.y };
  creatingRect = { x: pos.x, y: pos.y, w: 0, h: 0 };
  drawBoxes(currentBoxes);
});

window.addEventListener("mousemove", (e) => {
  if (!drag) {
    const pos = getMousePos(e);
    let cursor = "crosshair";
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
      const cls = currentBoxes.length ? currentBoxes[0].cls : DEFAULT_CLASS;
      currentBoxes.push({
        cls,
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
  if (e.key === "s") setDecision("select");
  else if (e.key === "n") setDecision("discard");
  else if (e.key === "l") setDecision("reserve");
  else if (e.key === "Backspace") setDecision(null);
  else if (e.key === "ArrowRight") { idx = Math.min(idx + 1, names.length - 1); render(); }
  else if (e.key === "ArrowLeft") { idx = Math.max(idx - 1, 0); render(); }
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

window.addEventListener("resize", () => { resizeCanvas(); drawBoxes(currentBoxes); });
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
            self._json({"names": names, "decisions": load_decisions(self.decisions_path)})
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
            boxes = []
            if lbl_path.exists():
                for line in lbl_path.read_text().splitlines():
                    parts = line.split()
                    if len(parts) == 5:
                        cls = int(float(parts[0]))
                        boxes.append([cls] + [float(v) for v in parts[1:]])
            self._json({"boxes": boxes})
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
                xc = min(1.0, max(0.0, float(xc)))
                yc = min(1.0, max(0.0, float(yc)))
                w = min(1.0, max(0.0, float(w)))
                h = min(1.0, max(0.0, float(h)))
                lines.append(f"{int(cls)} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
            lbl_path.parent.mkdir(parents=True, exist_ok=True)
            content = "\n".join(lines)
            lbl_path.write_text(content + "\n" if content else "")
            self._json({"ok": True})
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
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    if not (dataset_dir / "images").is_dir() or not (dataset_dir / "labels").is_dir():
        parser.error(f"{dataset_dir} deve contenere le sottocartelle images/ e labels/")

    Handler.dataset_dir = dataset_dir
    Handler.decisions_path = (args.decisions_file or dataset_dir / "review_decisions.json").resolve()
    Handler.backup_dir = (args.backup_dir or dataset_dir / "review_label_backups").resolve()

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Dataset: {dataset_dir}")
    print(f"Decisioni: {Handler.decisions_path}")
    print(f"Backup label: {Handler.backup_dir}")
    print(f"Apri http://localhost:{args.port} nel browser (Ctrl+C per fermare)")
    server.serve_forever()


if __name__ == "__main__":
    main()
