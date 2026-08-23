#!/usr/bin/env python3
"""Applicazione locale per la selezione manuale finale delle immagini del
dataset di unione (data/processed/union/).

Mostra un'immagine alla volta con le bbox disegnate (canvas HTML), e
registra la decisione dell'utente con un tasto:
    s = seleziona
    n = scarta
    l = seleziona con riserva
    freccia sinistra/destra = torna indietro / salta avanti senza decidere
    backspace = cancella la decisione sull'immagine corrente

Le decisioni sono scritte subito in data/review_decisions.json ad ogni
tasto premuto: la sessione si può interrompere e riprendere quando si vuole,
riparte dalla prima immagine ancora senza decisione.

Avvio:
    python3 scripts/review_app.py [--port 8765]
poi apri http://localhost:<port> nel browser.
"""
import argparse
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"
UNION_DIR = DATA_ROOT / "processed" / "union"
DECISIONS_PATH = DATA_ROOT / "review_decisions.json"


def load_decisions() -> dict:
    if DECISIONS_PATH.exists():
        return json.loads(DECISIONS_PATH.read_text())
    return {}


def save_decisions(decisions: dict) -> None:
    DECISIONS_PATH.write_text(json.dumps(decisions, indent=2, ensure_ascii=False) + "\n")


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
  #stage { position: relative; display: flex; align-items: center; justify-content: center; height: calc(100vh - 110px); }
  #stage img { max-width: 96vw; max-height: 100%; display: block; }
  #stage canvas { position: absolute; pointer-events: none; }
  #name { text-align: center; padding: 6px; font-size: 13px; color: #999; word-break: break-all; }
  #decision { font-weight: bold; }
  .sel { color: #4caf50; }
  .disc { color: #e53935; }
  .res { color: #ffb300; }
  #help { text-align: center; padding: 6px; font-size: 13px; color: #888; }
  #help kbd { background: #333; border-radius: 3px; padding: 1px 6px; margin: 0 2px; }
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
<div id="done">Fine: tutte le immagini sono state revisionate.</div>

<script>
let names = [];
let decisions = {};
let idx = 0;

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

function drawBoxes(boxes) {
  canvas.width = img.clientWidth;
  canvas.height = img.clientHeight;
  canvas.style.width = img.clientWidth + "px";
  canvas.style.height = img.clientHeight + "px";
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = "#ff2d55";
  ctx.lineWidth = 3;
  for (const [xc, yc, w, h] of boxes) {
    const x = (xc - w / 2) * canvas.width;
    const y = (yc - h / 2) * canvas.height;
    ctx.strokeRect(x, y, w * canvas.width, h * canvas.height);
  }
}

async function render() {
  if (names.length === 0) return;
  document.getElementById("done").style.display = "none";
  const name = names[idx];
  document.getElementById("idx").textContent = idx + 1;
  document.getElementById("name").textContent = name;
  img.src = "/api/image/" + encodeURIComponent(name);
  const boxesResp = await fetch("/api/boxes/" + encodeURIComponent(name));
  const { boxes } = await boxesResp.json();
  img.onload = () => drawBoxes(boxes);
  updateDecisionLabel();
  updateCounts();
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

document.addEventListener("keydown", (e) => {
  if (e.key === "s") setDecision("select");
  else if (e.key === "n") setDecision("discard");
  else if (e.key === "l") setDecision("reserve");
  else if (e.key === "Backspace") setDecision(null);
  else if (e.key === "ArrowRight") { idx = Math.min(idx + 1, names.length - 1); render(); }
  else if (e.key === "ArrowLeft") { idx = Math.max(idx - 1, 0); render(); }
});

window.addEventListener("resize", () => render());
loadState();
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def _json(self, obj: dict, status: int = 200) -> None:
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            body = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/state":
            names = sorted(p.name for p in (UNION_DIR / "images").iterdir())
            self._json({"names": names, "decisions": load_decisions()})
        elif self.path.startswith("/api/image/"):
            name = unquote(self.path[len("/api/image/"):])
            path = UNION_DIR / "images" / name
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
            lbl_path = UNION_DIR / "labels" / f"{Path(name).stem}.txt"
            boxes = []
            if lbl_path.exists():
                for line in lbl_path.read_text().splitlines():
                    parts = line.split()
                    if len(parts) == 5:
                        boxes.append([float(v) for v in parts[1:]])
            self._json({"boxes": boxes})
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        if self.path == "/api/decision":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            name, decision = body["name"], body["decision"]
            decisions = load_decisions()
            if decision is None:
                decisions.pop(name, None)
            else:
                decisions[name] = decision
            save_decisions(decisions)
            self._json({"ok": True})
        else:
            self.send_error(404)

    def log_message(self, format: str, *args) -> None:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Apri http://localhost:{args.port} nel browser (Ctrl+C per fermare)")
    server.serve_forever()


if __name__ == "__main__":
    main()
