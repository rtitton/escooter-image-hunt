#!/usr/bin/env python3
"""Archivia una run di training Ultralytics conclusa come milestone in
models/<nome>/ e ne scrive il README, così la cartella è autosufficiente:
pesi, argomenti e curve del training (copiati dalla run), report di
valutazione e un riepilogo di come è stata ottenuta.

La copia vale solo per run CONCLUSE: Ultralytics toglie dal last.pt lo
stato dell'ottimizzatore a fine training, e una run ancora riprendibile
(last.pt con quello stato) va lasciata dov'è, in runs/, perché
`train_yolo.py --resume` la cerca lì. Con --allow-unfinished si copia lo
stesso.

Il README ha una parte generata (fra due marcatori HTML) che si rigenera a
ogni esecuzione e una parte a mano ("Dove sono i pesi", "Note") che resta
com'è: le sezioni da compilare sono segnate "DA COMPILARE". La parte generata
contiene: origine, commit del repo al momento dell'esportazione (non per
forza quello con cui è stato fatto il training), argomenti principali da
args.yaml, riepilogo di results.csv, SHA-256 di best.pt e, per ogni
--eval-report, la tabella delle metriche dei pesi di questa milestone. I
report passati vengono filtrati: in models/<nome>/eval/ finiscono solo le
voci dei pesi di questa milestone, individuate dal nome della cartella
(della milestone o della run sorgente) nel percorso dei pesi.

Se --run è già models/<nome> (milestone esistente) non copia nulla e
rigenera solo README ed eval/.

Avvio:
    python3 scripts/export_milestone.py --run RUN_DIR [--name NOME]
        [--eval-report FILE.json ...] [--no-last] [--allow-unfinished]
"""
import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from datetime import date
from pathlib import Path

import yaml

MODELS_DIR = Path("models")
BEGIN = "<!-- inizio: generato da export_milestone.py, si rigenera a ogni esecuzione -->"
END = "<!-- fine: generato -->"
MANUAL_DEFAULT = """## Dove sono i pesi

DA COMPILARE: dove stanno `best.pt` e `last.pt` fuori dal repo (percorso su cloud storage, macchina) e, se serve, la data di caricamento. L'hash SHA-256 di `best.pt` è sopra.

## Note

DA COMPILARE: dataset e split usati, differenze rispetto ad altre milestone, anomalie della run.
"""
RESULT_KEYS = ("metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)")
ARG_KEYS = ("model", "data", "epochs", "patience", "batch", "imgsz", "workers", "seed", "freeze", "cls_pw")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True,
                               check=True).stdout.strip()
        return head + (" (con modifiche non committate)" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "non disponibile"


def is_unfinished(run: Path) -> bool:
    """True se last.pt ha ancora epoca e stato dell'ottimizzatore, cioè la run è riprendibile."""
    last = run / "weights" / "last.pt"
    if not last.is_file():
        return False
    from ultralytics import YOLO

    ckpt = YOLO(str(last)).ckpt or {}
    return ckpt.get("epoch", -1) >= 0 and ckpt.get("optimizer") is not None


def read_results(run: Path) -> list:
    path = run / "results.csv"
    if not path.is_file():
        return []
    return [{k.strip(): v for k, v in row.items()} for row in csv.DictReader(path.open())]


def metrics_line(row: dict) -> str:
    parts = []
    for key, label in zip(RESULT_KEYS, ("P", "R", "mAP50", "mAP50-95")):
        if key in row:
            parts.append(f"{label} {float(row[key]):.3f}")
    return ", ".join(parts)


def eval_entries(report_path: Path, names: set) -> dict:
    """Le voci del report (pesi -> test set -> metriche) i cui pesi stanno in una cartella di nome in names."""
    report = json.loads(report_path.read_text())
    return {w: ts for w, ts in report.items() if names & set(Path(w).parts)}


def build_block(name: str, source: Path, run: Path, eval_files: dict) -> str:
    lines = [BEGIN, "", f"# {name}", "", "## Run", ""]
    lines.append(f"- Origine: `{source}`")
    lines.append(f"- Esportata il {date.today().isoformat()}; commit corrente del repo: {git_commit()} (non per forza quello del training)")
    args_path = run / "args.yaml"
    if args_path.is_file():
        args = yaml.safe_load(args_path.read_text())
        shown = ", ".join(f"{k}={args[k]}" for k in ARG_KEYS if k in args)
        lines.append(f"- Argomenti principali (`args.yaml`): {shown}")
    rows = read_results(run)
    if rows:
        best = max(rows, key=lambda r: float(r.get("metrics/mAP50-95(B)", 0)))
        hours = float(rows[-1]["time"]) / 3600 if "time" in rows[-1] else None
        lines.append(f"- Epoche eseguite: {len(rows)}" + (f"; tempo dichiarato in `results.csv`: {hours:.1f} ore (può includere pause)" if hours else ""))
        lines.append(f"- Validation della run, tutte le classi, migliore epoca ({best['epoch']}): {metrics_line(best)}")
        lines.append(f"- Validation della run, ultima epoca ({rows[-1]['epoch']}): {metrics_line(rows[-1])}")
    best_pt = run / "weights" / "best.pt"
    if best_pt.is_file():
        lines.append(f"- `best.pt`: {best_pt.stat().st_size / 1e6:.1f} MB, SHA-256 `{sha256(best_pt)}`")
    lines += ["", "## Valutazione", ""]
    if not eval_files:
        lines.append("Nessun report di valutazione allegato (usare `--eval-report`).")
    for fname, entries in eval_files.items():
        lines += [f"Report `eval/{fname}`:", "", "| Test set | P | R | mAP50 | mAP50-95 | Recall piccole / medie / grandi |", "|---|---|---|---|---|---|"]
        for weights, testsets in entries.items():
            for testset, m in testsets.items():
                size = (m.get("size_breakdown") or {}).get("recall_by_gt_size", {})

                def recall(bucket):
                    d = size.get(bucket)
                    total = d["tp"] + d["fn"] if d else 0
                    return f"{d['tp'] / total:.3f}" if total else "n.d."

                lines.append(f"| {testset} | {m['precision']:.3f} | {m['recall']:.3f} | {m['mAP50']:.3f} | {m['mAP50-95']:.3f} | "
                             f"{recall('small')} / {recall('medium')} / {recall('large')} |")
        lines.append("")
    lines.append(END)
    return "\n".join(lines)


def write_readme(dest: Path, block: str) -> None:
    readme = dest / "README.md"
    manual = MANUAL_DEFAULT
    if readme.is_file():
        old = readme.read_text()
        manual = old.split(END, 1)[1].strip("\n") + "\n" if END in old else old.strip("\n") + "\n"
    readme.write_text(block + "\n\n" + manual)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True, help="Cartella della run Ultralytics (o di una milestone esistente)")
    parser.add_argument("--name", default=None, help="Nome della milestone in models/ (default: nome della cartella della run)")
    parser.add_argument("--eval-report", type=Path, nargs="*", default=[],
                         help="Report JSON di eval_testset.py da allegare (solo le voci di questi pesi)")
    parser.add_argument("--no-last", action="store_true", help="Non copiare last.pt (solo best.pt)")
    parser.add_argument("--allow-unfinished", action="store_true", help="Copia anche una run ancora riprendibile")
    args = parser.parse_args()

    source = args.run.resolve()
    if not (source / "weights" / "best.pt").is_file():
        raise SystemExit(f"{source}/weights/best.pt non trovato: non è la cartella di una run conclusa")
    name = args.name or source.name
    dest = (MODELS_DIR / name).resolve()

    if dest == source:
        print(f"{dest} è già una milestone: rigenero README ed eval/")
    elif dest.exists():
        raise SystemExit(f"{dest} esiste già: scegliere un altro --name")
    else:
        if is_unfinished(source) and not args.allow_unfinished:
            raise SystemExit(f"{source} sembra una run interrotta (last.pt ha ancora lo stato dell'ottimizzatore): "
                             "riprenderla con train_yolo.py --resume, o usare --allow-unfinished")
        shutil.copytree(source, dest, ignore=shutil.ignore_patterns("last.pt") if args.no_last else None)
        print(f"Copiata la run in {dest}")

    names = {name, source.name}
    eval_files = {}
    for report in args.eval_report:
        entries = eval_entries(report, names)
        if not entries:
            print(f"ATTENZIONE: nessuna voce di {names} in {report}, non allegato")
            continue
        (dest / "eval").mkdir(exist_ok=True)
        (dest / "eval" / report.name).write_text(json.dumps(entries, indent=2, ensure_ascii=False) + "\n")
        eval_files[report.name] = entries
    # report già allegati in precedenza, quando si rigenera una milestone esistente
    if (dest / "eval").is_dir():
        for existing in sorted((dest / "eval").glob("*.json")):
            if existing.name not in eval_files:
                eval_files[existing.name] = json.loads(existing.read_text())

    write_readme(dest, build_block(name, source, dest, eval_files))
    print(f"README scritto in {dest / 'README.md'}")


if __name__ == "__main__":
    main()
