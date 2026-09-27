#!/usr/bin/env python3
"""Copia last.pt di una run Ultralytics in corso ogni N epoche (e a fine
training), in <run-dir>/snapshots/epoch_<NNN>.pt.

Ultralytics tiene solo last.pt (sovrascritto a ogni epoca) e best.pt: senza
snapshot non si possono ricalcolare a posteriori le metriche di una singola
classe alle epoche intermedie, necessarie quando results.csv riporta medie su
più classi (v. compare_training_curves.py). La copia costa pochi MB e non
interferisce con il training.

Avvio (in background, mentre il training gira):
    python3 scripts/snapshot_checkpoints.py runs/detect/data/runs/multiclass_11n [--every 10]

Si ferma da solo quando il training finisce (nessun processo train_yolo.py
attivo e results.csv fermo da --idle-minutes).
"""
import argparse
import csv
import shutil
import subprocess
import time
from pathlib import Path


def n_epochs(results_csv: Path) -> int:
    if not results_csv.exists():
        return 0
    with open(results_csv, newline="") as f:
        return max(sum(1 for _ in csv.reader(f)) - 1, 0)


def training_running() -> bool:
    return subprocess.run(["pgrep", "-f", "scripts/train_yolo.py"], capture_output=True).returncode == 0


def snapshot(run_dir: Path, epoch: int) -> None:
    out = run_dir / "snapshots"
    out.mkdir(exist_ok=True)
    dst = out / f"epoch_{epoch:03d}.pt"
    if not dst.exists():
        shutil.copy2(run_dir / "weights" / "last.pt", dst)
        print(f"{time.strftime('%H:%M:%S')} snapshot epoca {epoch} -> {dst}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--poll", type=int, default=20, help="secondi tra un controllo e l'altro")
    ap.add_argument("--idle-minutes", type=int, default=5)
    args = ap.parse_args()

    results = args.run_dir / "results.csv"
    last_epoch, last_change = 0, time.time()
    while True:
        e = n_epochs(results)
        if e != last_epoch:
            last_epoch, last_change = e, time.time()
            if e and e % args.every == 0:
                time.sleep(20)  # results.csv viene scritto prima di last.pt: aspetta il salvataggio
                if n_epochs(results) == e:  # last.pt è ancora quello dell'epoca e (la successiva dura >1 minuto)
                    snapshot(args.run_dir, e)
        elif not training_running() and time.time() - last_change > args.idle_minutes * 60:
            if last_epoch and last_epoch % args.every:
                snapshot(args.run_dir, last_epoch)
            print("training terminato, esco", flush=True)
            return
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
