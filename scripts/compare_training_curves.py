#!/usr/bin/env python3
"""Confronta, ogni N epoche, le metriche di validation della SOLA classe
escooter di una run (tipicamente multiclasse) con quelle di una run monoclasse
di riferimento, sullo stesso valid.

Il results.csv di una run multiclasse riporta metriche medie su tutte le
classi, non confrontabili con quelle di una run monoclasse. Per la run di
confronto si usano quindi gli snapshot di last.pt prodotti da
snapshot_checkpoints.py (<run-dir>/snapshots/epoch_NNN.pt), valutati con
model.val() sul valid del data.yaml della run, leggendo solo la classe
--class-index. Per il riferimento monoclasse (una sola classe) bastano le righe
del suo results.csv, calcolate da Ultralytics con la stessa validazione.

Le valutazioni degli snapshot sono salvate in <run-dir>/snapshots/eval_cache.json
(non si rifanno). Girano su CPU per non contendere la GPU al training.

Avvio:
    python3 scripts/compare_training_curves.py \\
        --baseline-run runs/detect/data/runs/escooter_only_11n \\
        --run runs/detect/data/runs/multiclass_11n --class-index 3 [--every 10]
"""
import argparse
import csv
import json
import re
from pathlib import Path

import torch
import yaml
from ultralytics import YOLO

COLUMNS = ["metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"]


def read_results(run_dir: Path) -> dict:
    """epoca -> (P, R, mAP50, mAP50-95) dal results.csv di Ultralytics."""
    out = {}
    with open(run_dir / "results.csv", newline="") as f:
        for row in csv.DictReader(f):
            row = {k.strip(): v for k, v in row.items()}
            out[int(float(row["epoch"]))] = tuple(float(row[c]) for c in COLUMNS)
    return out


def eval_snapshot(snap: Path, data_yaml: Path, class_index: int, workers: int) -> tuple:
    metrics = YOLO(str(snap)).val(data=str(data_yaml), imgsz=640, device="cpu", workers=workers,
                                   plots=False, verbose=False)
    i = list(metrics.box.ap_class_index).index(class_index)
    return (float(metrics.box.p[i]), float(metrics.box.r[i]), float(metrics.box.ap50[i]), float(metrics.box.ap[i]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline-run", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--class-index", type=int, required=True, help="indice della classe escooter nella run")
    ap.add_argument("--every", type=int, default=10)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    data_yaml = Path(yaml.safe_load((args.run / "args.yaml").read_text())["data"])
    base = read_results(args.baseline_run)
    snaps = sorted(args.run.glob("snapshots/epoch_*.pt"))
    if not snaps:
        raise SystemExit(f"nessuno snapshot in {args.run}/snapshots (v. snapshot_checkpoints.py)")
    cache_path = args.run / "snapshots" / "eval_cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    rows = []
    for snap in snaps:
        epoch = int(re.search(r"epoch_(\d+)", snap.name).group(1))
        if str(epoch) not in cache:
            print(f"valuto lo snapshot dell'epoca {epoch}...", flush=True)
            cache[str(epoch)] = eval_snapshot(snap, data_yaml, args.class_index, workers=2)
            cache_path.write_text(json.dumps(cache, indent=1))
        rows.append((epoch, base.get(epoch), tuple(cache[str(epoch)])))

    print(f"\nValid interno, sola classe escooter. Riferimento: {args.baseline_run.name} "
          f"({len(base)} epoche); confronto: {args.run.name}, classe {args.class_index}")
    h = f"{'epoca':>5s} | {'mono P':>6s} {'R':>6s} {'mAP50':>6s} {'m50-95':>6s} | {'multi P':>7s} {'R':>6s} {'mAP50':>6s} {'m50-95':>6s} | {'Δ mAP50':>7s} {'Δ m50-95':>8s}"
    print(h + "\n" + "-" * len(h))
    for epoch, b, m in rows:
        if b is None:
            print(f"{epoch:5d} | {'(il riferimento si è fermato prima)':^29s} | {m[0]:7.3f} {m[1]:6.3f} {m[2]:6.3f} {m[3]:6.3f} |")
            continue
        print(f"{epoch:5d} | {b[0]:6.3f} {b[1]:6.3f} {b[2]:6.3f} {b[3]:6.3f} | {m[0]:7.3f} {m[1]:6.3f} {m[2]:6.3f} {m[3]:6.3f} "
              f"| {m[2] - b[2]:+7.3f} {m[3] - b[3]:+8.3f}")


if __name__ == "__main__":
    main()
