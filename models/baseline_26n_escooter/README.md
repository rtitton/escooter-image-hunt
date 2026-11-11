<!-- inizio: generato da export_milestone.py, si rigenera a ogni esecuzione -->

# baseline_26n_escooter

## Run

- Origine: `/home/cric/devp/escooter-image-hunt/runs/detect/data/runs/baseline_26n_escooter`
- Esportata il 2026-11-11; commit corrente del repo: 05a1fa4 (non per forza quello del training)
- Argomenti principali (`args.yaml`): model=yolo26n.pt, data=/home/cric/devp/escooter-image-hunt/data/processed/union_reviewed_coco_split_classes_80/data.yaml, epochs=100, patience=20, batch=16, imgsz=640, workers=4, seed=42, freeze=None, cls_pw=0.0
- Epoche eseguite: 100; tempo dichiarato in `results.csv`: 32.6 ore (può includere pause)
- Validation della run, tutte le classi, migliore epoca (82): P 0.965, R 0.898, mAP50 0.964, mAP50-95 0.742
- Validation della run, ultima epoca (100): P 0.943, R 0.894, mAP50 0.952, mAP50-95 0.740
- `best.pt`: 5.4 MB, SHA-256 `6036448db0e017ce45e5889319dfc386dac8b03b969c4e798245f8186749d3de`

## Valutazione

Report `eval/eval_report_baseline_26n.json`:

| Test set | P | R | mAP50 | mAP50-95 | Recall piccole / medie / grandi |
|---|---|---|---|---|---|
| video_testset_final | 0.809 | 0.610 | 0.726 | 0.502 | 0.151 / 0.738 / 0.859 |
| holdout_kickboard_a75qx_final | 0.986 | 0.939 | 0.972 | 0.816 | n.d. / 1.000 / 0.948 |

Report `eval/eval_baseline_on_valid.json`:

| Test set | P | R | mAP50 | mAP50-95 | Recall piccole / medie / grandi |
|---|---|---|---|---|---|
| valid | 0.965 | 0.897 | 0.964 | 0.742 | 0.786 / 0.773 / 0.970 |

<!-- fine: generato -->

## Dove sono i pesi

`best.pt` e `last.pt` (non nel repo) sono in `runs/detect/data/runs/baseline_26n_escooter/weights/` sulla macchina locale. L'hash SHA-256 di `best.pt` è sopra.

## Note

- È la baseline di riferimento: yolo26n, solo escooter (`--classes 80`, diventa la classe 0), 100 epoche, imgsz 640, batch 16, nessun freeze, seed 42; script `scripts/training/train_yolo26n_baseline.sh`.
- Dati: split `union_reviewed_coco_split` (1812 train, 270 valid), filtrato sulla sola escooter (`union_reviewed_coco_split_classes_80`).
- Il tempo in `results.csv` (32,6 ore) include una sospensione del PC all'epoca 91: la mediana è 120 s per epoca, cioè circa 3,3 ore per la run.
- Monoclasse: da valutare con `eval_testset.py --class-index 0`. Analisi e confronti in `VALUTAZIONI.md` (sezioni 3-6).
