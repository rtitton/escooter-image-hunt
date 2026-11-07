<!-- inizio: generato da export_milestone.py, si rigenera a ogni esecuzione -->

# coco_escooter_yolo11n_bt

## Run

- Origine: `/home/cric/devp/escooter-image-hunt/models/coco_escooter_yolo11n_bt`
- Esportata il 2026-11-07; commit corrente del repo: 7f327bd (con modifiche non committate) (non per forza quello del training)
- Argomenti principali (`args.yaml`): model=yolo11n.pt, data=dataYolo.yaml, epochs=150, patience=50, batch=120, imgsz=640, workers=8, seed=0, freeze=None
- Epoche eseguite: 150; tempo dichiarato in `results.csv`: 11.7 ore (può includere pause)
- Validation della run, tutte le classi, migliore epoca (150): P 0.649, R 0.494, mAP50 0.540, mAP50-95 0.384
- Validation della run, ultima epoca (150): P 0.649, R 0.494, mAP50 0.540, mAP50-95 0.384
- `best.pt`: 5.5 MB, SHA-256 `7eae40e5bd3353918a5c0c56ac242ddf7c194292b6b4736ebed0aeb2255fbab5`

## Valutazione

Report `eval/eval_coco_h100_models.json`:

| Test set | P | R | mAP50 | mAP50-95 | Recall piccole / medie / grandi |
|---|---|---|---|---|---|
| video_testset_final | 0.686 | 0.338 | 0.411 | 0.281 | 0.000 / 0.234 / 0.663 |
| holdout_kickboard_a75qx_final | 0.976 | 0.918 | 0.971 | 0.760 | n.d. / 1.000 / 0.918 |
| valid | 0.918 | 0.745 | 0.838 | 0.561 | 0.000 / 0.240 / 0.878 |

<!-- fine: generato -->

## Dove sono i pesi

DA COMPILARE: dove stanno `best.pt` e `last.pt` fuori dal repo (percorso su cloud storage, macchina) e, se serve, la data di caricamento. L'hash SHA-256 di `best.pt` è sopra.

## Note

- Run vecchia, non nostra: COCO più un dataset Roboflow di monopattini non curato (`dataYolo.yaml`), 81 classi con la classe 80 chiamata `Monopattino`. Serve solo come termine di confronto per il lavoro sul dataset curato.
- Ricetta diversa dalla milestone `coco_escooter_yolo26n_1`: yolo11n invece di yolo26n, batch 120, `multi_scale` attivo, patience 50, seed 0. Il confronto non è controllato; non è noto se il dataset Roboflow si sovrapponga ai nostri sorgenti o ai test set.
- Non si è fermata per early stopping: la migliore epoca è l'ultima.
