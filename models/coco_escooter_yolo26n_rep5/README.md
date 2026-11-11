<!-- inizio: generato da export_milestone.py, si rigenera a ogni esecuzione -->

# coco_escooter_yolo26n_rep5

## Run

- Origine: `/home/cric/devp/escooter-image-hunt/models/coco_escooter_yolo26n_rep5`
- Esportata il 2026-11-11; commit corrente del repo: ce448d0 (non per forza quello del training)
- Argomenti principali (`args.yaml`): model=tmp/runs/coco_escooter_yolo26n_rep5-10/weights/last.pt, data=/app/escooter-image-hunt/data/processed/coco_escooter_merge_rep5/data.yaml, epochs=150, patience=20, batch=32, imgsz=640, workers=16, seed=42, freeze=0, cls_pw=0.0
- Epoche eseguite: 150; tempo dichiarato in `results.csv`: 3.1 ore (può includere pause)
- Validation della run, tutte le classi, migliore epoca (150): P 0.662, R 0.499, mAP50 0.543, mAP50-95 0.390
- Validation della run, ultima epoca (150): P 0.662, R 0.499, mAP50 0.543, mAP50-95 0.390
- `best.pt`: 5.5 MB, SHA-256 `c3096831d5339072fc3b595eb47300e81c3293fef737a35df8e4e3d7b4f5ccc8`

## Valutazione

Report `eval/eval_coco_rep5.json`:

| Test set | P | R | mAP50 | mAP50-95 | Recall piccole / medie / grandi |
|---|---|---|---|---|---|
| video_testset_final | 0.785 | 0.430 | 0.530 | 0.365 | 0.000 / 0.400 / 0.674 |
| holdout_kickboard_a75qx_final | 0.974 | 0.949 | 0.960 | 0.817 | n.d. / 1.000 / 0.938 |
| valid | 0.963 | 0.865 | 0.952 | 0.731 | 0.143 / 0.693 / 0.973 |

<!-- fine: generato -->

## Dove sono i pesi

`best.pt` e `last.pt` (identici, finali, senza stato dell'ottimizzatore) sono in `/mnt/x/coco_escooter_yolo26n_rep5-10/weights/` e in locale in `models/coco_escooter_yolo26n_rep5/weights/` (non nel repo). Il checkpoint riprendibile della prima tranche (16 MB, epoca 93) è in `/mnt/x/coco_escooter_yolo26n_rep5/weights/`.

## Note

- Dati: COCO train2017 più il nostro split `union_reviewed_coco_split` con le nostre immagini ripetute 5 volte (lista `coco_escooter_merge_rep5`). Run sulla macchina H100.
- La run è stata interrotta all'epoca 93 e ripresa con `resume`; Ultralytics ha scritto la seconda tranche in una cartella nuova. Qui `results.csv` è l'unione delle due parti (epoche 1-93 da `/mnt/x/coco_escooter_yolo26n_rep5`, 94-150 da `/mnt/x/coco_escooter_yolo26n_rep5-10`); i `time` ripartono da zero dopo l'epoca 93. Curve, matrici e `args.yaml` sono della seconda tranche (in `args.yaml`: `resume` impostato e `warmup_bias_lr` 0.0 invece di 0.1).
- Non si è fermata per early stopping: la migliore epoca è l'ultima (150), la curva saliva ancora.
- Rispetto a `coco_escooter_yolo26n_1` (rep1) la escooter migliora molto (video mAP50 0,530 contro 0,501, valid 0,952 contro 0,910, kickboard 0,960 contro 0,959), ma resta sotto la baseline monoclasse sul video (0,726) e non trova nessuna escooter piccola (0/86). Ripetere le immagini aiuta, non basta.
