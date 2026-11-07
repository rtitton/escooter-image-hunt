<!-- inizio: generato da export_milestone.py, si rigenera a ogni esecuzione -->

# coco_escooter_yolo26n_1

## Run

- Origine: `/home/cric/devp/escooter-image-hunt/models/coco_escooter_yolo26n_1`
- Esportata il 2026-11-07; commit corrente del repo: 7f327bd (con modifiche non committate) (non per forza quello del training)
- Argomenti principali (`args.yaml`): model=yolo26n.pt, data=/app/escooter-image-hunt/data/processed/coco_escooter_merge/data.yaml, epochs=150, patience=20, batch=32, imgsz=640, workers=16, seed=42, freeze=None, cls_pw=0.0
- Epoche eseguite: 150; tempo dichiarato in `results.csv`: 16.7 ore (può includere pause)
- Validation della run, tutte le classi, migliore epoca (150): P 0.649, R 0.517, mAP50 0.560, mAP50-95 0.399
- Validation della run, ultima epoca (150): P 0.649, R 0.517, mAP50 0.560, mAP50-95 0.399
- `best.pt`: 5.5 MB, SHA-256 `f94f2b75b907ac5d8f077a684c02f45262364590230f1501d11491049b478da7`

## Valutazione

Report `eval/eval_coco_h100_models.json`:

| Test set | P | R | mAP50 | mAP50-95 | Recall piccole / medie / grandi |
|---|---|---|---|---|---|
| video_testset_final | 0.779 | 0.406 | 0.501 | 0.320 | 0.000 / 0.331 / 0.663 |
| holdout_kickboard_a75qx_final | 0.967 | 0.904 | 0.959 | 0.787 | n.d. / 1.000 / 0.948 |
| valid | 0.948 | 0.814 | 0.910 | 0.667 | 0.071 / 0.533 / 0.961 |

<!-- fine: generato -->

## Dove sono i pesi

DA COMPILARE: dove stanno `best.pt` e `last.pt` fuori dal repo (percorso su cloud storage, macchina) e, se serve, la data di caricamento. L'hash SHA-256 di `best.pt` è sopra.

## Note

- Dati: COCO train2017 più il nostro split `union_reviewed_coco_split` (1812 train, nostre immagini ripetute 1 volta, lista `coco_escooter_merge`); validation: COCO val2017 più il nostro valid. Run sulla macchina H100.
- Non si è fermata per early stopping: la migliore epoca è l'ultima, quindi stava ancora migliorando.
- Sulla escooter è sotto la baseline monoclasse e sotto il multiclasse a 81 classi sui soli nostri dati, perché le nostre immagini sono circa l'1,5% del train (v. `VALUTAZIONI.md`, sezione 6). Prossimo passo previsto: ripetere le nostre immagini 3-5 volte.
- Il nome della run in `args.yaml` (`coco_escooter_26s_rep1-16`) dice 26s, ma il modello è yolo26n. `args.yaml` contiene inoltre `cls_remap: true`, argomento che non è nell'Ultralytics 8.4.126 usato in locale.
