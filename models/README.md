# models/ — milestone di training

Qui stanno le cartelle di training che costituiscono una tappa da ricordare, copiate da `runs/` a run conclusa. `runs/` resta l'area di lavoro di Ultralytics (prove, validazioni, run scartate, run da riprendere con `train_yolo.py --resume`) e si può ripulire senza perdere le milestone.

## Cosa c'è in ogni milestone

```
models/<nome>/
  README.md        come è stata ottenuta: origine, argomenti, riepilogo del training, hash dei pesi,
                   metriche sui test set; parte generata + parte a mano ("Dove sono i pesi", "Note")
  args.yaml        argomenti di training (output di Ultralytics)
  results.csv      metriche per epoca; results.png, matrici di confusione (più, in locale, le curve per classe)
  eval/*.json      report di eval_testset.py relativi ai pesi di questa milestone
  weights/         best.pt e last.pt  (NON nel repo)
```

Nel repo vanno README, `args.yaml`, `results.csv`, `results.png`, le matrici di confusione e `eval/`. Restano fuori (v. `.gitignore`) i pesi, i campioni di immagini (`train_batch*.jpg`, `val_batch*.jpg`, `labels.jpg`) e le curve per classe (`*_curve.png`, circa 1 MB l'una con 81 classi): pesano molto. Dove stanno i pesi si scrive nel README della milestone.

## Come aggiungere una milestone

A training concluso, dalla radice del repo:

```
python3 scripts/export_milestone.py --run runs/detect/data/runs/<run> --name <nome> \
    --eval-report data/tmp/eval_<...>.json
```

copia la run in `models/<nome>/`, allega le voci del report di valutazione che riguardano i suoi pesi e scrive il README. Poi compilare a mano le sezioni segnate `DA COMPILARE` (dove sono i pesi, note). Si può rieseguire sulla milestone esistente (`--run models/<nome>`) per rigenerare README ed `eval/`; la parte a mano resta.

Nome: `<dati>_<modello>_<n>` (per esempio `coco_escooter_yolo26n_1`).

## Milestone

| Cartella | Dati | Modello | Nota |
|---|---|---|---|
| `coco_escooter_yolo26n_1` | COCO + nostro split (nostre immagini x1) | yolo26n, 150 epoche, H100 | prima run su COCO; escooter sotto la baseline (v. `VALUTAZIONI.md`) |
| `coco_escooter_yolo11n_bt` | COCO + vecchio dataset Roboflow non curato | yolo11n, 150 epoche | termine di confronto, non nostra |
