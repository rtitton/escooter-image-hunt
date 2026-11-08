# TODO

## 1. Training su H100: COCO completo + nostro split, REPEAT=5

Obiettivo: verificare se ripetere 5 volte le nostre immagini recupera la escooter (con REPEAT=1 la escooter peggiorava per diluizione: video mAP50 0,501 contro 0,726 della baseline monoclasse). Riferimento: `models/coco_escooter_yolo26n_1/` (REPEAT=1, yolo26n, 150 epoche).

Cambia solo REPEAT: stesso modello (yolo26n), stesse 150 epoche, stesso seme 42, stessi altri parametri.

### Prerequisiti sulla H100

1. Aggiornare il repo (devono esserci i commit con `scripts/training/` e `scripts/dataset/`):
   ```bash
   git pull
   ```
2. Ambiente `uv` pronto (lo script usa `uv run python3`), con `pyproject-h100.toml` come base.
3. COCO in formato Ultralytics (come lo scarica `scripts/training/download_coco.sh`: `images/{train2017,val2017}`, `labels/{train2017,val2017}`) in `/data/dataset/coco`. Se sta altrove, `COCO_DIR=<percorso>` davanti al comando.
4. Il nostro split in `data/processed/union_reviewed_coco_split` (cartelle `train/` e `valid/`, ciascuna con `images/` e `labels/`). Non è nel repo: si copia dalla macchina locale (stesso split usato per `coco_escooter_yolo26n_1`).
5. `yolo26n.pt` nella radice del repo (i `.pt` non sono nel repo).

### Avvio

Dalla radice del repo:

```bash
REPEAT=5 scripts/training/train_yolo26_coco_escooter.sh
```

Default dello script: `yolo26n.pt`, 150 epoche, batch 32, 16 workers, `patience` 20, `imgsz` 640, `--device 0`.
- Run: `coco_escooter_yolo26n_rep5`.
- Liste e `data.yaml`: `data/processed/coco_escooter_merge_rep5/` (rigenerate a ogni avvio; la `train.txt` contiene COCO train2017 + le 1812 nostre immagini ripetute 5 volte).
- Output di Ultralytics: sotto `runs/detect/data/runs/coco_escooter_yolo26n_rep5/` (cartella da verificare al primo avvio).

All'avvio lo script stampa i conteggi: attesi 118287 immagini COCO + 1812 × 5 nostre = 127347 righe in `train.txt`, e 5000 + 270 = 5270 in `val.txt`.

### Se si interrompe

Con lo stesso `NAME` e gli stessi parametri:

```bash
RESUME=1 REPEAT=5 scripts/training/train_yolo26_coco_escooter.sh
```

Non rigenera le liste e riparte da `last.pt`.

### A training concluso

1. Copiare `best.pt` (e il resto della run) sulla macchina locale.
2. Valutare sui test set esterni (classe escooter = 80):
   ```bash
   python3 scripts/training/eval_testset.py \
       --weights <percorso>/best.pt --class-index 80 --imgsz 640 \
       --testset data/processed/video_testset_final data/processed/holdout_kickboard_a75qx_final \
       --out data/tmp/eval_coco_escooter_yolo26n_rep5.json
   ```
3. Controllare che le classi COCO non siano peggiorate (richiede COCO in locale: `data/coco`):
   ```bash
   python3 scripts/training/compare_coco_models.py --baseline yolo26n.pt \
       --trained <percorso>/best.pt --coco-dir data/coco --out-dir data/tmp/coco_comparison --device 0
   ```
4. Esportare come milestone:
   ```bash
   python3 scripts/training/export_milestone.py --run <cartella della run> \
       --name coco_escooter_yolo26n_rep5 --eval-report data/tmp/eval_coco_escooter_yolo26n_rep5.json
   ```
5. Aggiornare `VALUTAZIONI.md` (sezione 6, confronto con REPEAT=1 e con la baseline monoclasse) e, se serve, aggiungere il modello a `MODELLI_DEFAULT` di `scripts/training/infer_videos.py`.

### Come leggere il risultato

- Escooter sul video test set da confrontare con 0,501 (REPEAT=1) e 0,726 (baseline); recall sui piccoli (<32 px) con 0,151 della baseline.
- Classi COCO: mAP50-95 globale di riferimento 0,399 (REPEAT=1).
- Se REPEAT=5 non basta, i passi successivi sono più esempi piccoli nel train e la linea del subset COCO (`scripts/training/make_coco_subset.sh`, trattata come step a sé).
