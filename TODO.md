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
3. COCO in formato Ultralytics (come lo scarica `scripts/training/download_coco.sh`: `images/{train2017,val2017}`, `labels/{train2017,val2017}`) in `/data/dataset/coco`. Se sta altrove, `COCO_DIR=<percorso>` davanti al comando di costruzione del dataset.
4. Il nostro split in `data/processed/union_reviewed_coco_split` (cartelle `train/` e `valid/`, ciascuna con `images/` e `labels/`). Non è nel repo: si copia dalla macchina locale (stesso split usato per `coco_escooter_yolo26n_1`).
5. `yolo26n.pt` nella radice del repo (i `.pt` non sono nel repo).

### Avvio

Dalla radice del repo, in due passi (la costruzione del dataset è separata dal training).

1. Liste e `data.yaml`:
   ```bash
   REPEAT=5 scripts/training/build_coco_merged_lists.sh
   ```
   Scrive in `data/processed/coco_escooter_merge_rep5/` (la `train.txt` contiene COCO train2017 + le 1812 nostre immagini ripetute 5 volte) e stampa alla fine il comando del passo 2.
2. Training:
   ```bash
   DATA_YAML=data/processed/coco_escooter_merge_rep5/data.yaml NAME=coco_escooter_yolo26n_rep5 \
       scripts/training/train_yolo26_coco_escooter.sh
   ```

Default dello script di training: `yolo26n.pt`, 150 epoche, batch 32, 16 workers, `patience` 20, `imgsz` 640, `--device 0`.
- Run: `coco_escooter_yolo26n_rep5` (senza `NAME` il default sarebbe `yolo26n_coco_escooter_merge_rep5`, dal nome della cartella del `data.yaml`).
- Output di Ultralytics: sotto `runs/detect/data/runs/coco_escooter_yolo26n_rep5/` (cartella da verificare al primo avvio).

Il passo 1 stampa i conteggi: attesi 118287 immagini COCO + 1812 × 5 nostre = 127347 righe in `train.txt`, e 5000 + 270 = 5270 in `val.txt`.

### Se si interrompe

Con lo stesso `NAME` e gli stessi parametri:

```bash
RESUME=1 DATA_YAML=data/processed/coco_escooter_merge_rep5/data.yaml NAME=coco_escooter_yolo26n_rep5 \
    scripts/training/train_yolo26_coco_escooter.sh
```

Riparte da `last.pt` con le liste già costruite (non rilanciare il passo 1).

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
- Se REPEAT=5 non basta, i passi successivi sono più esempi piccoli nel train e la linea del subset COCO (`scripts/training/make_coco_subset.sh`, trattata come step a sé: il subset si passa a `build_coco_merged_lists.sh` con `COCO_SUBSET_DIR`).

## 2. Training su H100: subset COCO (20000 immagini, K=4000) + nostro split, REPEAT=5

Obiettivo: vedere se il subset con le nostre immagini ripetute 5 volte si avvicina alla baseline monoclasse sul video test set (0,726 mAP50), soprattutto sulle medie e sulle piccole. Con REPEAT=1 il subset ha dato 0,540 (`models/yolo26n_cocosub_20000_4000_escooter_merge_rep1-2/`), meglio di COCO completo con REPEAT=1 (0,501) e con REPEAT=5 (0,530). Le nostre immagini passano da circa il 7% a circa il 27% del train.

Cambia solo REPEAT rispetto alla run con REPEAT=1: stesso subset, stesso modello (yolo26n), 150 epoche, batch 32, seme 42, stessi altri parametri. Prerequisiti come nel punto 1 (repo aggiornato con `git pull`, ambiente `uv`, COCO in `/data/dataset/coco`, nostro split, `yolo26n.pt`).

### Avvio

Dalla radice del repo, in tre passi.

1. Subset (se non c'è già `data/processed/coco_subset_20000_4000`; con lo stesso seme 42 e lo stesso COCO è identico a quello della run con REPEAT=1):
   ```bash
   COCO_DIR=/data/dataset/coco MAX_IMAGES=20000 MIN_PER_CLASS=4000 scripts/training/make_coco_subset.sh
   ```
   Con K=4000 il subset ha 24976 immagini di train e 3172 di val (lo script avvisa che superano `MAX_IMAGES`: è atteso, v. sotto).
2. Liste e `data.yaml`:
   ```bash
   COCO_SUBSET_DIR=data/processed/coco_subset_20000_4000 REPEAT=5 \
       OUT_DIR=data/processed/cocosub_20000_4000_escooter_merge_rep5 \
       scripts/training/build_coco_merged_lists.sh
   ```
   Righe attese: `train.txt` 24976 COCO + 1812 × 5 nostre = 34036; `val.txt` 3172 + 270 = 3442.
3. Training:
   ```bash
   DATA_YAML=data/processed/cocosub_20000_4000_escooter_merge_rep5/data.yaml \
       NAME=yolo26n_cocosub_20000_4000_escooter_merge_rep5 \
       scripts/training/train_yolo26_coco_escooter.sh
   ```

Note:
- `NAME` e `OUT_DIR` espliciti per non sovrascrivere la run con REPEAT=1 e quelle su COCO completo.
- Se si interrompe: stesso comando del passo 3 con `RESUME=1` davanti (stessi `DATA_YAML` e `NAME`, senza rilanciare i passi 1 e 2).
- Con la run precedente la curva è piatta dopo circa 100 epoche e `patience` 20 non è scattato: per accorciare si può passare `EPOCHS=100`, ma allora il confronto con REPEAT=1 (150 epoche) non è più a parità di condizioni.

### A training concluso

1. Copiare `best.pt` (e il resto della run) sulla macchina locale, in `models/`.
2. Valutare sui test set esterni (classe escooter = 80; in locale l'eval gira su CPU, è lenta ma valida):
   ```bash
   uv run python3 scripts/training/eval_testset.py \
       --weights models/yolo26n_cocosub_20000_4000_escooter_merge_rep5/weights/best.pt \
       --class-index 80 --imgsz 640 \
       --testset data/processed/video_testset_final data/processed/holdout_kickboard_a75qx_final \
       --out data/tmp/eval_cocosub_20000_4000_rep5.json
   ```
3. Se interessa il confronto delle classi COCO, `compare_coco_models.py` va letto solo sulle 13 classi del subset (le altre 67 sono state allenate a ignorarle).
4. Esportare come milestone con `export_milestone.py` e aggiornare `VALUTAZIONI.md`.

### Come leggere il risultato

Riferimenti sul video test set (mAP50 / mAP50-95 / recall small, medium, large):

| modello | mAP50 | mAP50-95 | recall small / medium / large |
|---|---|---|---|
| baseline monoclasse | 0,726 | 0,502 | 0,15 / 0,74 / 0,86 |
| COCO completo REPEAT=1 | 0,501 | 0,320 | 0,00 / 0,33 / 0,66 |
| COCO completo REPEAT=5 | 0,530 | 0,365 | 0,00 / 0,40 / 0,67 |
| subset 20000_4000 REPEAT=1 | 0,540 | 0,385 | 0,01 / 0,43 / 0,68 |

Il video test set ha 120 immagini e 323 box, e ogni run ha un solo seme: differenze di 0,01–0,04 sono nel rumore. Sull'holdout tutti i modelli stanno vicino alla baseline (mAP50 0,96–0,97), quindi non discrimina. Il divario da chiudere è sulle medie e sulle piccole.
