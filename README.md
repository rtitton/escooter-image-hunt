# escooter-image-hunt

Dataset e modelli per il **rilevamento di monopattini elettrici (e-scooter)** con YOLO (Ultralytics). Il modello finale deve riconoscere le 80 classi COCO e in più la classe **escooter** (indice 80), partendo da pesi pretrained su COCO.

Il repo contiene due filoni di lavoro distinti:

1. **Costruire e curare il nostro dataset escooter** a partire da dataset pubblici di Roboflow: download, deduplica, selezione automatica, revisione manuale, annotazione delle classi COCO, split train/valid.
2. **Addestrare e valutare i modelli**: unione con COCO (o un suo subset), training, valutazione su test set indipendenti, confronto fra modelli, milestone.

## Stato attuale

- Dataset finale: `data/processed/union_reviewed_coco`, 2085 immagini, classi COCO 0-79 più escooter 80, revisionato a mano due volte. Split in `data/processed/union_reviewed_coco_split` (1812 train, 270 valid).
- Baseline yolo26n (solo escooter, 640): mAP50 0,964 sul valid, 0,726 sul video test set, 0,972 sul kickboard. Il punto debole sono gli escooter piccoli (<32 px).
- Training con COCO + nostro split: le classi COCO restano intatte, ma la escooter peggiora per diluizione. Prossimo esperimento: ripetere le nostre immagini nel merge (`REPEAT=5`), poi eventualmente un subset di COCO.
- Numeri, metodo e limiti delle misure: [VALUTAZIONI.md](VALUTAZIONI.md).

## Struttura del repo

| Percorso | Contenuto |
|---|---|
| `scripts/dataset/` | Script per creare e curare il dataset escooter |
| `scripts/training/` | COCO (download, subset, merge), training, valutazione, inferenze, milestone |
| `scripts/config.py`, `scripts/.env` | Configurazione condivisa (percorsi, soglie, chiave Roboflow); `.env` non è nel repo, si parte da `scripts/.env.template` |
| `data/` | Dati (non nel repo): `raw/` dataset scaricati, `interim/` versioni deduplicate, `processed/` dataset derivati e split, `coco/` COCO 2017 |
| `models/` | Milestone di training con README e report di valutazione (pesi esclusi) |
| `runs/` | Area di lavoro di Ultralytics (prove, run da riprendere); non nel repo |
| `inferences/` | Frame dei video con le box dei modelli e fogli di confronto; non nel repo, tranne il README |
| `datasets_to_download.csv` / `.md` | Elenco dei dataset Roboflow sorgente e parametri di selezione per ciascuno |
| `PIPELINE.md` | Descrizione dettagliata della pipeline e indice di tutti gli script |
| `VALUTAZIONI.md` | Registro delle valutazioni e dei confronti fra modelli |
| `backup*.sh` | Backup di dataset intermedi |

## Il dataset escooter

Fonti: dataset pubblici di Roboflow (licenze per lo più CC BY 4.0), con i problemi tipici: bassa risoluzione, immagini molto simili fra loro, oggetto in primo piano con poca varietà di classi, pochi escooter piccoli. Il criterio di qualità per scegliere le immagini è la parte difficile del progetto; il resto è automatismo.

Passi, in sintesi (dettagli e comandi in [PIPELINE.md](PIPELINE.md)):

1. **Download** da Roboflow in formato YOLO (`download_dataset.py`, `download_batch.py`); il raw non si modifica.
2. **Deduplica delle augmentation** di Roboflow e conversione dei poligoni in bounding box (`dedupe_augmented.py`).
3. **Selezione automatica** delle candidate con criteri di qualità (`select_images.py`).
4. **Dataset di unione** con la sola classe escooter, rimappata a 80 (`build_union_dataset.py`).
5. **Revisione manuale** con interfaccia web (`review_app.py`), poi materializzazione del dataset finale.
6. **Annotazione delle classi COCO** con un modello grande pretrained (`annotate_coco_classes.py`) e revisione manuale.
7. **Split train/valid** senza fughe di informazione fra i due lati (`split_dataset.py`, `split_review_app.py`).

Convenzioni sulle box: solo la parte visibile (niente box amodali); per le classi COCO si seguono le convenzioni di COCO (per esempio `truck` esclude pick-up e furgoni); le box non escono dall'immagine.

## Training, valutazione, confronto

Dalla radice del repo, con l'ambiente `uv`:

```bash
# COCO in formato Ultralytics in data/coco (circa 20 GB)
scripts/training/download_coco.sh

# subset di COCO per classi e numero massimo di immagini (opzionale)
python3 scripts/training/make_coco_subset.py --coco-dir data/coco --classes traffic \
    --max-images 30000 --out-dir data/processed/coco_subset_30000

# liste e data.yaml di COCO (o del subset, COCO_SUBSET_DIR=...) + nostro split (nostre immagini ripetute REPEAT volte)
REPEAT=5 scripts/training/build_coco_merged_lists.sh

# training sul dataset così costruito
DATA_YAML=data/processed/coco_escooter_merge_rep5/data.yaml scripts/training/train_yolo26_coco_escooter.sh

# statistiche del dataset (ordinamento: frequency, id, name; CSV opzionale)
python3 scripts/training/dataset_stats.py data/processed/coco_escooter_merge_rep5/data.yaml \
    --sort frequency --csv statistiche.csv

# baseline solo escooter
scripts/training/train_yolo26n_baseline.sh

# valutazione su test set indipendenti (recall per taglia dell'oggetto)
python3 scripts/training/eval_testset.py --help

# inferenza sui video e fogli di confronto fra modelli
python3 scripts/training/infer_videos.py /mnt/x/media
python3 scripts/training/contact_sheets.py

# a run conclusa: copia in models/ come milestone
python3 scripts/training/export_milestone.py --help
```

Test set indipendenti: `video_testset_final` (frame da video, molti oggetti piccoli) e `holdout_kickboard_a75qx_final` (oggetti grandi). Entrambi sono in `data/processed/`.

Il training usa la GPU (`--device 0`); le macchine a disposizione sono auros, con una GTX 1050 Ti da 4 GB (per prove con modelli piccoli), e una H100 per i training completi. `train_yolo.py --resume` riprende un run interrotto.

## Ambiente

Python 3.12 gestito con [uv](https://docs.astral.sh/uv/): `pyproject.toml` è quello in uso sulla macchina corrente; `pyproject-auros.toml` e `pyproject-h100.toml` sono le varianti per auros (GTX 1050 Ti) e per la H100. Gli script si lanciano dalla radice del repo. La configurazione sta in `scripts/.env` (copiare da `scripts/.env.template` e inserire la chiave `RF_API_KEY` di Roboflow per i download).

## Dove leggere di più

- [PIPELINE.md](PIPELINE.md): la pipeline passo per passo, i comandi in sequenza e l'indice di tutti gli script.
- [VALUTAZIONI.md](VALUTAZIONI.md): risultati, analisi del leakage, confronti fra modelli, limiti e prossimi passi.
- [models/README.md](models/README.md): come funzionano le milestone.
- [datasets_to_download.md](datasets_to_download.md): il significato delle colonne del CSV dei dataset sorgente.
