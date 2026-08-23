# Pipeline dati — stato attuale

Descrive la sequenza di script che porta dai dataset pubblici Roboflow al
dataset di unione e al relativo campione di controllo visivo. Per il
contesto e gli obiettivi del progetto vedi [README.md](README.md); per i
criteri di selezione vedi
[claude-instruct-01-automatic-image-selection.md](claude-instruct-01-automatic-image-selection.md).

## Panoramica

```
data/datasets_to_download.csv
        │  (download_batch.py)
        ▼
data/raw/<id>/                    ── un dataset Roboflow per id, invariato
        │  (dedupe_augmented.py, usa bbox_convert.py)
        ▼
data/interim/<id>-dedup/          ── deduplicato da augmentation, poligoni→bbox
        │  (select_images.py)
        ▼
data/selected_images.txt          ── elenco path delle immagini candidate
        │  (build_union_dataset.py)
        ▼
data/processed/union/             ── immagini + label (solo classe escooter = 80)
        │  (visual_check_sample.py)
        ▼
data/processed/union_review_sample/  ── campione con bbox disegnata, per QA manuale
```

Ogni passo aggiorna anche `data/datasets.json` (indice macchina) e
`data/README.md` (vista leggibile), tranne gli ultimi due che lavorano sul
dataset di unione aggregato.

## 1. Download di un dataset — `download_dataset.py`

Scarica un progetto Roboflow (formato YOLO) in `data/raw/<project>-v<versione>/`.

```
python3 scripts/download_dataset.py --workspace <ws> --project <project> \
    [--version N] [--escooter-class-names "nome1|nome2"]
```

- se `--version` è omesso, sceglie la versione più recente senza augmentation
  (o la più recente in assoluto, se tutte hanno augmentation)
- rileva il formato annotazioni (`bbox`, `poligono` o `misto`) — non scarta
  più i dataset a poligono, verranno convertiti allo stadio successivo
- se sono passati `--escooter-class-names`, verifica che quei nomi esistano
  tra le classi del `data.yaml` scaricato e segnala quelli mancanti
- aggiorna `data/datasets.json` con id, sorgente, classi, conteggio immagini
  raw, formato annotazioni, nomi classe escooter (e quelli eventualmente
  mancanti)

## 1b. Download batch da CSV — `download_batch.py`

Automatizza il passo 1 (download) + il passo 2 (dedupe) per più dataset,
leggendo `data/datasets_to_download.csv` (colonne: `status`, `workspace_id`,
`project_id`, `escooter_class_name` con nomi separati da `|`, `notes`).

```
python3 scripts/download_batch.py
```

Elabora solo le righe con `status=todo`. Una riga passa a `status=downloaded`
solo se download + validazione classi + dedupe vanno tutti a buon fine;
altrimenti resta `todo` e viene segnalato l'errore, così può essere corretta
e rilanciata.

## 2. Deduplica augmentation + conversione poligoni — `dedupe_augmented.py`

Costruisce la versione "interim" di un dataset senza toccare il raw.

```
python3 scripts/dedupe_augmented.py --dataset-dir data/raw/<id> [--out-dir ...]
```

Per ogni gruppo di varianti augmentate generate da Roboflow (stesso
nome-base, suffisso `.rf.<hash>`), tiene una sola immagine — preferendo,
quando possibile, quella senza segni di rotazione (rilevati dal padding a
cuneo ai bordi, distinto dal letterboxing) — e copia la coppia
immagine+label in `data/interim/<id>-dedup/`. Le annotazioni a poligono
vengono convertite nel bounding box minimo che le contiene
(`scripts/bbox_convert.py`). Scrive un log per esecuzione in
`data/logs/<id>.log` con l'elenco degli scarti e delle conversioni, e
aggiorna l'indice (`images_dedup`, `dedup_dir`, `converted_polygon_annotations`).

## 3. Selezione delle immagini candidate — `select_images.py`

Applica i criteri di qualità (vedi
[claude-instruct-01-automatic-image-selection.md](claude-instruct-01-automatic-image-selection.md))
su tutti i dataset deduplicati registrati nell'indice.

```
python3 scripts/select_images.py [--stage cheap|dedup|variety|all] [--limit N]
```

Tre stadi in sequenza, ciascuno eseguibile isolatamente per test incrementali:

1. **filtri economici** — scarta immagini senza istanze escooter, con
   un'istanza escooter "primo piano" (area ≥ 80% dell'immagine), o troppo
   piccole (< 160.000 px totali)
2. **dedup cross-dataset** — calcola il perceptual hash di ogni immagine
   sopravvissuta e raggruppa (union-find, a blocchi per contenere la
   memoria) quelle a distanza di Hamming ≤ 5; per ogni gruppo tiene
   l'immagine con più bounding box totali
3. **filtro varietà** — scarta le immagini in cui un modello Ultralytics
   pretrained su COCO (`yolo11x.pt`, batch da 16, GPU se disponibile) non
   rileva nessuna istanza di nessuna classe COCO

Scrive `data/selected_images.txt` (un path per riga, relativo a
`data/interim/`, nel formato `<dataset_id>/<split>/images/<file>`) e un log
degli scarti con il motivo in `data/logs/select_images.log`.

Sull'ultimo run completo: 9638 immagini di partenza → 7547 dopo i filtri
economici → 7278 dopo la dedup cross-dataset → **6160 candidate finali**.

## 4. Costruzione del dataset di unione — `build_union_dataset.py`

Copia le immagini candidate in un unico dataset piatto, tenendo solo la
classe escooter rimappata a id `80` (le eventuali altre classi dei dataset
sorgente vengono scartate — le classi COCO saranno annotate in un passo
successivo, non ancora implementato, con un modello pretrained di grandi
dimensioni).

```
python3 scripts/build_union_dataset.py [--out-dir data/processed/union] [--limit N]
```

- i nomi dei file di destinazione sono prefissati con l'id del dataset
  sorgente (`<dataset_id>__<nome-file>`) per evitare collisioni
- gestisce correttamente i dataset con più nomi di classe per l'escooter
  (es. `electric-scooter-dpwkl-v1`, che ne ha due): tutti vengono unificati
  sotto la classe 80
- output: `data/processed/union/images/`, `data/processed/union/labels/`,
  log in `data/logs/build_union_dataset.log`

Sull'ultimo run completo: 6160/6160 candidate copiate (nessuna scartata per
assenza di istanze escooter).

## 5. Campione per controllo visivo — `visual_check_sample.py`

Esporta un campione casuale del dataset di unione con la bbox disegnata, per
intercettare a colpo d'occhio i problemi più macroscopici (box palesemente
sbagliate, immagini corrotte, ecc.).

```
python3 scripts/visual_check_sample.py [-n 150]
```

Ad ogni esecuzione la cartella `data/processed/union_review_sample/` viene
svuotata e ripopolata con un nuovo campione casuale (nessun seed fisso):
per rigenerare il campione basta rilanciare lo script.

## Stato e prossimi passi

- fatto: download (singolo e batch), deduplica + conversione poligoni,
  selezione a 3 stadi, costruzione del dataset di unione, campione di QA
  visiva
- da fare: annotazione delle classi COCO sul dataset di unione con un
  modello Ultralytics pretrained di grandi dimensioni (vedi README.md,
  "Aspetti pratici"); split train/valid/test del dataset di unione (non
  ancora definito — i candidati non ereditano lo split del dataset sorgente)
