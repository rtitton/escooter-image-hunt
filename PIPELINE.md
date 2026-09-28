# Pipeline dati — stato attuale

- [Pipeline dati — stato attuale](#pipeline-dati--stato-attuale)
  - [Panoramica](#panoramica)
  - [Comandi in sequenza (avvio da zero)](#comandi-in-sequenza-avvio-da-zero)
  - [1. Download di un dataset — `download_dataset.py`](#1-download-di-un-dataset--download_datasetpy)
  - [1b. Download batch da CSV — `download_batch.py`](#1b-download-batch-da-csv--download_batchpy)
  - [1c. Verifica sincronizzazione CSV/indice — `check_dataset_sync.py`](#1c-verifica-sincronizzazione-csvindice--check_dataset_syncpy)
  - [2. Deduplica augmentation + conversione poligoni — `dedupe_augmented.py`](#2-deduplica-augmentation--conversione-poligoni--dedupe_augmentedpy)
  - [3. Selezione delle immagini candidate — `select_images.py`](#3-selezione-delle-immagini-candidate--select_imagespy)
  - [3b. Campione dall'indice immagini — `build_index_sample.py`](#3b-campione-dallindice-immagini--build_index_samplepy)
  - [4. Costruzione di un dataset da un elenco di candidate — `build_union_dataset.py`](#4-costruzione-di-un-dataset-da-un-elenco-di-candidate--build_union_datasetpy)
  - [5. Campione per controllo visivo — `build_visual_check_sample.py`](#5-campione-per-controllo-visivo--build_visual_check_samplepy)
  - [5b. Immagini annotate per dataset — `build_bydataset_annotated.py`](#5b-immagini-annotate-per-dataset--build_bydataset_annotatedpy)
  - [6. Selezione manuale finale — `review_app.py`](#6-selezione-manuale-finale--review_apppy)
  - [7. Materializzazione del dataset finale — `materialize_union_reviewed.py`](#7-materializzazione-del-dataset-finale--materialize_union_reviewedpy)
  - [8. (opzionale) Dedup pHash a posteriori — `dedupe_phash.py`](#8-opzionale-dedup-phash-a-posteriori--dedupe_phashpy)
  - [9. Annotazione classi COCO — `annotate_coco_classes.py`](#9-annotazione-classi-coco--annotate_coco_classespy)
  - [9b. Revisione manuale e materializzazione finale](#9b-revisione-manuale-e-materializzazione-finale)
  - [10. Split train/valid — `split_dataset.py`](#10-split-trainvalid--split_datasetpy)
  - [Indice degli script](#indice-degli-script)

Descrive la sequenza di script che porta dai dataset pubblici Roboflow al
dataset di unione, alla revisione automatica e manuale. Per il contesto e
gli obiettivi del progetto vedi [README.md](README.md); per i criteri di
selezione vedi
[claude-instruct-01-automatic-image-selection.md](claude-instruct-01-automatic-image-selection.md).

## Panoramica
Il file `datasets_to_download.csv`, nella root del progetto, contiene l'elenco dei dataset Roboflow da trattare, con il relativo stato (downloaded, todo, ignore, ecc.) e i parametri di selezione per-dataset, e determina il contenuto della directory `data`. Il tracciato è documentato in [datasets_to_download.md](datasets_to_download.md).  
La directory `data` contiene i dataset scaricati e tutto il materiale relativo alle elaborazioni successive della pipeline.  

`datasets_to_download.csv` è versionato in git ma resta specifico dell'istanza; la directory `data` è invece in `.gitignore`. Per una nuova istanza si crea `datasets_to_download.csv` a partire dall'elenco generale dei dataset Roboflow selezionati `roboflow-datasets_to_download.csv`.


```
datasets_to_download.csv
        │  (download_batch.py)
        ▼
data/raw/<id>/                    ── un dataset Roboflow per id, invariato
        │  (dedupe_augmented.py, usa bbox_convert.py)
        ▼
data/interim/<id>-dedup/          ── deduplicato da augmentation, poligoni→bbox
        │  (select_images.py)
        ├──────────────────────────────────────────────┐
        ▼                                               ▼
data/selected_images.txt                 data/flagged_rider_contamination.txt
(candidate)                              (bbox con conducente incluso)
        │  (build_union_dataset.py)                     │  (build_union_dataset.py
        ▼                                                │   --candidates-file ...)
data/processed/union/                                    ▼
(immagini + label, classe 80)                data/processed/rider_review/
        │  (visual_check_sample.py)         (da correggere manualmente)
        ▼
data/processed/union_review_sample/  ── campione con bbox disegnata, per QA visiva
        │
        ▼
review_app.py  ── selezione manuale finale (s/n) → data/processed/union/review_decisions.json
        │  (materialize_union_reviewed.py)
        ▼
data/processed/union_reviewed/            ── solo le immagini "select", pronte all'uso
        │  (dedupe_phash.py, opzionale)
        ▼
data/processed/union_reviewed_phashdedup/ ── dedup pHash a posteriori, soglia libera
        │  (annotate_coco_classes.py)
        ▼
data/processed/union_reviewed_coco/       ── + classi COCO annotate, pronto per il training
```

Ogni passo aggiorna anche `data/datasets.json` (indice macchina) e
`data/README.md` (vista leggibile), tranne gli ultimi che lavorano sul
dataset di unione aggregato.

## Comandi in sequenza (avvio da zero)

Riepilogo eseguibile della pipeline completa, dal download alla revisione
manuale finale (vedi le sezioni sotto per il dettaglio di ogni passo e le
opzioni disponibili).

```bash
PYTHONCMD=uv run python3
# 1. Popolare datasets_to_download.csv (righe enabled=1, download=1), poi
#    download + dedupe in batch per tutte quelle righe (chiama
#    internamente download_dataset.py e dedupe_augmented.py per ognuna:
#    non vanno lanciati a mano in questo flusso)
$PYTHONCMD scripts/download_batch.py

# 1c. (opzionale, utile lavorando su più macchine) verifica che i dataset
#     previsti dal CSV siano tutti presenti nell'indice/su disco locali
$PYTHONCMD scripts/check_dataset_sync.py

# 2. Selezione delle immagini candidate su tutti i dataset deduplicati registrati
$PYTHONCMD scripts/select_images.py

# 3. Costruzione del dataset di unione dalle candidate (classe escooter rimappata a 80)
$PYTHONCMD scripts/build_union_dataset.py

#    ...e delle immagini flaggate per conducente incluso, per la correzione manuale
$PYTHONCMD scripts/build_union_dataset.py --candidates-file data/flagged_rider_contamination.txt \
    --out-dir data/processed/flagged_rider_contamination

# 3b. (opzionale, indipendente dal passo 3) estrae le candidate mantenendo i
#     dataset sorgente separati e le classi originali (senza remap a 80),
#     in data/interim/<id>-selected/ — utile per ispezionare la selezione
#     dataset per dataset
$PYTHONCMD scripts/build_selected_datasets.py

# 4. (opzionale) Campione per controllo visivo del dataset di unione
$PYTHONCMD scripts/build_visual_check_sample.py

# 5. Web app per review manuale di un dataset in formato yolo con struttura base/images base/labels. (http://localhost:8765)
$PYTHONCMD scripts/review_app.py

# 6. Materializza la versione finale del dataset (solo le immagini "select")
$PYTHONCMD scripts/materialize_union_reviewed.py

# 6b. (opzionale) Dedup pHash a posteriori del dataset finale, soglia scelta da riga di comando
$PYTHONCMD scripts/dedupe_phash.py [soglia]

# 7. Annotazione delle classi COCO sul dataset finale (oltre alla classe escooter già presente)
$PYTHONCMD scripts/annotate_coco_classes.py
```

`download_dataset.py` e `dedupe_augmented.py` (sezioni 1 e 2 sotto) restano
utili per aggiungere o rigenerare un singolo dataset fuori dal CSV, ma non
fanno parte del giro "da zero": in quel caso il punto di ingresso è sempre
`download_batch.py`.

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
leggendo `datasets_to_download.csv` (tracciato completo in
[datasets_to_download.md](datasets_to_download.md)). Usa le colonne `enabled`,
`download`, `workspace_id`, `project_id`, `version`, `escooter_class_name`.

- `version` vuoto: usa il comportamento di default di `download_dataset.py`
  (versione più recente senza augmentation); la versione effettivamente
  scaricata viene stampata (banner in `download_dataset.py`) ma non scritta
  nel CSV — fissarla nella colonna `version` resta a discrezione manuale
- `version` valorizzato: scarica esattamente quella versione

```
python3 scripts/download_batch.py
```

Elabora solo le righe con `enabled=1` e `download=1`. Questo script non
modifica mai il CSV: si limita a segnalare a schermo, per ogni riga, se
download + validazione classi + dedupe sono andati a buon fine o se c'è
stato un errore (in tal caso va corretto a mano quanto necessario e la riga
va rilanciata).

## 1c. Verifica sincronizzazione CSV/indice — `check_dataset_sync.py`

Dato che `data/` non è versionata, lavorando su più macchine ogni istanza
del progetto può avere scaricato solo un sottoinsieme dei dataset previsti
dal CSV locale. Questo script confronta `datasets_to_download.csv` con
`data/datasets.json` (e con `data/raw/`) e segnala:

- dataset `enabled=1` con `version` fissata ma assenti dall'indice (da
  scaricare su questa macchina)
- dataset presenti nell'indice ma la cui cartella in `data/raw/` non c'è più
- dataset `enabled=1` senza `version` fissata, presenti o assenti
- dataset presenti nell'indice ma non più corrispondenti a nessuna riga
  `enabled=1` del CSV (es. dopo un allineamento manuale delle versioni)

```
python3 scripts/check_dataset_sync.py
```

Uscita `0` se non ci sono dataset mancanti o disallineati, `1` altrimenti.

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

Nota: l'euristica anti-rotazione qui (`looks_rotated`, basata sul padding
nero ai bordi) si è rivelata inaffidabile per rilevare immagini
ruotate/flippate in generale — vedi il filtro conducente allo stadio 3, che
per questo controlla sempre tutte le orientazioni invece di fare affidamento
su un'euristica di pre-filtro.

## 3. Selezione delle immagini candidate — `select_images.py`

Applica i criteri di qualità (vedi
[claude-instruct-01-automatic-image-selection.md](claude-instruct-01-automatic-image-selection.md))
su tutti i dataset deduplicati registrati nell'indice.

```
python3 scripts/select_images.py [--stage cheap|temporal|dedup|variety|all] [--temporal-dedup] [--limit N]
```

Quattro stadi in sequenza più uno opzionale (i primi eseguibili isolatamente
per test incrementali; il filtro conducente è parte dello stadio `variety`):

1. **filtri economici** — scarta immagini senza istanze escooter, con
   un'istanza escooter "primo piano" (area ≥ 40% dell'immagine), o troppo
   piccole (< 160.000 px totali)
1-bis. **dedup temporale** *(opzionale, solo con `--temporal-dedup` o
   `--stage temporal`)* — parecchi dataset sorgente sono campionamenti fitti
   di poche riprese video (nomi tipo `frame_00000`, `frame_00010`). Dentro
   ogni gruppo `(dataset, split, clip)` — `clip` e indice di frame ricavati
   dal nome file — ordinato per indice, scarta un frame solo se è visivamente
   vicino all'ultimo frame tenuto (Hamming del pHash < `TEMPORAL_KEEP_DISTANCE`,
   default 10) e a non più di `TEMPORAL_MAX_GAP` indici da esso (default 60),
   tenendo comunque il primo e l'ultimo del gruppo. Gruppi sotto
   `TEMPORAL_MIN_SEQ` frame (default 5) e nomi non numerati restano intatti.
   Riusa la cache dei perceptual hash dello stadio dedup. Nota: siccome la
   dedup cross-dataset è un clustering a catena, questo stadio può far
   *aumentare* di poche unità le candidate finali (rimuovendo un frame-ponte
   si spezza il suo cluster pHash) — effetto atteso e difendibile
2. **dedup cross-dataset** — calcola il perceptual hash di ogni immagine
   sopravvissuta e raggruppa (union-find, a blocchi per contenere la
   memoria) quelle a distanza di Hamming ≤ `PHASH_DISTANCE_THRESHOLD`; per
   ogni gruppo tiene l'immagine con più bounding box totali
3. **filtro varietà** — scarta le immagini in cui un modello Ultralytics
   pretrained su COCO (`yolo11l.pt`, batch da 16, GPU se disponibile) rileva
   meno di `VARIETY_MIN_INSTANCES` istanze di classi COCO
   nell'orientazione originale
4. **filtro conducente incluso** — alcuni dataset sorgente annotano l'intera
   persona invece del solo monopattino (es. `electric-scooter-dpwkl-v1`, ma
   non solo). Stima quanta parte di ogni bbox escooter è spiegata da una
   detection "persona" del modello COCO, controllando **tutte e 4 le
   orientazioni** (0/90/180/270°) — alcune immagini sorgente sono
   ruotate/flippate e un rilevatore addestrato su foto diritte spesso manca
   la persona in quell'orientazione; un'euristica più economica basata sul
   solo padding nero si è rivelata inaffidabile su questi casi. Se anche una
   sola bbox dell'immagine è contaminata, l'intera immagine viene esclusa
   dalle candidate (escludere solo la bbox lascerebbe un monopattino
   visibile ma non annotato) e finisce invece in
   `data/flagged_rider_contamination.txt`, da correggere manualmente in un
   secondo momento — non viene buttata, perché ha superato tutti gli altri
   criteri di qualità.

**Override per-dataset.** Le soglie `CLOSEUP_AREA_THRESHOLD` e `FARAWAY_AREA_THRESHOLD` (stadio 1),
`PHASH_DISTANCE_THRESHOLD` (stadio 2) e `VARIETY_MIN_INSTANCES` (stadio 3)
valgono di default quelle di `scripts/.env`, ma si possono sovrascrivere per
singolo dataset nelle colonne omonime di `datasets_to_download.csv`
(`variety_min_instances`, `closeup_area_threshold`, `faraway_area_threshold`,
`phash_distance_threshold`);
cella vuota o `default` = valore di `.env`. Serve a trattare a parte sorgenti
particolari — p.es. footage con escooter piccoli (varietà più permissiva,
closeup più alto, faraway più basso) o molto ripetitiva (pHash più stretto
per non collassarla).
Nella dedup cross-dataset la soglia di una coppia di immagini di dataset
diversi è la **più stretta** delle due. Gli override attivi compaiono nel
report di `report_image_index.py`.

Scrive `data/selected_images.txt` (candidate), `data/flagged_rider_contamination.txt`
(da rivedere) e `data/flagged_area_threshold.txt` (scartate per soglia di
area, solo diagnostico), tutti un path per riga relativo a `data/interim/`
nel formato `<dataset_id>/<split>/images/<file>`, più un log degli scarti
con il motivo in `data/logs/select_images.log`. Scrive anche
`data/image_index.json`: un indice con, per ogni immagine esaminata,
percorso completo, dimensioni ed eventuale decisione di esclusione (stadio
e motivo) — vedi sezione 3b per come campionarlo.

Sull'ultimo run completo: 9638 immagini di partenza → 5999 dopo i filtri
economici → 5643 dopo la dedup cross-dataset → 2772 candidate finali (1955
scartate per conducente incluso, finite in coda di revisione).

## 3b. Campione dall'indice immagini — `build_index_sample.py`

Costruisce un campione di immagini a partire da `data/image_index.json`,
filtrando su condizioni a piacere sugli attributi di ciascuna voce
(`dataset_id`, `split`, `image_path`, `width`, `height`, `excluded`,
`exclusion_stage`, `exclusion_reason`), e le salva in una cartella con le
bounding box escooter disegnate sopra. Utile per ispezionare a occhio un
sottoinsieme scelto in base alle decisioni di `select_images.py` (es. solo
le scartate per soglia di area, solo quelle di un dataset specifico) senza
rilanciare la pipeline.

```
python3 scripts/build_index_sample.py --filter "<espressione Python>" --out-dir <cartella> [-n 150]
```

Il filtro è un'espressione Python valutata su ogni voce dell'indice, coi
suoi campi disponibili come variabili, es.:
`--filter "exclusion_stage == 'cheap' and 'lontana' in (exclusion_reason or '')"`.
Con `-n 0` copia tutte le immagini che soddisfano il filtro invece di
campionarne un sottoinsieme casuale. Ad ogni esecuzione la cartella di
output viene svuotata e ripopolata.

## 4. Costruzione di un dataset da un elenco di candidate — `build_union_dataset.py`

Copia le immagini di un elenco (di norma `data/selected_images.txt`) in una
cartella piatta, tenendo solo la classe escooter rimappata a id `80` (le
eventuali altre classi dei dataset sorgente vengono scartate — le classi
COCO vengono annotate in un passo successivo, sul dataset finale, da
`annotate_coco_classes.py` con un modello pretrained di grandi dimensioni;
v. sezione 9).

```
python3 scripts/build_union_dataset.py [--candidates-file ...] [--out-dir ...] [--limit N]
```

- i nomi dei file di destinazione sono prefissati con l'id del dataset
  sorgente (`<dataset_id>__<nome-file>`) per evitare collisioni
- gestisce correttamente i dataset con più nomi di classe per l'escooter
  (es. `electric-scooter-dpwkl-v1`, che ne ha due): tutti vengono unificati
  sotto la classe 80
- con `--candidates-file data/flagged_rider_contamination.txt --out-dir
  data/processed/rider_review` copia invece le immagini scartate per
  conducente incluso, mantenendo le bbox originali (comprese quelle
  "sbagliate") come base di partenza per la correzione manuale
- output di default: `data/processed/union/images/`,
  `data/processed/union/labels/`, log in
  `data/logs/build_union_dataset-<nome-elenco>.log`

Sull'ultimo run completo: 2772/2772 candidate copiate in `data/processed/union/`,
1955/1955 flaggate copiate in `data/processed/rider_review/`.

## 5. Campione per controllo visivo — `build_visual_check_sample.py`

Esporta un campione casuale del dataset di unione con la bbox disegnata, per
intercettare a colpo d'occhio i problemi più macroscopici (box palesemente
sbagliate, immagini corrotte, ecc.) — è così che è stato scoperto il
problema del conducente incluso, incluso il caso delle immagini ruotate.

```
python3 scripts/visual_check_sample.py [-n 150]
```

Ad ogni esecuzione la cartella `data/processed/union_review_sample/` viene
svuotata e ripopolata con un nuovo campione casuale (nessun seed fisso):
per rigenerare il campione basta rilanciare lo script.

## 5b. Immagini annotate per dataset — `build_bydataset_annotated.py`

Esporta la selezione finale (di norma il dataset di unione,
`data/processed/union/`) in `data/processed/bydataset/`, con una cartella
per ogni dataset sorgente (ricavato dal prefisso `<dataset_id>__` del nome
file) contenente **tutte** le immagini di quel dataset con le sole bounding
box escooter disegnate sopra. Utile per rivedere dataset per dataset la
qualità delle annotazioni sull'intera selezione, non su un campione.

```
python3 scripts/build_bydataset_annotated.py [-d DIR] [-o DIR] \
    [--decisions-file FILE] [--include-discard] [--limit N]
```

Ad ogni esecuzione la cartella di output viene svuotata e ripopolata. Con
`--decisions-file data/processed/union/review_decisions.json` filtra la
selezione con le decisioni di `review_app.py` (di default tiene solo
`select` e `reserve`; con `--include-discard` anche gli scarti); senza,
esporta tutte le immagini della sorgente.

## 6. Selezione manuale finale — `review_app.py`

Applicazione locale (solo libreria standard, nessuna dipendenza aggiuntiva)
per una revisione manuale immagine-per-immagine del dataset di unione,
prima di considerarlo definitivo.

```
python3 scripts/review_app.py [--port 8765]
```

Apre un server su `http://localhost:<port>`: mostra un'immagine alla volta
con le bbox disegnate (canvas HTML) e registra la decisione con un tasto —
`s` seleziona, `n` scarta, frecce per navigare senza decidere, `end` per
saltare alla prima immagine non ancora revisionata, backspace per
cancellare la decisione corrente. Le bbox sono anche modificabili
direttamente sul canvas (creare/spostare/ridimensionare/eliminare, con
ripristino dell'originale). Barra di filtro/ordinamento per dataset
sorgente, stato e pHash. Ogni decisione è salvata subito in
`data/processed/union/review_decisions.json`: la sessione si può
interrompere e riprendere quando si vuole, ripartendo dalla prima immagine
ancora senza decisione.

## 7. Materializzazione del dataset finale — `materialize_union_reviewed.py`

Copia in una cartella a parte solo le immagini con decisione `select` in
`review_decisions.json`, insieme ai rispettivi label — che riflettono già
le eventuali modifiche alle bbox fatte durante la review, essendo scritti
subito su disco da `review_app.py`. A differenza di
`build_bydataset_annotated.py` (bbox disegnate, cartelle separate per
dataset sorgente, pensato per il controllo visivo) qui l'output è un
dataset YOLO pronto all'uso: stessa struttura `images/`+`labels/` del
dataset sorgente, nomi file invariati.

```
python3 scripts/materialize_union_reviewed.py
    [--source-dir DIR] [--decisions-file FILE] [--out-dir DIR] [--limit N]
```

- default: sorgente `data/processed/union/`, decisioni
  `<source-dir>/review_decisions.json`, output
  `data/processed/union_reviewed/`
- avvisa a console se restano immagini del sorgente senza decisione (review
  non completa) senza per questo bloccare l'esecuzione
- stampa per ogni dataset sorgente quante immagini sono state selezionate
  sul totale disponibile (`<id>: N / M immagini selezionate`)
- ad ogni esecuzione la cartella di output viene svuotata e ripopolata

## 8. (opzionale) Dedup pHash a posteriori — `dedupe_phash.py`

Deduplica per contenuto (perceptual hash) `data/processed/union_reviewed/`,
raggruppando in cluster le immagini a distanza di Hamming ≤ soglia e
tenendo per ciascun cluster solo quella con più bbox escooter annotate (a
parità, il nome file più piccolo). A differenza della dedup cross-dataset
di `select_images.py` — eseguita *prima* della selezione/review, con soglia
stretta per non perdere varietà — questo script opera *dopo* la review
manuale, con una soglia scelta liberamente da riga di comando: utile per
stringere la dedup a posteriori (es. per un dataset di validazione senza
quasi-duplicati) senza dover rifare selezione o review.

```
python3 scripts/dedupe_phash.py [soglia] [--source-dir DIR] [--out-dir DIR] [--phash-cache FILE]
```

- `soglia` (posizionale, opzionale): distanza di Hamming massima fra due
  pHash perché due immagini siano quasi-duplicate (default:
  `PHASH_DISTANCE_THRESHOLD` di `.env`, di norma 10)
- default: sorgente `data/processed/union_reviewed/`, output
  `data/processed/union_reviewed_phashdedup/`
- le immagini escluse (quelle "in più" di ogni cluster) non vengono
  cancellate: restano, con le rispettive label, in `<out-dir>/excluded/`
- riusa la cache pHash di `review_app.py` (stesso meccanismo di
  invalidazione per mtime/size), quindi è economico rilanciarlo con soglie
  diverse per confrontarle
- stampa a console le immagini escluse per dataset sorgente, il numero di
  cluster di quasi-duplicati trovati e i totali; log dettagliato (immagine
  scartata → immagine tenuta, distanza) in
  `data/logs/dedupe_phash-t<soglia>.log`

## 9. Annotazione classi COCO — `annotate_coco_classes.py`

Ultimo passo: annota tutte le classi COCO (0-79) sul dataset finale, in
aggiunta alla classe escooter (80) già presente. Usa lo stesso modello
Ultralytics pretrained di grandi dimensioni del filtro varietà
(`COCO_MODEL`, di norma `yolo11l.pt`), ma su una cache dedicata che salva
anche la confidence di ogni detection (non solo classe e bbox) a una soglia
di cattura bassa (`COCO_ANNOTATION_CAPTURE_CONF`, default 0.1): questo
separa il costo dell'inferenza — fatta una volta — dalla soglia di
confidenza effettiva (`COCO_ANNOTATION_CONF_THRESHOLD`, default 0.45,
applicata in scrittura), ritarabile senza ricalcolare nulla. La soglia
effettiva è più conservativa del default Ultralytics (0.25) perché qui le
detection diventano etichette di training permanenti: un falso positivo pesa
più di un mancato rilevamento.

Una detection di classe "sosia" del monopattino (`COCO_ESCOOTER_LOOKALIKE_
CLASSES`, default `bicycle,motorcycle,parking meter,snowboard,skateboard`)
viene scartata se la sua bbox è coperta per più di `COCO_ESCOOTER_OVERLAP_
THRESHOLD` (default 0.6, frazione di area della detection) dall'**unione**
delle bbox escooter dell'immagine: è quasi certamente lo stesso oggetto
fisico, tenuto per intero o in parte (es. quando il modello rileva solo il
pianale, come `skateboard`/`snowboard`, o solo lo stelo/manubrio, come
`parking meter` — un oggetto verticale sottile scambiato per un altro,
verificato empiricamente responsabile della maggioranza delle istanze
`parking meter` altrimenti annotate), e tenerla creerebbe due etichette
contraddittorie sulla stessa area. Due scelte deliberate rispetto a un
semplice IoU contro la bbox escooter più vicina:
- **frazione di area della detection, non IoU simmetrico**: una detection
  molto più piccola della bbox escooter (es. il solo pianale) può ricadere
  quasi per intero al suo interno pur avendo IoU basso, per la differenza
  di area — l'IoU da solo non la catturerebbe;
- **unione delle bbox escooter, non la singola più vicina**: in una fila di
  monopattini ravvicinati (dock di sharing, parcheggi fitti) una detection
  sosia può ricadere a cavallo di due bbox escooter adiacenti — containment
  basso contro ciascuna singolarmente (l'area si divide fra le due) pur
  essendo quasi interamente coperta da *qualche* escooter nel complesso.
  Caso osservato in pratica (v. `union_containment_ratio()` in
  `annotate_coco_classes.py`, calcolata per campionamento su una griglia
  nella bbox, esatta a meno dell'errore di discretizzazione).

Il controllo è **limitato alle classi sosia**, non esteso a tutte: una
prima versione di questo script scartava qualunque classe diversa da
"persona" sopra soglia, ma questo eliminava anche oggetti reali chiaramente
distinti (un'auto, una borsa, una panchina sullo sfondo...) il cui bbox
ricadeva per intero in quello, spesso ampio, del monopattino solo per
prospettiva/profondità — non perché coincidessero fisicamente con esso (il
caso descritto in `todo.md`: "posso avere un oggetto dietro il
monopattino"). "Persona" non è comunque mai nella lista: il conducente è
una detection legittima da annotare, non un duplicato.

Anche ristretto alle classi sosia il criterio resta geometrico e non
infallibile — verificato empiricamente su un campione casuale: un oggetto
sosia reale (tipicamente una bici vera) parcheggiato proprio dietro/accanto
al monopattino può avere la stessa containment di un vero doppione, e non
esiste una soglia che separi in modo pulito i due casi (i falsi scarti non
si concentrano a containment basso). Per questo ogni detection scartata per
overlap sosia viene **segnalata, non solo buttata via**, in
`FLAGGED_COCO_OVERLAP_PATH` (default `data/flagged_coco_overlap.txt`), con
classe, confidence e containment. `--export-flagged-sample N` esporta un
campione casuale con le bbox disegnate (escooter in rosso, scartata in
arancione) per la revisione visiva; le immagini per cui si decide di
recuperare le detection sosia vanno aggiunte, un nome per riga, a
`COCO_OVERLAP_RESCUE_PATH` (default `data/coco_overlap_rescue.txt`) — al
run successivo lo scarto per overlap non viene applicato per quelle
immagini, senza rifare l'inferenza (riusa la cache).

Indipendentemente da confidenza e overlap, le detection di una classe
**implausibile in una scena esterna** (`COCO_IMPLAUSIBLE_CLASSES`, default
oggetti da interno come `toilet`, `couch`, `tv`, `sink`, elettrodomestici da
cucina/bagno — lista completa in `scripts/.env`) vengono sempre scartate:
sono quasi sempre un errore di classificazione ad alta confidenza (es. un
cestino/scatola sullo sfondo letto come `toilet`), non intercettabile
alzando la sola soglia generale senza perdere molto recall altrove. Lista
liberamente modificabile in `scripts/.env`; una stringa vuota la disattiva
in permanenza, `--no-implausible-filter` la disattiva per una singola
esecuzione.

Solo l'orientazione nativa dell'immagine (le immagini di
`union_reviewed_phashdedup` non sono ruotate/corrette).

```
python3 scripts/annotate_coco_classes.py
    [--source-dir DIR] [--out-dir DIR] [--cache-path FILE] [--model NAME]
    [--batch-size N] [--capture-conf F] [--conf-threshold F]
    [--overlap-threshold F] [--no-implausible-filter] [--refresh-cache]
    [--limit N] [--sample N] [--sample-dir DIR] [--flagged-path FILE]
    [--rescue-path FILE] [--export-flagged-sample N] [--flagged-sample-dir DIR]
```

- default: sorgente `data/processed/union_reviewed_phashdedup/`, output
  `data/processed/union_reviewed_coco/` — dataset pronto per il training
- ad ogni esecuzione la cartella di output viene svuotata e ripopolata; le
  righe classe escooter vengono rilette dal sorgente e riscritte tali e
  quali (eventuali righe COCO di un'esecuzione precedente di questo script
  vengono ignorate e ricalcolate, per idempotenza)
- `--export-flagged-sample N` esporta un campione casuale di N detection
  scartate per overlap sosia (bbox escooter in rosso, scartata in
  arancione) in `<out-dir>_flagged_sample/` (o `--flagged-sample-dir`), per
  decidere quali recuperare aggiungendole a `--rescue-path`
- `--sample N` esporta dopo l'annotazione un campione casuale di N immagini
  con le bbox disegnate (escooter in rosso, persona in verde, altre classi
  COCO in blu con nome classe e confidence) in `<out-dir>_sample/` (o
  `--sample-dir`), per una verifica visiva della soglia scelta prima di
  considerare il dataset definitivo
- stampa a console il totale di istanze annotate per classe COCO, il numero
  di detection scartate per overlap con una bbox escooter e quelle scartate
  per classe implausibile; log in `data/logs/annotate_coco_classes.log`

## 9b. Revisione manuale e materializzazione finale

Dopo l'annotazione COCO si può rivedere il dataset a occhio con
`review_app.py` (`scripts/review_app_union_reviewed_coco.sh`), che segnala
anche i casi flaggati per overlap sosia (v. sopra). Le modifiche alle bbox
sono scritte subito nei label di `union_reviewed_coco/`; per ottenere una
copia pulita con solo le immagini decise "select" (escludendo quelle
scartate durante questa revisione), si usa
`scripts/materialize_union_reviewed_coco.sh`, che richiama
`materialize_union_reviewed.py` con sorgente/output appropriati
(`UNION_REVIEWED_COCO_DIRNAME` → `UNION_REVIEWED_COCO_FINAL_DIRNAME`). Da
rilanciare ogni volta che si riprende la revisione.

## 10. Split train/valid — `split_dataset.py`

Ultimo passo prima del training: divide un dataset YOLO (`images/` +
`labels/`, tipicamente `union_reviewed_coco_final/`) in `train/` e
`valid/`, scrivendo anche un `data.yaml` pronto per Ultralytics (classi
COCO 0-79 + escooter 80, lo schema fisso del progetto — non dedotto dalle
label presenti, è sempre quello completo).

Lo split è **stratificato per dataset sorgente** (prefisso `<dataset_id>__`
nel nome file): senza, un dataset sorgente piccolo potrebbe finire per caso
quasi tutto in un solo split. All'interno di ogni dataset sorgente, le
immagini che sembrano frame consecutivi della stessa ripresa (nomi tipo
`frame_00010`, stessa euristica di `clip_and_frame()` in
`select_images.py`, duplicata qui per non dipendere da quel modulo)
vengono tenute nello stesso split: frame vicini nel tempo sono
quasi-duplicati, e separarli fra train e valid farebbe trapelare
informazione, gonfiando artificialmente le metriche di validazione. I
gruppi sono assegnati con un bilanciamento greedy per deficit (euristica
LPT — dal gruppo più grande al più piccolo, ogni gruppo va allo split che
ne ha più bisogno per avvicinarsi al proprio target): necessario perché
diversi dataset sorgente sono di fatto un'unica clip lunga (centinaia di
frame consecutivi), e un riempimento ingenuo può sbilanciare l'intero
dataset su un solo split. Quando un dataset sorgente è essenzialmente una
sola clip, finisce inevitabilmente tutto in un solo split (di norma train)
— non c'è modo di dargli rappresentanza in valid senza leakage.

Split deterministico (`--seed`, default 42), a differenza del
campionamento senza seme di `build_visual_check_sample.py`: qui la
riproducibilità conta, per confrontare run di training diversi sugli
stessi identici split.

```
python3 scripts/split_dataset.py <dataset_dir>
    [--out-dir DIR] [--valid-frac F] [--seed N] [--no-clip-grouping]
```

- default: `--out-dir <dataset_dir>_split`, `--valid-frac 0.15`
- ad ogni esecuzione la cartella di output viene svuotata e ripopolata
- stampa il totale train/valid e il dettaglio per dataset sorgente

## Indice degli script

Tutti gli script di `scripts/`, in ordine alfabetico, con una riga di descrizione. Le sezioni sopra
raccontano solo la sequenza principale (download → selezione → revisione → split); qui ci sono anche
gli script satellite (training, valutazione, mining, revisione dello split). Per i dettagli — argomenti,
default, formato dei file — vedi la docstring in cima a ciascuno script.

| Script | Descrizione |
|---|---|
| `annotate_coco_classes.py` | Annota tutte le classi COCO (0-79) sul dataset finale, oltre alla classe escooter già presente (v. [sezione 9](#9-annotazione-classi-coco--annotate_coco_classespy)). |
| `bbox_convert.py` | Libreria di conversione di annotazioni YOLO da poligono a bounding box; usata da `dedupe_augmented.py`, non uno script a sé. |
| `build_all.sh` | Lancia in sequenza `build_union_dataset.py` e `build_visual_check_sample.py` sul dataset di unione e sui due elenchi flaggati (conducente incluso, soglia di area). |
| `build_bydataset_annotated.py` | Esporta la selezione finale con le bbox disegnate, organizzata in una cartella per dataset sorgente, per un controllo visivo (v. [sezione 5b](#5b-immagini-annotate-per-dataset--build_bydataset_annotatedpy)). |
| `build_index_sample.py` | Costruisce un campione di immagini filtrato a piacere sugli attributi di `data/image_index.json`, con le bbox disegnate (v. [sezione 3b](#3b-campione-dallindice-immagini--build_index_samplepy)). |
| `build_separated_datasets.py` | Copia le immagini candidate mantenendo i dataset sorgente separati e le classi originali, senza remap a 80 — utile per ispezionare la selezione dataset per dataset. |
| `build_split_mined.sh` | Dopo la revisione manuale delle candidate di `mine_hard_examples.py`: le aggiunge al train di una copia dello split di training corrente, senza toccare l'originale. |
| `build_union_dataset.py` | Copia le immagini candidate in un dataset di unione, con la classe escooter rimappata a 80 (v. [sezione 4](#4-costruzione-di-un-dataset-da-un-elenco-di-candidate--build_union_datasetpy)). |
| `build_visual_check_sample.py` | Esporta un campione casuale del dataset di unione con le bbox disegnate, per un controllo visivo rapido (v. [sezione 5](#5-campione-per-controllo-visivo--build_visual_check_samplepy)). |
| `check_dataset_sync.py` | Confronta `datasets_to_download.csv` con `data/datasets.json` (e con i dataset presenti su disco), segnalando disallineamenti (v. [sezione 1c](#1c-verifica-sincronizzazione-csvindice--check_dataset_syncpy)). |
| `clear_selection.sh` | Cancella, chiedendo conferma, la cartella `data/processed` e i file di selezione, per ripartire da zero. |
| `compare_multiclass_curves.sh` | Lancia `compare_training_curves.py` sul confronto nano monoclasse (`escooter_only_11n`) / nano multiclasse (`multiclass_11n`). |
| `compare_training_curves.py` | Confronta, ogni N epoche, le metriche di validation della sola classe escooter tra una run multiclasse (dagli snapshot di `snapshot_checkpoints.py`) e una run monoclasse di riferimento. |
| `config.py` | Libreria di configurazione centralizzata: legge `scripts/.env` ed espone le costanti usate dagli altri script; non eseguibile direttamente. |
| `dataset_index.py` | Libreria che mantiene aggiornati `data/datasets.json` e `data/README.md` a partire dal CSV dei dataset; usata da `download_dataset.py` e `dedupe_augmented.py`, non uno script a sé. |
| `dedupe_augmented.py` | Deduplica le varianti augmentate di un dataset scaricato e converte i poligoni in bounding box (v. [sezione 2](#2-deduplica-augmentation--conversione-poligoni--dedupe_augmentedpy)). |
| `dedupe_phash.py` | Deduplica per contenuto (perceptual hash) il dataset `union_reviewed`, dopo la revisione manuale (v. [sezione 8](#8-opzionale-dedup-phash-a-posteriori--dedupe_phashpy)). |
| `download_batch.py` | Esegue download + dedupe per ogni dataset abilitato nel CSV (v. [sezione 1b](#1b-download-batch-da-csv--download_batchpy)). |
| `download_dataset.py` | Scarica un singolo dataset Roboflow in formato YOLO in `data/raw/` (v. [sezione 1](#1-download-di-un-dataset--download_datasetpy)). |
| `eval_mined.sh` | Confronta lo small sul pool attuale con lo stesso small addestrato anche sulle candidate del mining, sui due test set esterni. |
| `eval_multiclass.sh` | Confronta il nano monoclasse col nano multiclasse (person/bicycle/motorcycle + escooter) sui due test set esterni, sulla sola classe escooter. |
| `eval_nano_vs_small.sh` | Confronta il nano e lo small sui due test set esterni (frame da video e holdout kickboard). |
| `eval_testset.py` | Valida uno o più pesi YOLO (monoclasse o multiclasse, v. `--class-index`) su uno o più test set esterni, con scomposizione facoltativa delle metriche per taglia small/medium/large. |
| `extract_video_frames.py` | Estrae frame a frequenza fissa da una cartella di video, per costruire un test set indipendente dai dataset di training. |
| `materialize_union_reviewed.py` | Copia solo le immagini con decisione "select" di `review_app.py` in una cartella pronta all'uso (v. [sezione 7](#7-materializzazione-del-dataset-finale--materialize_union_reviewedpy)). |
| `materialize_union_reviewed_coco.sh` | Richiama `materialize_union_reviewed.py` su `union_reviewed_coco` dopo la sua revisione manuale (v. [sezione 9b](#9b-revisione-manuale-e-materializzazione-finale)). |
| `mine_hard_examples.py` | Hard-example mining: seleziona da un dataset esterno già annotato le immagini dove un modello nostro sbaglia (box mancati o deboli), filtrate per area e deduplicate per pHash contro il pool e i test set. |
| `mine_scooter_detect.sh` | Lancia `mine_hard_examples.py` sul dataset Ultralytics Platform `scooter-detectyolov8`. |
| `mine_ver2_bikes_scooters.sh` | Lancia `mine_hard_examples.py` sul dataset Ultralytics Platform `ver2-bikes-scooters-and-others` (molto augmentato: candidate da rivedere con più cautela). |
| `report_image_index.py` | Produce un report dettagliato sulle immagini di `data/image_index.json` (conteggi, motivi di esclusione, ecc.). |
| `review_app.py` | Applicazione locale (browser) per la revisione manuale di un dataset YOLO: selezione delle immagini e modifica delle bbox, di qualunque classe (v. [sezione 6](#6-selezione-manuale-finale--review_apppy)). |
| `review_app_rider_contaminated.sh` | Apre `review_app.py` sulle immagini flaggate per conducente incluso nella bbox escooter. |
| `review_app_union.sh` | Apre `review_app.py` sul dataset di unione grezzo. |
| `review_app_union_reviewed_coco.sh` | Apre `review_app.py` su `union_reviewed_coco`, dopo l'annotazione delle classi COCO. |
| `select_images.py` | Seleziona le immagini "migliori" tra i dataset deduplicati in `data/interim` (v. [sezione 3](#3-selezione-delle-immagini-candidate--select_imagespy)). |
| `snapshot_checkpoints.py` | Copia `last.pt` di una run Ultralytics in corso ogni N epoche, per poterne rivalutare a posteriori le metriche alle epoche intermedie. |
| `split_dataset.py` | Divide un dataset YOLO in train/valid, stratificato per dataset sorgente e con raggruppamento delle clip video (v. [sezione 10](#10-split-trainvalid--split_datasetpy)). |
| `split_review_app.py` | Applicazione locale per revisionare a mano lo split train/valid: sposta o esclude immagini fra i due, guidata dai quasi-duplicati per pHash. |
| `train_yolo.py` | Avvia un training Ultralytics YOLO su un dataset train/valid, con filtro/remap opzionale delle classi (`--classes`). |
| `train_yolon_1.sh` | Training di riferimento del nano (yolo26n) monoclasse sul pool attuale. |
| `train_yolon_multiclass_1.sh` | Come `train_yolon_1.sh` ma con le classi person/bicycle/motorcycle oltre all'escooter, per isolare l'effetto del training monoclasse sui falsi positivi. |
| `train_yolon_ultralytics_scooter_1.sh` | Training del nano (yolo26n) da zero sul dataset Ultralytics Platform `scooter-detectyolov8`, senza il pool Roboflow. |
| `train_yolos_1.sh` | Training di riferimento dello small (yolo26s) monoclasse sul pool attuale. |
| `train_yolos_mined_1.sh` | Come `train_yolos_1.sh`, sullo split con le candidate del mining aggiunte al train (v. `build_split_mined.sh`). |

