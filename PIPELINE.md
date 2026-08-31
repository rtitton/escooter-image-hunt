# Pipeline dati — stato attuale

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
review_app.py  ── selezione manuale finale (s/l/n) → data/review_decisions.json
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
COCO saranno annotate in un passo successivo, non ancora implementato, con
un modello pretrained di grandi dimensioni).

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
`s` seleziona, `l` seleziona con riserva, `n` scarta, frecce per navigare
senza decidere, backspace per cancellare la decisione corrente. Ogni
decisione è salvata subito in `data/review_decisions.json`: la sessione si
può interrompere e riprendere quando si vuole, ripartendo dalla prima
immagine ancora senza decisione.

