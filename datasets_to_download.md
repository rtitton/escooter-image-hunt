# `datasets_to_download.csv` — tracciato

Elenco dei dataset Roboflow da trattare in questa istanza del progetto e dei
parametri con cui trattarli. È il punto di ingresso della pipeline: determina
cosa viene scaricato, deduplicato e passato alla selezione delle immagini
(vedi [PIPELINE.md](PIPELINE.md)).

- Sta nella **root del progetto** ed è **versionato in git** (a differenza di
  `data/`, che è in `.gitignore`). È comunque specifico dell'istanza: per una
  nuova istanza si parte da [`roboflow-datasets_to_download.csv`](roboflow-datasets_to_download.csv),
  l'elenco generale dei dataset selezionati (vedi sotto).
- **Nessuno script lo modifica**: è mantenuto a mano.

## Formato del file

- CSV con **virgola** come separatore, **una riga di header**, codifica UTF-8,
  fine riga LF.
- **Nessun campo può contenere una virgola** (non si usano le virgolette di
  quoting): vale anche per `notes`. Per elencare più valori in un campo si usa
  la barra verticale `|` (vedi `escooter_class_name`).
- Ogni riga deve avere **esattamente lo stesso numero di campi dell'header**
  (11). `download_batch.py` si rifiuta di partire se una riga è malformata.
- Non esistono righe di commento.

## Tracciato

| # | colonna | valori | obbligatorio | significato |
|--:|---|---|:--:|---|
| 1 | `enabled` | `0` / `1` | sì | `1` = il dataset partecipa alla pipeline. `0` = ignorato ovunque (resta nell'elenco senza essere elaborato). |
| 2 | `download` | `0` / `1` | sì | `1` = `download_batch.py` (ri)scarica e rideduplica questa riga. Si mette a `0` una volta scaricato, per non rifare il lavoro. |
| 3 | `workspace_id` | slug Roboflow | sì | Workspace del progetto su Roboflow Universe. |
| 4 | `project_id` | slug Roboflow | sì | Progetto su Roboflow Universe. |
| 5 | `version` | intero, oppure vuoto | no | Versione del dataset Roboflow. Vuoto = `download_dataset.py` sceglie la più recente senza augmentation. Vedi *Derivazione dell'id*. |
| 6 | `escooter_class_name` | uno o più nomi separati da `\|` | sì | Nomi delle classi del `data.yaml` del dataset che corrispondono a un escooter. Vedi *Classi escooter*. |
| 7 | `variety_min_instances` | intero, oppure vuoto / `default` | no | Override per-dataset di `VARIETY_MIN_INSTANCES`. Vedi *Override di selezione*. |
| 8 | `closeup_area_threshold` | float, oppure vuoto / `default` | no | Override per-dataset di `CLOSEUP_AREA_THRESHOLD`. |
| 9 | `faraway_area_threshold` | float, oppure vuoto / `default` | no | Override per-dataset di `FARAWAY_AREA_THRESHOLD`. |
| 10 | `phash_distance_threshold` | intero, oppure vuoto / `default` | no | Override per-dataset di `PHASH_DISTANCE_THRESHOLD`. |
| 11 | `notes` | testo libero senza virgole | no | Annotazioni per chi mantiene il file. Non letto da nessuno script. |

## Derivazione dell'id del dataset

Gli script identificano un dataset con `<project_id>-v<version>` (lo stesso id
usato in `data/datasets.json` e come nome cartella in `data/raw/` e
`data/interim/`). Esempio: riga `kts-data, -9gklp, 7` → id `-9gklp-v7`.

Una riga con `version` **vuoto** viene scaricata (`download_batch.py` prende la
versione più recente senza aug) ma **non entra** negli stadi che si basano
sull'id (`select_images.py`): finché non si fissa a mano il numero di versione
effettivo nella colonna `version`, `check_dataset_sync.py` la segnala come
"versione non fissata".

## Classi escooter (`escooter_class_name`)

- Uno o più nomi, **separati da `|`**, che devono comparire tra le `names:` del
  `data.yaml` del dataset scaricato. Esempio:
  `kickboard|--- - v5 2025-07-10 11-27am`.
- `download_dataset.py` verifica che quei nomi esistano davvero e registra in
  `data/datasets.json` sia `escooter_class_names` sia gli eventuali
  `escooter_class_names_missing`; se ne mancano, `download_batch.py` salta la
  deduplica di quella riga.
- In `select_images.py` tutte le classi elencate contano come "escooter"; le
  altre classi del dataset sorgente vengono ignorate (le classi COCO saranno
  ri-annotate a valle con un modello pretrained).
- Nomi di classe corrotti (Roboflow a volte ci infila stringhe di versione)
  vanno riportati **verbatim**, spazi compresi.

## Override di selezione (colonne 7–10)

Permettono di trattare un singolo dataset con soglie diverse da quelle globali,
senza toccare `scripts/.env`. Utile per sorgenti particolari — footage con
escooter piccoli (varietà più permissiva, closeup più alto, faraway più basso)
o footage molto ripetitiva (pHash più stretto per non collassarla).

- **Cella vuota o `default`** → si usa il valore di `scripts/.env` (o, se
  assente lì, il default di `scripts/config.py`).
- **Valore numerico** → sovrascrive quel parametro **solo per quel dataset**.
  Un valore non numerico fa fallire la pipeline con un errore che indica riga e
  colonna.

| colonna | parametro | stadio di `select_images.py` | effetto |
|---|---|---|---|
| `variety_min_instances` | `VARIETY_MIN_INSTANCES` | filtro varietà | minimo di istanze di classi COCO (oltre l'escooter) perché l'immagine sia "varia" |
| `closeup_area_threshold` | `CLOSEUP_AREA_THRESHOLD` | filtri economici | area relativa della bbox escooter oltre la quale è "primo piano" e l'immagine è scartata |
| `faraway_area_threshold` | `FARAWAY_AREA_THRESHOLD` | filtri economici | area relativa sotto la quale la bbox escooter è "troppo lontana" e l'immagine è scartata |
| `phash_distance_threshold` | `PHASH_DISTANCE_THRESHOLD` | dedup cross-dataset | distanza di Hamming del perceptual hash sotto la quale due immagini sono quasi-duplicati |

Nota su `phash_distance_threshold`: la dedup è **cross-dataset**. Per una
coppia di immagini di due dataset con soglie diverse vale la **più stretta
delle due** (`min`), così un dataset con soglia bassa non viene assorbito nei
cluster di dataset più permissivi.

Gli override attivi sono ricapitolati da `report_image_index.py` (riga
`[override CSV: ...]` sotto ogni dataset).

## Chi legge quali colonne

| script | colonne usate | quando |
|---|---|---|
| `download_batch.py` | `enabled`, `download`, `workspace_id`, `project_id`, `version`, `escooter_class_name` | elabora le righe con `enabled=1` **e** `download=1` |
| `dataset_index.enabled_ids()` | `enabled`, `project_id`, `version` | usato da `select_images.py` e `build_union_dataset.py` per sapere quali dataset processare (serve `enabled=1` **e** `version` valorizzato) |
| `dataset_index.selection_overrides()` | `project_id`, `version` + colonne 7–10 | letto da `select_images.py` e `report_image_index.py` |
| `check_dataset_sync.py` | `enabled`, `workspace_id`, `project_id`, `version` | confronto CSV ↔ `data/datasets.json` ↔ `data/raw/` |

## Relazione con gli altri file

- **`roboflow-datasets_to_download.csv`** — elenco generale (versionato) dei
  dataset Roboflow selezionati, con le sole 7 colonne base (senza gli override
  e di norma senza `version`). È il punto di partenza per creare
  `datasets_to_download.csv` in una nuova istanza. Il codice accetta anche un
  CSV senza le colonne 7–10: le tratta come `default`.
- **`data/datasets.json`** — indice macchina, popolato da `download_dataset.py`
  a partire da questo CSV (id, sorgente, classi, conteggi, formato
  annotazioni). `data/README.md` ne è la vista leggibile.

## Errori comuni

- **Virgola dentro un campo** (tipicamente in `notes` o in un nome di classe):
  spezza la riga in più campi → `download_batch.py` la segnala come malformata.
  Sostituire con `-` o `/`.
- **Numero di campi diverso dall'header**: succede aggiungendo una colonna
  all'header ma non a tutte le righe. Ogni riga deve avere 11 campi (anche se
  vuoti: `...,,,,...`).
- **Override non numerico** (`0,6` con la virgola invece di `0.6`, oppure
  testo): la pipeline si ferma indicando riga e colonna.
- **`version` vuoto** e ci si aspetta che il dataset entri in `select_images.py`:
  non succede finché non si fissa la versione.
