"""Configurazione centralizzata: legge scripts/.env ed espone le costanti
usate dagli altri script della pipeline. Non pensato per essere eseguito
direttamente.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


# --- Roboflow ---
RF_API_KEY = os.environ.get("RF_API_KEY")
DOWNLOAD_FORMAT = os.environ.get("DOWNLOAD_FORMAT", "yolov8")

# --- Percorsi (relativi alla root del progetto / a DATA_ROOT) ---
DATA_ROOT = REPO_ROOT / os.environ.get("DATA_ROOT", "data")
RAW_DIR = DATA_ROOT / os.environ.get("RAW_DIRNAME", "raw")
INDEX_PATH = DATA_ROOT / os.environ.get("INDEX_FILENAME", "datasets.json")
README_PATH = DATA_ROOT / os.environ.get("README_FILENAME", "README.md")
CSV_PATH = REPO_ROOT / os.environ.get("CSV_FILENAME", "datasets_to_download.csv")
CANDIDATES_PATH = DATA_ROOT / os.environ.get("CANDIDATES_FILENAME", "selected_images.txt")
IMAGE_INDEX_PATH = DATA_ROOT / os.environ.get("IMAGE_INDEX_FILENAME", "image_index.json")
FLAGGED_RIDER_PATH = DATA_ROOT / os.environ.get("FLAGGED_RIDER_FILENAME", "flagged_rider_contamination.txt")
FLAGGED_AREA_PATH = DATA_ROOT / os.environ.get("FLAGGED_AREA_FILENAME", "flagged_area_threshold.txt")
VARIETY_CACHE_PATH = DATA_ROOT / os.environ.get("VARIETY_CACHE_FILENAME", "cache/variety_filter_cache.json")
PHASH_CACHE_PATH = DATA_ROOT / os.environ.get("PHASH_CACHE_FILENAME", "cache/phash_cache.json")
SELECT_IMAGES_LOG_PATH = DATA_ROOT / os.environ.get("SELECT_IMAGES_LOG_FILENAME", "logs/select_images.log")
UNION_DIR = DATA_ROOT / os.environ.get("UNION_DIRNAME", "processed/union")
UNION_REVIEW_SAMPLE_DIR = DATA_ROOT / os.environ.get("UNION_REVIEW_SAMPLE_DIRNAME", "processed/union_review_sample")
UNION_REVIEWED_DIR = DATA_ROOT / os.environ.get("UNION_REVIEWED_DIRNAME", "processed/union_reviewed")
UNION_REVIEWED_PHASHDEDUP_DIR = DATA_ROOT / os.environ.get(
    "UNION_REVIEWED_PHASHDEDUP_DIRNAME", "processed/union_reviewed_phashdedup"
)
RIDER_CONTAMINATED_DIR = DATA_ROOT / os.environ.get("RIDER_CONTAMINATED_DIRNAME", "processed/rider_contaminated")
BYDATASET_DIR = DATA_ROOT / os.environ.get("BYDATASET_DIRNAME", "processed/bydataset")
UNION_REVIEWED_COCO_DIR = DATA_ROOT / os.environ.get("UNION_REVIEWED_COCO_DIRNAME", "processed/union_reviewed_coco")
# target per union_reviewed_coco dopo la revisione manuale (review_app.py, review_decisions.json
# nella stessa cartella): solo le immagini "select", prodotto da materialize_union_reviewed.py
UNION_REVIEWED_COCO_FINAL_DIR = DATA_ROOT / os.environ.get(
    "UNION_REVIEWED_COCO_FINAL_DIRNAME", "processed/union_reviewed_coco_final"
)
COCO_ANNOTATION_CACHE_PATH = DATA_ROOT / os.environ.get(
    "COCO_ANNOTATION_CACHE_FILENAME", "cache/coco_annotation_cache.json"
)
# elenco delle detection scartate per overlap con bbox escooter (classi sosia), per revisione manuale
FLAGGED_COCO_OVERLAP_PATH = DATA_ROOT / os.environ.get(
    "FLAGGED_COCO_OVERLAP_FILENAME", "flagged_coco_overlap.txt"
)
# elenco (un nome immagine per riga) delle immagini per cui, dopo revisione manuale del file sopra,
# lo scarto per overlap sosia va disattivato (la detection va invece tenuta)
COCO_OVERLAP_RESCUE_PATH = DATA_ROOT / os.environ.get(
    "COCO_OVERLAP_RESCUE_FILENAME", "coco_overlap_rescue.txt"
)

# --- Test set da video (extract_video_frames.py) ---
VIDEO_TESTSET_DIR = DATA_ROOT / os.environ.get("VIDEO_TESTSET_DIRNAME", "processed/video_testset")
VIDEO_TESTSET_FPS = _env_float("VIDEO_TESTSET_FPS", 2.0)  # frame estratti per secondo di video

# --- Classi ---
ESCOOTER_CLASS_ID = _env_int("ESCOOTER_CLASS_ID", 80)
PERSON_CLASS_ID = _env_int("PERSON_CLASS_ID", 0)  # classe "person" in COCO

# --- Filtri selezione immagini (select_images.py) ---
CLOSEUP_AREA_THRESHOLD = _env_float("CLOSEUP_AREA_THRESHOLD", 0.4)
FARAWAY_AREA_THRESHOLD = _env_float("FARAWAY_AREA_THRESHOLD", 0.001)
MIN_PIXELS = _env_int("MIN_PIXELS", 160_000)
PHASH_DISTANCE_THRESHOLD = _env_int("PHASH_DISTANCE_THRESHOLD", 8)

# --- Dedup temporale (select_images.py, opzionale con --temporal-dedup) ---
# Assottiglia le sequenze di frame consecutivi estratti dallo stesso video.
TEMPORAL_MIN_SEQ = _env_int("TEMPORAL_MIN_SEQ", 5)  # frame minimi in un gruppo (dataset, split, clip) perché venga assottigliato
TEMPORAL_KEEP_DISTANCE = _env_int("TEMPORAL_KEEP_DISTANCE", 10)  # distanza di Hamming del pHash dall'ultimo frame tenuto sotto la quale un frame è ridondante
TEMPORAL_MAX_GAP = _env_int("TEMPORAL_MAX_GAP", 60)  # massima distanza di indice entro cui un frame di riferimento "copre" i successivi
RIDER_OVERLAP_THRESHOLD = _env_float("RIDER_OVERLAP_THRESHOLD", 0.3)  # frazione dell'area della bbox escooter coperta da una detection "persona" perché la coppia sia considerata (precondizione spaziale prima del confronto altezze)
RIDER_HEIGHT_RATIO_THRESHOLD = _env_float("RIDER_HEIGHT_RATIO_THRESHOLD", 1.0)  # rapporto (altezza escooter / altezza persona) oltre il quale si considera il conducente incluso nell'annotazione

# --- Modello varietà COCO (select_images.py) ---
COCO_MODEL = os.environ.get("COCO_MODEL", "yolo11l.pt")
COCO_BATCH_SIZE = _env_int("COCO_BATCH_SIZE", 16)
VARIETY_CACHE_SAVE_EVERY = _env_int("VARIETY_CACHE_SAVE_EVERY", 20)
VARIETY_MIN_INSTANCES = _env_int("VARIETY_MIN_INSTANCES", 1)  # istanze COCO minime nell'orientazione originale perché un'immagine sia di buona varietà

# --- Annotazione classi COCO sul dataset finale (annotate_coco_classes.py) ---
# soglia di confidenza usata durante l'inferenza (e quindi salvata in cache): bassa di proposito,
# per poter ritarare COCO_ANNOTATION_CONF_THRESHOLD senza dover rifare l'inferenza
COCO_ANNOTATION_CAPTURE_CONF = _env_float("COCO_ANNOTATION_CAPTURE_CONF", 0.1)
# soglia di confidenza effettiva applicata in scrittura: più alta della capture conf perché qui
# le detection diventano etichette di training permanenti (un falso positivo pesa più di un mancato rilevamento)
COCO_ANNOTATION_CONF_THRESHOLD = _env_float("COCO_ANNOTATION_CONF_THRESHOLD", 0.45)
# frazione di area di una detection COCO coperta da una bbox escooter oltre la quale viene scartata
# perché ritenuta lo stesso oggetto fisico (es. monopattino misclassificato come bicicletta, o il
# solo pianale letto come skateboard/snowboard); non è un IoU simmetrico apposta, per catturare
# anche i casi in cui la detection COCO è molto più piccola della bbox escooter
COCO_ESCOOTER_OVERLAP_THRESHOLD = _env_float("COCO_ESCOOTER_OVERLAP_THRESHOLD", 0.6)
# classi COCO plausibili come misclassificazione dell'intero monopattino o di una sua parte (bici,
# moto, skateboard, snowboard, o lo stelo/manubrio letto come colonnina/parchimetro): solo queste
# sono soggette allo scarto per overlap sopra. Applicarlo a tutte le classi scarterebbe anche oggetti
# reali chiaramente distinti (es. un'auto o una borsa) che ricadono per intero nella bbox escooter
# solo per prospettiva/profondità, non perché coincidano fisicamente con essa (v. PIPELINE.md sezione 9)
COCO_ESCOOTER_LOOKALIKE_CLASSES = {
    int(v) for v in os.environ.get("COCO_ESCOOTER_LOOKALIKE_CLASSES", "1,3,12,31,36").split(",")
}
# classi COCO implausibili in una scena esterna (marciapiede/strada) come quelle del dataset:
# oggetti da interno (elettrodomestici, arredo da cucina/bagno) che se rilevati sono quasi certamente
# un errore di classificazione ad alta confidenza, non intercettabile alzando la sola soglia generale
# senza perdere moltissimo recall altrove. Scartate a prescindere dalla confidenza. Stringa vuota per
# disattivare il filtro (nessuna classe scartata per implausibilità)
_IMPLAUSIBLE_DEFAULT = "40,42,44,57,59,60,61,62,64,65,66,68,69,70,71,72,78,79"  # wine glass, fork, spoon,
# couch, bed, dining table, toilet, tv, mouse, remote, keyboard, microwave, oven, toaster, sink,
# refrigerator, hair drier, toothbrush
_implausible_env = os.environ.get("COCO_IMPLAUSIBLE_CLASSES", _IMPLAUSIBLE_DEFAULT)
COCO_IMPLAUSIBLE_CLASSES = {int(v) for v in _implausible_env.split(",") if v.strip()}
COCO_ANNOTATION_CACHE_SAVE_EVERY = _env_int("COCO_ANNOTATION_CACHE_SAVE_EVERY", 20)

# --- Dedupe augmented (dedupe_augmented.py) ---
BLACK_THRESHOLD = _env_int("BLACK_THRESHOLD", 10)
EDGE_SAMPLE = _env_int("EDGE_SAMPLE", 30)

# --- Controllo visivo campione (build_visual_check_sample.py) ---
BOX_COLOR = tuple(int(v) for v in os.environ.get("BOX_COLOR", "255,0,0").split(","))
BOX_WIDTH = _env_int("BOX_WIDTH", 4)
PERSON_BOX_COLOR = tuple(int(v) for v in os.environ.get("PERSON_BOX_COLOR", "0,255,0").split(","))
# colore delle bbox delle altre classi COCO (diverse da persona), disegnate da annotate_coco_classes.py --sample
COCO_BOX_COLOR = tuple(int(v) for v in os.environ.get("COCO_BOX_COLOR", "0,128,255").split(","))
# colore delle bbox scartate per overlap sosia, disegnate da annotate_coco_classes.py --export-flagged-sample
COCO_FLAGGED_BOX_COLOR = tuple(int(v) for v in os.environ.get("COCO_FLAGGED_BOX_COLOR", "255,140,0").split(","))

# --- App di revisione (review_app.py) ---
REVIEW_APP_PORT = _env_int("REVIEW_APP_PORT", 8765)
