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
RIDER_CONTAMINATED_DIR = DATA_ROOT / os.environ.get("RIDER_CONTAMINATED_DIRNAME", "processed/rider_contaminated")

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

# --- Dedupe augmented (dedupe_augmented.py) ---
BLACK_THRESHOLD = _env_int("BLACK_THRESHOLD", 10)
EDGE_SAMPLE = _env_int("EDGE_SAMPLE", 30)

# --- Controllo visivo campione (build_visual_check_sample.py) ---
BOX_COLOR = tuple(int(v) for v in os.environ.get("BOX_COLOR", "255,0,0").split(","))
BOX_WIDTH = _env_int("BOX_WIDTH", 4)
PERSON_BOX_COLOR = tuple(int(v) for v in os.environ.get("PERSON_BOX_COLOR", "0,255,0").split(","))

# --- App di revisione (review_app.py) ---
REVIEW_APP_PORT = _env_int("REVIEW_APP_PORT", 8765)
