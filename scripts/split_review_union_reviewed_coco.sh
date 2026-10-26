#!/bin/sh
# Avvia split_review_app.py sullo split train/valid di union_reviewed_coco
# (generato da split_dataset.py). Poi apri http://localhost:<porta> nel browser.
# Eventuali argomenti (es. --port 8800) vengono passati allo script.
. scripts/.env

.venv/bin/python scripts/split_review_app.py "$DATA_ROOT/processed/union_reviewed_coco_split" "$@"
