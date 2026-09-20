#!/bin/bash

. scripts/.env

# Materializza la versione finale di union_reviewed_coco dopo una revisione manuale con
# review_app.py (es. scripts/review_app_union_reviewed_coco.sh): tiene solo le immagini con
# decisione "select" in review_decisions.json (scritto da review_app.py nella stessa cartella
# del dataset), con le bbox così come modificate durante la revisione. Da rilanciare ogni volta
# che si escludono altre immagini o si ritoccano altre bbox.
python scripts/materialize_union_reviewed.py \
    --source-dir "$DATA_ROOT/$UNION_REVIEWED_COCO_DIRNAME" \
    --out-dir "$DATA_ROOT/$UNION_REVIEWED_COCO_FINAL_DIRNAME"
