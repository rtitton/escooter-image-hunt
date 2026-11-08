#!/bin/bash
# Crea un subset di COCO con make_coco_subset.py. Parametri da variabili d'ambiente (default tra parentesi):
#   COCO_DIR        COCO in formato YOLO (data/coco)
#   SPLIT           train2017 o val2017 (train2017)
#   CLASSES         nomi COCO, indici o preset, separati da spazio (traffic)
#   MAX_IMAGES      numero massimo di immagini (30000)
#   MIN_PER_CLASS   minimo di immagini per classe; se non impostato, lo script mostra le frequenze e lo chiede
#   SEED            seme (42)
#   OUT_DIR         dove scrivere train.txt e train.csv (data/processed/coco_subset_<MAX_IMAGES>)
#
# Avvio (dalla radice del repo):
#   scripts/training/make_coco_subset.sh
#   MAX_IMAGES=20000 CLASSES="car truck bus" scripts/training/make_coco_subset.sh
# Il subset si passa poi a build_coco_merged_lists.py con --coco-train-list <OUT_DIR>/train.txt.
set -e
cd "$(dirname "$0")/../.."

COCO_DIR="${COCO_DIR:-data/coco}"
SPLIT="${SPLIT:-train2017}"
CLASSES="${CLASSES:-traffic}"
MAX_IMAGES="${MAX_IMAGES:-30000}"
SEED="${SEED:-42}"
OUT_DIR="${OUT_DIR:-data/processed/coco_subset_${MAX_IMAGES}}"

ARGS=(--coco-dir "$COCO_DIR" --split "$SPLIT" --max-images "$MAX_IMAGES" --seed "$SEED"
      --out-list "$OUT_DIR/train.txt")
[ -n "${MIN_PER_CLASS:-}" ] && ARGS+=(--min-per-class "$MIN_PER_CLASS")

# CLASSES va espanso in più argomenti (word splitting voluto)
# shellcheck disable=SC2086
uv run python3 scripts/training/make_coco_subset.py "${ARGS[@]}" --classes $CLASSES
