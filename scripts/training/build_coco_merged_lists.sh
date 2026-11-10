#!/usr/bin/env bash
# Costruisce le liste train.txt/val.txt e il data.yaml a 81 classi (COCO 0-79 + escooter 80) che uniscono COCO
# (o un suo subset) e il nostro split, con build_coco_merged_lists.py. Non copia immagini. Il training si lancia
# poi a parte con train_yolo26_coco_escooter.sh e DATA_YAML=<OUT_DIR>/data.yaml (il nome è stampato alla fine).
# Richiede bash (usa array): non lanciarlo con "sh". Usa "uv run python3" come gli altri script.
#
#   scripts/training/build_coco_merged_lists.sh                          # nostre immagini ripetute 1 volta
#   REPEAT=5 scripts/training/build_coco_merged_lists.sh                 # nostre immagini ripetute 5 volte
#   COCO_SUBSET_DIR=data/processed/coco_subset_30000_0 scripts/training/build_coco_merged_lists.sh
#
# Variabili d'ambiente, tutte opzionali (i default sono quelli della macchina H100):
#   COCO_DIR         COCO in formato YOLO (default /data/dataset/coco)
#   SPLIT_DIR        il nostro split (default data/processed/union_reviewed_coco_split)
#   REPEAT           quante volte inserire le nostre immagini nel train (default 1)
#   COCO_SUBSET_DIR  cartella del subset di COCO (output di make_coco_subset.sh) al posto di tutto COCO
#   OUT_DIR          dove scrivere liste e data.yaml (default data/processed/coco_escooter_merge_rep<REPEAT>:
#                    una cartella per valore di REPEAT, così una run con REPEAT diverso non sovrascrive le
#                    liste di un'altra che si potrebbe dover riprendere). Con un subset conviene un nome
#                    esplicito, per non confonderlo con la versione su COCO completo
set -e
cd "$(dirname "$0")/../.."

COCO_DIR="${COCO_DIR:-/data/dataset/coco}"
SPLIT_DIR="${SPLIT_DIR:-data/processed/union_reviewed_coco_split}"
REPEAT="${REPEAT:-1}"
OUT_DIR="${OUT_DIR:-data/processed/coco_escooter_merge_rep${REPEAT}}"
COCO_SUBSET_DIR="${COCO_SUBSET_DIR:-}"

ARGS=(--coco-dir "$COCO_DIR" --split-dir "$SPLIT_DIR" --out-dir "$OUT_DIR" --repeat "$REPEAT")
[[ -n "$COCO_SUBSET_DIR" ]] && ARGS+=(--subset-dir "$COCO_SUBSET_DIR")

uv run python3 scripts/training/build_coco_merged_lists.py "${ARGS[@]}"
echo "Training: DATA_YAML=$OUT_DIR/data.yaml scripts/training/train_yolo26_coco_escooter.sh"
