#!/bin/bash
set -euo pipefail
. scripts/.env

# Dopo la revisione manuale delle candidate del mining (review_app.py su
# data/processed/mined_scooter-detectyolov8): tiene solo le "select", le aggiunge al train di una COPIA dello
# split attuale (il valid resta identico, lo split originale non viene toccato) e riscrive il data.yaml
# della copia, che altrimenti punterebbe ancora all'originale.
SRC=data/processed/mined_scooter-detectyolov8
FINAL=${SRC}_final
SPLIT=data/processed/union_reviewed_coco_split
DST=data/processed/union_reviewed_coco_split_mined

if [ -e "$DST" ]; then
  echo "$DST esiste già: rimuovilo a mano se vuoi rigenerarlo (lo script non cancella nulla)" >&2
  exit 1
fi

# esce con errore se manca review_decisions.json (revisione non fatta)
uv run python3 scripts/materialize_union_reviewed.py --source-dir "$SRC" --out-dir "$FINAL"

cp -r "$SPLIT" "$DST"
sed -i "s|^path: .*|path: $(pwd)/$DST|" "$DST/data.yaml"

n=0
for img in "$FINAL"/images/*; do
  name=$(basename "$img")
  if [ -e "$DST/train/images/$name" ] || [ -e "$DST/valid/images/$name" ]; then
    echo "collisione di nome: $name" >&2
    exit 1
  fi
  cp "$img" "$DST/train/images/$name"
  cp "$FINAL/labels/${name%.*}.txt" "$DST/train/labels/${name%.*}.txt"
  n=$((n + 1))
done

echo "aggiunte $n immagini a $DST/train"
echo "train: $(ls "$DST/train/images" | wc -l) immagini, valid: $(ls "$DST/valid/images" | wc -l) immagini"
