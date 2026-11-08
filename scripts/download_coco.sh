#!/bin/bash
# Scarica COCO 2017 nel layout Ultralytics (etichette YOLO + immagini train/val, senza test2017):
#   <COCO_DIR>/images/{train2017,val2017}  <COCO_DIR>/labels/{train2017,val2017}
# Riprendibile (wget -c): se si interrompe basta rilanciarlo. Gli zip restano in <COCO_DIR>/_zips
# (circa 20 GB, da cancellare a fine lavoro).
#
# Avvio (dalla radice del repo):
#   scripts/download_coco.sh                 # default: data/coco
#   COCO_DIR=/altro/percorso scripts/download_coco.sh
set -e
cd "$(dirname "$0")/.."
COCO_DIR="${COCO_DIR:-data/coco}"
ASSETS_URL=https://github.com/ultralytics/assets/releases/download/v0.0.0

mkdir -p "$COCO_DIR/images" "$COCO_DIR/_zips"
cd "$COCO_DIR/_zips"
wget -c "$ASSETS_URL/coco2017labels.zip"
wget -c http://images.cocodataset.org/zips/val2017.zip
wget -c http://images.cocodataset.org/zips/train2017.zip
cd ..
# Lo zip delle etichette contiene già la cartella coco/ (labels/, train2017.txt, val2017.txt):
# va estratto nella cartella padre
unzip -q -n _zips/coco2017labels.zip -d ..
unzip -q -n _zips/val2017.zip -d images
unzip -q -n _zips/train2017.zip -d images
echo DONE
