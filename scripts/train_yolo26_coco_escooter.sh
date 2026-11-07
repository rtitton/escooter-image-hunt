#!/usr/bin/env bash
# Training a 81 classi (COCO 0-79 + escooter 80) su COCO + il nostro split, yolo26n pretrained COCO.
# Pensato per la macchina con la H100, dove c'è COCO (formato YOLO di Ultralytics: images/train2017,
# labels/train2017, ...). Da lanciare dalla radice del repo clonato lì, dopo aver copiato il nostro split
# (data/processed/union_reviewed_coco_split) e yolo26n.pt (non è nel repo: .gitignore ignora i .pt).
# Richiede bash (usa array): non lanciarlo con "sh".
#
#   scripts/train_yolo26_coco_escooter.sh                      # prima run: nostre immagini ripetute 1 volta
#   REPEAT=5 scripts/train_yolo26_coco_escooter.sh             # nostre immagini ripetute 5 volte
#
# Variabili d'ambiente, tutte opzionali: i default sono quelli della macchina H100 e della prima run
# (150 epoche), e vengono sostituiti da quanto si passa da fuori.
#   COCO_DIR     COCO in formato YOLO (default /data/dataset/coco)
#   SPLIT_DIR    il nostro split (default data/processed/union_reviewed_coco_split)
#   REPEAT       quante volte inserire le nostre immagini nel train (default 1: prima run senza correttivi
#                per lo squilibrio; se la escooter peggiora rispetto alla baseline monoclasse si riprova con 3-5)
#   OUT_DIR      dove scrivere liste e data.yaml (default data/processed/coco_escooter_merge_rep<REPEAT>:
#                una cartella per valore di REPEAT, così una run con REPEAT diverso non sovrascrive le
#                liste di un'altra che si potrebbe dover riprendere)
#   YOLO_MODEL   pesi di partenza (default yolo26n.pt)
#   EPOCHS (150), BATCH (32), WORKERS (16: sulla H100 il collo di bottiglia è la CPU), FREEZE (0)
#   NAME         nome della run (default coco_escooter_<modello>_rep<REPEAT>)
#   BUILD_LISTS  1 = rigenera liste e data.yaml prima del training (default); 0 = riusa quelli in OUT_DIR
#   RESUME       1 = riprende la run interrotta chiamata NAME (da last.pt, v. sotto); non rigenera le liste
#
# Per interrompere e riprendere in seguito: Ctrl+C e poi, con lo stesso NAME e gli stessi parametri,
#   RESUME=1 NAME=<nome della run> scripts/train_yolo26_coco_escooter.sh
#
# Ricetta: pesi pretrained COCO, patience 20, imgsz 640, seed 42, nessun cls_pw (le classi COCO rare non vanno
# protette). La testa di classificazione ha 81 uscite invece delle 80 dei pesi COCO: quei pesi non si ricaricano
# e la testa riparte da zero, mentre backbone e neck partono pretrained; con COCO nel train dovrebbe recuperare.
# Valutare poi sui test set esterni (copiando best.pt sulla macchina locale):
#   eval_testset.py --weights ... --class-index 80 --imgsz 640 (v. VALUTAZIONI.md)
set -e

COCO_DIR="${COCO_DIR:-/data/dataset/coco}"
SPLIT_DIR="${SPLIT_DIR:-data/processed/union_reviewed_coco_split}"
REPEAT="${REPEAT:-1}"
OUT_DIR="${OUT_DIR:-data/processed/coco_escooter_merge_rep${REPEAT}}"
YOLO_MODEL="${YOLO_MODEL:-yolo26n.pt}"
EPOCHS="${EPOCHS:-150}"
BATCH="${BATCH:-32}"
WORKERS="${WORKERS:-16}"
FREEZE="${FREEZE:-0}"
NAME="${NAME:-coco_escooter_${YOLO_MODEL%.pt}_rep${REPEAT}}"  # senza ".pt" nel nome
BUILD_LISTS="${BUILD_LISTS:-1}"
RESUME="${RESUME:-0}"

if [[ "$RESUME" != "0" && "$RESUME" != "1" ]]; then
  echo "RESUME deve essere 0 oppure 1 (valore ricevuto: $RESUME)" >&2
  exit 2
fi

# Liste e data.yaml: si rigenerano a ogni run (con REPEAT diverso cambiano), non quando si riprende una run
# (i dati devono restare quelli della run originale).
if [[ "$RESUME" == "0" && "$BUILD_LISTS" == "1" ]]; then
  python3 scripts/build_coco_merged_lists.py \
    --coco-dir "$COCO_DIR" \
    --split-dir "$SPLIT_DIR" \
    --out-dir "$OUT_DIR" \
    --repeat "$REPEAT"
fi

TRAIN_ARGS=(
  --data "$OUT_DIR/data.yaml"
  --model "$YOLO_MODEL"
  --epochs "$EPOCHS"
  --batch "$BATCH"
  --imgsz 640
  --patience 20
  --workers "$WORKERS"
  --seed 42
  --device 0
  --project data/runs
  --name "$NAME"
  --freeze "$FREEZE"
)
if [[ "$RESUME" == "1" ]]; then
  TRAIN_ARGS+=(--resume)  # train_yolo.py cerca last.pt della run <project>/<NAME>
fi

python3 scripts/train_yolo.py "${TRAIN_ARGS[@]}"
