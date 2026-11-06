# Training a 81 classi (COCO 0-79 + escooter 80) su COCO + il nostro split, yolo26n pretrained COCO.
# Pensato per la macchina con la H100, dove c'è COCO (formato YOLO di Ultralytics: images/train2017,
# labels/train2017, ...). Da lanciare dalla radice del repo clonato lì, dopo aver copiato il nostro split
# (data/processed/union_reviewed_coco_split) e yolo26n.pt (non è nel repo: .gitignore ignora i .pt).
#
#   COCO_DIR=/percorso/coco SPLIT_DIR=/percorso/union_reviewed_coco_split scripts/train_yolo26_coco_escooter.sh
#
# Variabili (tutte con default tranne le prime due):
#   COCO_DIR, SPLIT_DIR   obbligatorie
#   OUT_DIR               dove scrivere le liste e il data.yaml (default data/processed/coco_escooter_merged)
#   REPEAT                quante volte inserire le nostre immagini nel train (default 1: prima run senza
#                         correttivi per lo squilibrio; se la escooter peggiora rispetto alla baseline monoclasse
#                         si riprova con 3-5)
#   WORKERS               worker del dataloader (default 16: sulla H100 il collo di bottiglia è la CPU, aumentare
#                         se ha più core); BATCH (default 128); EPOCHS (default 50); NAME
#   RESUME=1              riprende la run interrotta con lo stesso NAME (da last.pt)
#
# Per interrompere e riprendere in seguito, interrompere il processo con Ctrl+C e rilanciare:
#   RESUME=1 NAME=coco_escooter_26s_rep1 scripts/train_yolo26_coco_escooter.sh
#
# Ricetta: 50 epoche da pesi pretrained COCO, patience 20, imgsz 640, nessun freeze, seed 42, nessun cls_pw (le
# classi COCO rare non vanno protette). La testa di classificazione ha 81 uscite invece delle 80 dei pesi COCO:
# quei pesi non si ricaricano e la testa riparte da zero, mentre backbone e neck partono pretrained; con COCO nel
# train dovrebbe recuperare in poche epoche.
# Valutare poi sui test set esterni (copiando best.pt sulla macchina locale):
#   eval_testset.py --weights ... --class-index 80 --imgsz 640 (v. VALUTAZIONI.md)
set -e

# setup h100 inizio
export COCO_DIR=/data/dataset/coco
export SPLIT_DIR=data/processed/union_reviewed_coco_split
export OUT_DIR=data/processed/coco_escooter_merge
export YOLO_MODEL=yolo26n.pt
export EPOCHS=150
export WORKERS=16
export BATCH=32
export FREEZE=0
# setup h100 fine 

: "${COCO_DIR:?impostare COCO_DIR}"
: "${SPLIT_DIR:?impostare SPLIT_DIR}"
OUT_DIR="${OUT_DIR:-data/processed/coco_escooter_merged}"
REPEAT="${REPEAT:-1}"
WORKERS="${WORKERS:-8}"
YOLO_MODEL="${YOLO_MODEL:-yolo26n.pt}"
BATCH="${BATCH:-128}"
EPOCHS="${EPOCHS:-50}"
NAME="${NAME:-coco_escooter_${YOLO_MODEL}_rep${REPEAT}}"
RESUME="${RESUME:-0}"

# python3 scripts/build_coco_merged_lists.py \
#   --coco-dir "$COCO_DIR" \
#   --split-dir "$SPLIT_DIR" \
#   --out-dir "$OUT_DIR" \
#   --repeat "$REPEAT"

TRAIN_ARGS=(
  --data "$OUT_DIR/data.yaml" \
  --model "$YOLO_MODEL" \
  --epochs "$EPOCHS" \
  --batch "$BATCH" \
  --imgsz 640 \
  --patience 20 \
  --workers "$WORKERS" \
  --seed 42 \
  --device 0 \
  --project data/runs \
  --name "$NAME" \
  --freeze "$FREEZE"
)
if [[ "$RESUME" == "1" ]]; then
  TRAIN_ARGS+=(--resume)
elif [[ "$RESUME" != "0" ]]; then
  echo "RESUME deve essere 0 oppure 1 (valore ricevuto: $RESUME)" >&2
  exit 2
fi

python3 scripts/train_yolo.py "${TRAIN_ARGS[@]}"
