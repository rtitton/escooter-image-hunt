#!/usr/bin/env bash
# Training a 81 classi (COCO 0-79 + escooter 80) su COCO + il nostro split, yolo26n pretrained COCO.
# Pensato per la macchina con la H100, dove c'è COCO (formato YOLO di Ultralytics: images/train2017,
# labels/train2017, ...). Da lanciare dalla radice del repo clonato lì, dopo aver copiato il nostro split
# (data/processed/union_reviewed_coco_split) e yolo26n.pt (non è nel repo: .gitignore ignora i .pt).
# Richiede bash (usa array): non lanciarlo con "sh". Usa "uv run python3" come gli altri script, quindi sulla
# macchina serve uv con l'ambiente del progetto.
#
# Il dataset (liste e data.yaml) si costruisce prima, a parte, con build_coco_merged_lists.sh (REPEAT, subset di
# COCO, ecc. si scelgono lì). Qui si lancia solo il training:
#   DATA_YAML=data/processed/coco_escooter_merge_rep5/data.yaml scripts/training/train_yolo26_coco_escooter.sh
#
# Variabili d'ambiente (DATA_YAML è obbligatoria, le altre opzionali: i default sono quelli della macchina H100
# e della prima run, 150 epoche):
#   DATA_YAML    data.yaml del dataset da usare (output di build_coco_merged_lists.sh)
#   YOLO_MODEL   pesi di partenza (default yolo26n.pt)
#   EPOCHS (150), BATCH (32), WORKERS (16: sulla H100 il collo di bottiglia è la CPU), FREEZE (0)
#   NAME         nome della run (default <modello>_<cartella del data.yaml>, es.
#                yolo26n_coco_escooter_merge_rep5)
#   RESUME       1 = riprende la run interrotta chiamata NAME (da last.pt, v. sotto)
#
# Per interrompere e riprendere in seguito: Ctrl+C e poi, con lo stesso NAME e gli stessi parametri,
#   RESUME=1 DATA_YAML=<stesso data.yaml> NAME=<nome della run> scripts/training/train_yolo26_coco_escooter.sh
#
# Ricetta: pesi pretrained COCO, patience 20, imgsz 640, seed 42, nessun cls_pw (le classi COCO rare non vanno
# protette). La testa di classificazione ha 81 uscite invece delle 80 dei pesi COCO: quei pesi non si ricaricano
# e la testa riparte da zero, mentre backbone e neck partono pretrained; con COCO nel train dovrebbe recuperare.
# Valutare poi sui test set esterni (copiando best.pt sulla macchina locale):
#   eval_testset.py --weights ... --class-index 80 --imgsz 640 (v. VALUTAZIONI.md)
set -e

DATA_YAML="${DATA_YAML:-}"
YOLO_MODEL="${YOLO_MODEL:-yolo26n.pt}"
EPOCHS="${EPOCHS:-150}"
BATCH="${BATCH:-32}"
WORKERS="${WORKERS:-16}"
FREEZE="${FREEZE:-0}"
RESUME="${RESUME:-0}"

if [[ -z "$DATA_YAML" || ! -f "$DATA_YAML" ]]; then
  echo "DATA_YAML deve essere un data.yaml esistente (v. build_coco_merged_lists.sh); ricevuto: '$DATA_YAML'" >&2
  exit 2
fi
if [[ "$RESUME" != "0" && "$RESUME" != "1" ]]; then
  echo "RESUME deve essere 0 oppure 1 (valore ricevuto: $RESUME)" >&2
  exit 2
fi
NAME="${NAME:-${YOLO_MODEL%.pt}_$(basename "$(dirname "$(realpath "$DATA_YAML")")")}"  # senza ".pt" nel nome

TRAIN_ARGS=(
  --data "$DATA_YAML"
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

uv run python3 scripts/training/train_yolo.py "${TRAIN_ARGS[@]}"
