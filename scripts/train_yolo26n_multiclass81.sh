. scripts/.env

# Come train_yolo26n_baseline.sh (stesso split, yolo26n, imgsz 640, batch 16, nessun freeze, patience 20,
# 100 epoche, workers 4, seed 42) ma su tutte le 81 classi (COCO 0-79 + escooter 80) invece della sola
# escooter: niente --classes, quindi si usa direttamente il data.yaml dello split, senza dataset filtrato.
# Motivo: nel multiclasse nano di prima (person/bicycle/motorcycle + escooter) il recall sulle escooter
# piccole del video test set è raddoppiato (0,15 -> 0,33); qui si sfruttano tutte le classi COCO riviste.
# La testa di classificazione ha 81 uscite invece delle 80 del modello COCO pretrained: i suoi pesi non si
# ricaricano e la testa riparte da zero (come nelle run monoclasse).
# Valutare con eval_testset.py --class-index 80 --imgsz 640 (la classe escooter qui ha indice 80, non 0).
uv run python3 scripts/train_yolo.py \
  --data data/processed/union_reviewed_coco_split/data.yaml \
  --model yolo26n.pt \
  --epochs 100 \
  --batch 16 \
  --imgsz 640 \
  --patience 20 \
  --workers 4 \
  --seed 42 \
  --device 0 \
  --project data/runs \
  --name multiclass81_26n
