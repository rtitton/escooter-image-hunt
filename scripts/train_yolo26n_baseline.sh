. scripts/.env

# Baseline sullo split di union_reviewed_coco così com'è (1815 train / 270 valid), yolo26n.
# Solo escooter (--classes 80 -> classe 0), come le run escooter_only precedenti, per poterle confrontare.
# Per allenare su tutte le 81 classi (COCO + escooter) togliere --classes.
# Ricetta volutamente semplice: nessun freeze, augmentation di default di Ultralytics, seed fisso.
# batch 16 fisso invece di -1 (autobatch): stesso batch in ogni run, confronti riproducibili
# (sulla GTX 1050 Ti da 4,3 GB è lo stesso valore delle run precedenti, ci sta con yolo nano a 640).
# workers 4: la macchina ha 9 GB di RAM, con 8 worker il dataloader rischia di strozzarla.
uv run python3 scripts/train_yolo.py \
  --data data/processed/union_reviewed_coco_split/data.yaml \
  --model yolo26n.pt \
  --classes 80 \
  --epochs 100 \
  --batch 16 \
  --imgsz 640 \
  --patience 20 \
  --workers 4 \
  --seed 42 \
  --device 0 \
  --project data/runs \
  --name baseline_26n_escooter
