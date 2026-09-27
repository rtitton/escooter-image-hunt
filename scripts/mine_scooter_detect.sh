. scripts/.env

# Hard-example mining su scooter-detectyolov8 (Ultralytics Platform, classe 1 = scooter): tiene le
# immagini dove il modello small manca o vede con poca confidenza un monopattino reale, ne fa
# il pHash contro il pool attuale e i due test set esterni, e le scrive in
# data/processed/mined_scooter-detectyolov8 pronte per review_app.py.
# GPU (device 0): se in parallelo gira un training, meglio --device cpu per non contendere GPU e memoria.
uv run python3 scripts/mine_hard_examples.py \
  data/ultralytics-platform/scooter-detectyolov8-394bb401 \
  --weights runs/detect/data/runs/escooter_only_26s/weights/best.pt \
  --escooter-classes 1 \
  --device 0 \
  --threads 4
