. scripts/.env

# Come train_yolo26n_baseline.sh (stesso split, yolo26n, solo escooter, seed, patience, 100 epoche) ma con
# imgsz 960 invece di 640: sul video test set la baseline ritrova solo il 15% degli escooter piccoli (<32px),
# a risoluzione maggiore dovrebbero restare più pixel utili.
# batch 8 invece di 16: a 960 i pixel per immagine sono 2,25 volte quelli a 640; con batch 8 il test breve
# ha usato circa 2,9 GB di GPU (4,3 GB disponibili), con batch 12 si rischierebbe l'out of memory.
# Unica altra differenza oltre a imgsz e batch: il nome della run.
# Valutare con eval_testset.py --imgsz 960 (il default è 640 e darebbe numeri non confrontabili).
uv run python3 scripts/training/train_yolo.py \
  --data data/processed/union_reviewed_coco_split/data.yaml \
  --model yolo26n.pt \
  --classes 80 \
  --epochs 100 \
  --batch 8 \
  --imgsz 960 \
  --patience 20 \
  --workers 4 \
  --seed 42 \
  --device 0 \
  --project data/runs \
  --name imgsz960_26n_escooter
