. scripts/.env
uv run python3 scripts/train_yolo.py \
  --data data/processed/union_reviewed_coco_split/data.yaml \
  --model yolo11n.pt \
  --classes 80 \
  --epochs 100 \
  --batch 16 \
  --imgsz 640 \
  --patience 20 \
  --device 0 \
  --project data/runs \
  --name escooter_only_nofreeze_v1
