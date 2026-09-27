. scripts/.env

# Stessa ricetta di train_yolos_1.sh (yolo26s, batch 16, freeze 10, patience 20, --classes 80), sullo split
# con le candidate del mining aggiunte al train (v. build_split_mined.sh): confrontabile con escooter_only_26s.
uv run python3 scripts/train_yolo.py \
  --data data/processed/union_reviewed_coco_split_mined/data.yaml \
  --model yolo26s.pt \
  --classes 80 \
  --epochs 100 \
  --batch 16 \
  --imgsz 640 \
  --freeze 10 \
  --patience 20 \
  --device 0 \
  --project data/runs \
  --name escooter_only_s_mined_v1
