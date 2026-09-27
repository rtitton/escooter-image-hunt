. scripts/.env
uv run python3 scripts/train_yolo.py \
  --data /home/cric/devp/escooter-image-hunt/data/ultralytics-platform/scooter-detectyolov8-394bb401/data.yaml \
  --model yolo26n.pt \
  --epochs 100 \
  --batch 16 \
  --imgsz 640 \
  --freeze 10 \
  --patience 20 \
  --device 0 \
  --project data/runs \
  --name ultralytics_scooter_1
