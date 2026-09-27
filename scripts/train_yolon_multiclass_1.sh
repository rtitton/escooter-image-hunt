. scripts/.env

# Stessa ricetta di escooter_only_11n (yolo11n, freeze 10, patience 20, batch automatico, 100 epoche),
# ma con person/bicycle/motorcycle oltre all'escooter: --classes 0 1 3 80 -> indici 0,1,2,3 (escooter = 3).
# Serve a capire se i falsi positivi su pedoni e ciclisti dipendono dall'aver allenato monoclasse.
# Valutare con eval_testset.py --class-index 3 (v. eval_multiclass.sh).
uv run python3 scripts/train_yolo.py \
  --data data/processed/union_reviewed_coco_split/data.yaml \
  --model yolo11n.pt \
  --classes 0 1 3 80 \
  --epochs 100 \
  --batch -1 \
  --imgsz 640 \
  --freeze 10 \
  --patience 20 \
  --device 0 \
  --project data/runs \
  --name multiclass_11n
