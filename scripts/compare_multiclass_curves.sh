. scripts/.env

# Confronto ogni 10 epoche, sul valid interno e sulla sola classe escooter, fra il nano monoclasse
# (escooter_only_11n) e il nano multiclasse in training (multiclass_11n, escooter = indice 3).
# Richiede gli snapshot di last.pt prodotti da snapshot_checkpoints.py, avviato insieme al training:
#   python3 scripts/snapshot_checkpoints.py runs/detect/data/runs/multiclass_11n --every 10
# Le epoche già valutate sono in cache (snapshots/eval_cache.json): si può rilanciare a ogni nuovo snapshot.
uv run python3 scripts/compare_training_curves.py \
  --baseline-run runs/detect/data/runs/escooter_only_11n \
  --run runs/detect/data/runs/multiclass_11n \
  --class-index 3 --every 10
