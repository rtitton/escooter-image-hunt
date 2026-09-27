. scripts/.env

# Confronta il nano monoclasse (escooter_only_11n, escooter = indice 0) con il nano multiclasse
# (multiclass_11n: person, bicycle, motorcycle, escooter = indice 3) sui due test set esterni, sulla sola
# classe monopattino. --class-index vale per tutti i pesi di una invocazione, quindi due chiamate.
# Da lanciare a training finito. Se la run viene rifatta con lo stesso --name, Ultralytics aggiunge un
# suffisso -N alla cartella: aggiornare il path dei pesi.
TESTSETS="data/processed/video_testset_final data/processed/holdout_kickboard_a75qx_final"

uv run python3 scripts/eval_testset.py \
  --weights runs/detect/data/runs/escooter_only_11n/weights/best.pt \
  --testset $TESTSETS --class-index 0 --device cpu \
  --out data/tmp/eval_report_mono_11n.json

uv run python3 scripts/eval_testset.py \
  --weights runs/detect/data/runs/multiclass_11n/weights/best.pt \
  --testset $TESTSETS --class-index 3 --device cpu \
  --out data/tmp/eval_report_multiclass_11n.json
