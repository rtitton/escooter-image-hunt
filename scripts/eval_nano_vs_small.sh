. scripts/.env

# Confronta il nano (escooter_only_11n, 79 epoche early stop) e lo small (escooter_only_s_v1-5,
# 100 epoche) sui due test set esterni (mai visti in training): i frame dai video in
# /mnt/x/media e l'holdout kickboard_a75qx. --device cpu di proposito, per non contendere la
# GPU a un training in corso. I path dei pesi vanno aggiornati se si rilancia un training con
# lo stesso --name (Ultralytics aggiunge un suffisso -N alla cartella invece di sovrascrivere).
uv run python3 scripts/eval_testset.py \
  --weights runs/detect/data/runs/escooter_only_11n/weights/best.pt \
            runs/detect/data/runs/escooter_only_26s/weights/best.pt \
  --testset data/processed/video_testset_final data/processed/holdout_kickboard_a75qx_final \
  --device cpu \
  --out data/tmp/eval_report.json
