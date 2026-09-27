. scripts/.env

# Confronta lo small sul pool (escooter_only_26s) con lo small addestrato anche sulle candidate del mining
# (escooter_only_s_mined_v1) sui due test set esterni. Se il training viene rilanciato con lo stesso --name,
# Ultralytics aggiunge un suffisso -N alla cartella: aggiornare il path dei pesi.
uv run python3 scripts/eval_testset.py \
  --weights runs/detect/data/runs/escooter_only_26s/weights/best.pt \
            runs/detect/data/runs/escooter_only_s_mined_v1/weights/best.pt \
  --testset data/processed/video_testset_final data/processed/holdout_kickboard_a75qx_final \
  --device cpu \
  --out data/tmp/eval_report_mined.json
