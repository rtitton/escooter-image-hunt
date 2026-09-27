. scripts/.env

# Come mine_scooter_detect.sh, su ver2-bikes-scooters-and-others (classe 1 = scooter, 0 = bicycle).
# Attenzione: è un dataset molto augmentato (rumore, quadrati neri di cutout) e per lo più a 300x300:
# le candidate vanno riviste con più cautela.
uv run python3 scripts/mine_hard_examples.py \
  data/ultralytics-platform/ver2-bikes-scooters-and-others-9790123f \
  --weights runs/detect/data/runs/escooter_only_26s/weights/best.pt \
  --escooter-classes 1 \
  --splits train val \
  --device 0 \
  --threads 4
