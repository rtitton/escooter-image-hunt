# TODO

## 1. Tre training su H100 sul subset COCO 20000_4000, REPEAT=3 (un solo cambio ciascuno)

Obiettivo: capire quale leva riduce il divario dalla baseline monoclasse sul video test set (0,726 mAP50) senza peggiorare le classi COCO, in particolare la person. Riferimento: `models/yolo26n_cocosub_20000_4000_escooter_merge_rep3/` (yolo26n, 640, 100 epoche, batch 32, seme 42, `freeze` 0). Sul video il riferimento fa 0,606 / 0,429 (mAP50 / mAP50-95) a 640, 0,720 / 0,510 a 960 senza riallenare; sul kickboard 0,980 / 0,835 a 640, 0,924 / 0,671 a 960 (v. `VALUTAZIONI.md`, sezione 6).

Prerequisiti come nelle run precedenti: repo aggiornato (`git pull`), ambiente `uv`, nostro split, `yolo26n.pt` e `yolo26s.pt` nella radice, liste di REPEAT=3 già costruite in `data/processed/cocosub_20000_4000_escooter_merge_rep3/` (altrimenti: `make_coco_subset.sh` con `MAX_IMAGES=20000 MIN_PER_CLASS=4000`, poi `build_coco_merged_lists.sh` con `COCO_SUBSET_DIR=data/processed/coco_subset_20000_4000 REPEAT=3 OUT_DIR=data/processed/cocosub_20000_4000_escooter_merge_rep3`).

```bash
D=data/processed/cocosub_20000_4000_escooter_merge_rep3/data.yaml

# 1. risoluzione di training 960 (2-2,5 volte il tempo di 3,5 ore; verificare che il batch 32 stia in memoria)
DATA_YAML=$D EPOCHS=100 IMGSZ=960 NAME=yolo26n_cocosub_20000_4000_escooter_merge_rep3_imgsz960 \
    scripts/training/train_yolo26_coco_escooter.sh

# 2. modello più grande (FATTO: risultati in VALUTAZIONI.md, sezione 6, "yolo26s")
DATA_YAML=$D EPOCHS=100 YOLO_MODEL=yolo26s.pt NAME=yolo26s_cocosub_20000_4000_escooter_merge_rep3 \
    scripts/training/train_yolo26_coco_escooter.sh

# 3. backbone congelato (primi 10 layer)
DATA_YAML=$D EPOCHS=100 FREEZE=10 NAME=yolo26n_cocosub_20000_4000_escooter_merge_rep3_freeze10 \
    scripts/training/train_yolo26_coco_escooter.sh
```

Se una si interrompe: stesso comando con `RESUME=1` davanti (stessi `DATA_YAML` e `NAME`).

### A training concluso (per ciascuna run)

1. Copiare la cartella della run in `models/` (con i pesi: sono ignorati da git).
2. Test set esterni (classe escooter = 80), con `eval_testset.py --imgsz <S>`; per la run a 960 valutare sia a 960 sia a 640 (per le altre due basta 640, ma vale la pena anche 960, per vedere se il guadagno dell'inferenza a risoluzione maggiore si somma):
   ```bash
   uv run python3 scripts/training/eval_testset.py --weights models/<run>/weights/best.pt \
       --class-index 80 --imgsz 640 \
       --testset data/processed/video_testset_final data/processed/holdout_kickboard_a75qx_final \
       --out data/tmp/eval_<run>.json
   ```
3. Classi COCO: `compare_coco_models.py --baseline yolo26n.pt --trained models/<run>/weights/best.pt --coco-dir data/coco --out-dir data/tmp/coco_comparison_<run> --device 0 [--imgsz <S>]`, letto sulle 13 classi del subset, con la **person in evidenza** (rep3: 0,479 contro 0,510 del pretrained).
4. Aggiornare `VALUTAZIONI.md` (sezione 6) e committare la cartella senza pesi.

### Come leggere il risultato

- Un solo seme e 120 immagini nel video test set: differenze sotto 0,03 sono nel rumore. Un secondo seme di REPEAT=3 darebbe la misura del rumore.
- Il kickboard peggiora quando si alza la risoluzione di inferenza (0,835 → 0,671 di mAP50-95 da 640 a 960): se la run a 960 tiene entrambi i test set, è la risposta migliore.
- `freeze=10` potrebbe aiutare le classi COCO (backbone intatto) a spese della escooter.
