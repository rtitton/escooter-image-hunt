# Valutazioni sul dataset `union_reviewed_coco` e sui primi modelli

Registro delle analisi e delle valutazioni fatte fra il 26 ottobre e il 2 novembre 2026: qualità dello split, esito della seconda review, baseline yolo26n, test su due test set indipendenti ed esperimento con `imgsz` 960. Per la descrizione degli script vedi [PIPELINE.md](PIPELINE.md).

- [Sintesi](#sintesi)
- [1. Dataset e split](#1-dataset-e-split)
- [2. Review delle label](#2-review-delle-label)
- [3. Baseline yolo26n a 640](#3-baseline-yolo26n-a-640)
- [4. Test set indipendenti](#4-test-set-indipendenti)
- [5. Esperimento imgsz 960 e sweep sulla risoluzione](#5-esperimento-imgsz-960-e-sweep-sulla-risoluzione)
- [6. Confronto con le run precedenti](#6-confronto-con-le-run-precedenti)
- [7. Limiti delle misure](#7-limiti-delle-misure)
- [8. Prossimi passi](#8-prossimi-passi)
- [9. Come riprodurre](#9-come-riprodurre)

## Sintesi

- Sul **valid dello split** la baseline arriva a mAP50 0,964 e mAP50-95 0,742: un numero ottimistico.
- Sul **video test set** (frame da video, molti oggetti piccoli) la stessa baseline scende a mAP50 0,726 e mAP50-95 0,502; sul **kickboard** (oggetti grandi) regge: 0,972 e 0,816.
- Il collo di bottiglia sono gli **escooter piccoli** (<32 px): la baseline ne ritrova il 15%. Nel train sono 52 su 2765 istanze (1,9%).
- **Allenare a 960 aiuta poco** (video: mAP50 0,731 contro 0,726) e costa il doppio del tempo (7,1 ore contro 3,3). Usare la baseline a 960 *senza riaddestrarla* dà di più sul video (0,789) ma peggiora il kickboard (0,834). Nessuna delle due risoluzioni vince su entrambi i test set.
- La leva più promettente è **aggiungere esempi piccoli** al train.

## 1. Dataset e split

Dataset sorgente: `data/processed/union_reviewed_coco`, 2085 immagini, classi COCO (0-79) più escooter (80), revisione manuale. Split con `split_dataset.py`, seme 42, `--valid-frac` 0,15.

### Leakage trovato e corretto

L'analisi del primo split ha trovato fuga di informazione fra train e valid:

- 4 coppie speculari identiche e circa 21 copie della stessa foto con augmentation Roboflow (flip, rotazioni, cutout, colore), con prefisso dataset diverso.
- Le copie augmentate arrivano a distanze pHash di 20-30, quindi un filtro semplice a soglia bassa non le vede.
- Un'immagine speculare di una immagine di train che finisce in valid è leakage a tutti gli effetti: Ultralytics usa `fliplr=0.5`, quindi il modello l'ha di fatto già vista.

Correzioni:

- `split_dataset.py` unisce i gruppi anche fra dataset diversi quando hanno lo stesso nome-base (senza prefisso dataset, senza suffisso Roboflow, senza `_flip`). Fughe per nome: da 20 a 0.
- `split_review_app.py` ha le schede *Coppie sospette* (pHash, anche speculare, e stesso nome-base) e *Copie sospette yolodet* (stesse classi e box delle detection originali che si sovrappongono, con N e tolleranza regolabili). Con la seconda sono state trovate 2 coppie speculari ancora attraverso i due lati; sono state sistemate escludendo 3 immagini dal train.
- I falsi positivi della scheda yolodet si scartano a colpo d'occhio: serve come strumento di rilevazione, non come criterio automatico.
- Gli embedding non sono stati usati (provati in un altro progetto, deludenti).

### Composizione dello split corrente

| | Train | Valid |
|---|---|---|
| Immagini | 1812 | 270 (12,9%) |
| Istanze escooter | 2765 | 424 |
| Area mediana dell'istanza (% immagine) | 10,11 | 10,31 |
| Istanze piccole (<32²) | 52 | 14 |

- In valid ci sono anche 74 istanze medie (<96²).
- Nessun file in comune fra i due lati, nessun nome-base condiviso.
- Sei dataset sorgente non hanno immagini in valid (479 istanze escooter, il 17% del train): il valid non copre quei domini.
- Distribuzioni di taglia e istanze per immagine bilanciate fra i due lati.

## 2. Review delle label

Seconda review completa di tutte le immagini (la prima parte del dataset aveva molti falsi negativi, la qualità era cresciuta durante la prima passata). Confronto con le box originali di yolo11l (backup in `review_label_backups`, che contiene sempre la prima versione):

| | Misura intermedia (a metà della seconda review) | A review conclusa |
|---|---|---|
| Immagini con backup | 1034 | 1177 |
| Box invariate | 5506 | 6263 |
| Box aggiunte (falsi negativi) | 1758 | 2240 |
| Box rimosse | 209 | 229 |
| Box ritoccate | 250 | 296 |

- Le box ritoccate non sono allargate in modo sistematico: rapporto d'area mediano 1,00 (138 più grandi, 121 più piccole).
- Il dataset è coerente: nessun label mancante o orfano, nessun file vuoto, nessuna box duplicata.
- 185 box uscivano dal bordo dell'immagine (fino al 9,4%), create trascinando oltre il bordo in `review_app.py`: ritagliate con `clip_label_boxes.py`, e `review_app.py` ora impedisce che accada.

### Convenzioni adottate per le classi COCO

Poiché il dataset viene unito a COCO, per le classi 0-79 vale la convenzione COCO: box sul solo visibile (niente box amodali), `truck` come in COCO (pick-up e furgoni inclusi, nessuna riclassificazione). Per escooter la convenzione sulle occlusioni non è fissata.

Verifica ancora da fare: un campione delle ~2240 box aggiunte che si sovrappongono molto ad altre o toccano il bordo, per capire se alcune sono amodali.

## 3. Baseline yolo26n a 640

`scripts/train_yolo26n_baseline.sh`: yolo26n, solo escooter (`--classes 80`, la classe diventa 0 in un dataset filtrato), 100 epoche, patience 20, imgsz 640, batch 16, nessun freeze, augmentation di default, workers 4, seed 42. Run: `runs/detect/data/runs/baseline_26n_escooter`.

Risultati sul valid (migliore epoca, la 82):

| Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|
| 0,965 | 0,898 | 0,964 | 0,742 |

- Ha usato tutte le 100 epoche; l'ultima ha mAP50-95 0,740.
- La mAP50-95 sale fino all'epoca 57 circa (0,72) e poi cresce di un punto in circa 25 epoche; la loss di validazione è ferma, quella di train scende.
- Il calo della loss di train dopo l'epoca 90 è il mosaic che Ultralytics spegne nelle ultime 10 epoche.
- Tempo: mediana 120 s per epoca (circa 3 ore e 20 minuti). Un'epoca (la 91) riporta 29 ore: il PC è stato probabilmente sospeso e ha ripreso, il processo non è morto. Non verificato sul registro di Windows.
- Perde il 10% circa delle istanze (recall 0,898 contro precision 0,965).

## 4. Test set indipendenti

Due test set esterni già nel progetto, valutati con `eval_testset.py` (`--class-index 0`):

- `video_testset_final`: 120 immagini, 323 box escooter (86 piccole, 145 medie, 92 grandi). Frame da video.
- `holdout_kickboard_a75qx_final`: 90 immagini, 98 box escooter (quasi tutte grandi).

Baseline a 640:

| Test set | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|
| Valid dello split | 0,965 | 0,898 | 0,964 | 0,742 |
| `holdout_kickboard_a75qx_final` | 0,986 | 0,939 | 0,972 | 0,816 |
| `video_testset_final` | 0,809 | 0,610 | **0,726** | **0,502** |

Scomposizione per taglia (conf ≥ 0,25, IoU ≥ 0,5):

| Test set | Taglia | Box | Recall | Falsi positivi |
|---|---|---|---|---|
| Video | piccole | 86 | **0,151** | 5 |
| Video | medie | 145 | 0,738 | 45 |
| Video | grandi | 92 | 0,859 | 12 |
| Kickboard | grandi | 97 | 0,948 | 13 |

Il valid dello split è quindi troppo ottimista: il video test set è più realistico, e il problema sono gli oggetti piccoli (nel valid ce ne sono solo 14, per questo non emergevano).

## 5. Esperimento imgsz 960 e sweep sulla risoluzione

`scripts/train_yolo26n_imgsz960.sh`: stessa ricetta della baseline, ma `imgsz` 960 e `batch` 8 (con batch 12 la GTX 1050 Ti da 4,3 GB andrebbe in out of memory; il test breve ha usato 2,9 GB). Run: `runs/detect/data/runs/imgsz960_26n_escooter`.

- Valid (epoca 88): P 0,955, R 0,902, mAP50 0,958, mAP50-95 0,739, praticamente uguale alla baseline.
- Tempo: 255 s per epoca, 7,09 ore per 100 epoche (3,6 volte più lento per immagine, 2,25 per i soli pixel).

Valutazione di entrambi i modelli a risoluzioni di inferenza diverse, su GPU (mAP50 / mAP50-95 e recall sui piccoli del video):

| Modello | imgsz inferenza | Video mAP50 | Video mAP50-95 | Video recall piccoli | Kickboard mAP50 | Kickboard mAP50-95 |
|---|---|---|---|---|---|---|
| Baseline (train 640) | 640 | 0,726 | 0,502 | 0,151 | 0,972 | 0,816 |
| Baseline | 800 | 0,753 | 0,525 | 0,233 | 0,943 | 0,677 |
| Baseline | 960 | **0,789** | **0,548** | **0,419** | 0,834 | 0,520 |
| Train 960 | 640 | 0,639 | 0,439 | 0,070 | 0,991 | 0,857 |
| Train 960 | 800 | 0,683 | 0,476 | 0,163 | 0,993 | 0,849 |
| Train 960 | 960 | 0,731 | 0,525 | 0,233 | 0,988 | 0,804 |
| Train 960 | 1280 | 0,752 | 0,534 | 0,291 | 0,948 | 0,667 |

Dettaglio a 960 (precision / recall / FP):

| Modello a 960 | Video P | Video R | Piccole recall (FP) | Medie recall (FP) | Grandi recall (FP) | Kickboard P / R | Kickboard grandi recall (FP) |
|---|---|---|---|---|---|---|---|
| Baseline | 0,824 | 0,740 | 0,419 (5) | 0,821 (35) | 0,880 (10) | 0,745 / 0,837 | 0,887 (41) |
| Train 960 | 0,853 | 0,641 | 0,233 (3) | 0,738 (25) | 0,891 (9) | 0,979 / 0,949 | 0,979 (4) |

Letture:

- **Allenare a 960 aiuta poco:** il risultato sul video (0,731 di mAP50) è vicino alla baseline a 640 (0,726). Il recall sui piccoli passa da 0,151 a 0,233, cioè 7 box in più su 86.
- **Il vantaggio viene in gran parte dall'inferenza a più pixel:** la baseline usata a 960, senza riaddestrarla, ritrova il 42% dei piccoli del video.
- **Non è un guadagno netto:** la baseline a 960 sul kickboard crolla (0,972 → 0,834) con 41 falsi positivi sui grandi. L'ingrandimento aumenta la sensibilità ma abbassa la precisione sui grandi.
- **Ogni dominio ha la sua risoluzione migliore:** il video vuole alta risoluzione, il kickboard sta meglio a 640-800. Il modello allenato a 960 è il più equilibrato, ma non vince in nessuno dei due.
- Il modello allenato a 960 va usato a 960 (a 640 perde molto sul video: 0,639).
- Costo dell'inferenza a 960: circa 12 GFLOPs contro 5,3 a 640 (2,25 volte).

## 6. Confronto con le run precedenti

Le run precedenti (`escooter_only_11n`, `escooter_only_nofreeze_v1`) sono state fatte su uno split diverso, quindi il confronto è indicativo. Test set indipendenti, imgsz 640:

| Run | Video mAP50 / mAP50-95 | Kickboard mAP50 / mAP50-95 |
|---|---|---|
| `escooter_only_11n` | 0,683 / 0,537 | 0,974 / 0,816 |
| `escooter_only_nofreeze_v1` | 0,711 / 0,486 | 0,969 / 0,788 |
| `baseline_26n_escooter` | 0,726 / 0,502 | 0,972 / 0,816 |

Le differenze sono piccole su 323 box: il passaggio a yolo26n non fa la differenza da solo.

## 7. Limiti delle misure

- **Valid piccolo:** 270 immagini e 424 istanze, di cui 14 piccole. Differenze di un punto di mAP50-95 non sono significative.
- **Video test set:** 323 box in 120 frame, probabilmente da pochi clip, quindi correlati. Sotto 2-3 punti non conta.
- **Selezione sul valid:** `best.pt` è l'epoca migliore sul valid, quindi quel numero ha una piccola distorsione verso l'alto.
- **Dominio:** sei dataset sorgente non hanno immagini in valid.
- **Soglia della scomposizione per taglia:** conf ≥ 0,25 e IoU ≥ 0,5; non è una AP per taglia.
- **Valutazioni su CPU:** la baseline a 640 è stata valutata su CPU, le altre su GPU; le metriche dovrebbero coincidere, i tempi per immagine no.
- **Sospensione del PC:** le durate misurate nella baseline sono corrette dalla mediana per epoca (120 s); il tempo totale nel `results.csv` include la sospensione.

## 8. Prossimi passi

1. Aggiungere esempi di escooter piccoli al train (nel train sono l'1,9% delle istanze), per esempio con immagini del tipo del video test set, e riallenare a 640 (circa 3 ore); valutare poi a 640 e a 800.
2. Verificare a campione le box aggiunte che si sovrappongono ad altre o toccano il bordo, per vedere se alcune sono amodali (v. sezione 2).
3. Se serve una validazione più affidabile, ampliare il valid con dataset che oggi non vi sono rappresentati.
4. Decidere la risoluzione di inferenza in base all'uso finale: il costo a 960 è 2,25 volte.

## 9. Come riprodurre

```
# split e verifica
python3 scripts/split_dataset.py data/processed/union_reviewed_coco
scripts/split_review_union_reviewed_coco.sh

# correzione dei bordi (con --dry-run per vedere prima)
python3 scripts/clip_label_boxes.py data/processed/union_reviewed_coco

# training
scripts/train_yolo26n_baseline.sh
scripts/train_yolo26n_imgsz960.sh

# valutazione sui test set indipendenti (--imgsz come in training)
python3 scripts/eval_testset.py \
  --weights runs/detect/data/runs/baseline_26n_escooter/weights/best.pt \
  --testset data/processed/video_testset_final data/processed/holdout_kickboard_a75qx_final \
  --class-index 0 --device 0 --imgsz 640 --out data/tmp/eval_report_baseline_26n.json
```

Report JSON delle valutazioni in `data/tmp/`: `eval_report_baseline_26n.json`, `eval_960_at960.json`, `eval_base_at960.json`, `eval_base_at800.json`, `eval_960_at640.json`, `eval_960_at800.json`, `eval_960_at1280.json`.
