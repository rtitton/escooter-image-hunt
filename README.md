# escooter-image-hunt

## Obiettivo
Costruire un dataset di immagini per il **rilevamento (object detection)** di monopattini elettrici (e-scooter), da usare per addestrare/validare un modello di computer vision YOLO, partendo da un modello pretrained su COCO di Ultralytics.  
Il modello finale deve riconoscere le classi COCO e in aggiunta la classe e-scooter.  

## Fonti dati
- **Roboflow** — dataset pubblici esistenti con annotazioni già pronte. Il problema della maggior parte dei dataset pubblici è la scarsa qualità delle immagini: immagini a bassa risoluzione, immagini molto simili fra loro, immagini molto focalizzate sull'oggetto e-scooter e quindi con poca varietà di classi, scarsità di istanze e-scooter piccole. 

## Selezione delle immagini
Implementare un criterio di qualità sulle immagini da utilizzare per selezionare immagini dai dataset Roboflow scaricati in locale e formare un nuovo dataset.

## Aspetti pratici
- formato per i dataset Roboflow scaricati: YOLO
- i dataset con annotazioni a poligono (maschere di segmentazione) vengono convertiti in bounding box: il box minimo che contiene esattamente il poligono (min/max delle coordinate dei vertici). La conversione avviene nella versione interim, il raw resta poligono se tale era in origine.
- i vari dataset avranno in generale id di classi non uniformi e potrebbero contenere, oltre a e-scooter, anche classi COCO e altre classi non COCO.
- i dataset scaricati vanno deduplicati dalle immagini generate da augmentation (Roboflow le aggiunge tipicamente solo allo split di train): si raggruppano i file per nome-base e si copia una sola immagine per gruppo (preferendo quella senza segni di rotazione, quando disponibile) in `data/interim/<id>-dedup/`, lasciando `data/raw/<id>/` invariato (`scripts/dedupe_augmented.py`, che applica anche la conversione poligono→bbox sopra descritta).
- sul dataset unione manteniamo solo la classe e-scooter e la rimappiamo con id 80
- sul dataset unione annotiamo tutte le classi COCO usando un modello Ultralytics pretrained di grandi dimensioni per avere massima accuratezza.
- il problema principale del progetto è la definizione del criterio di qualità per la selezione delle immagini; gli altri step sono automatismi relativamente semplici.
