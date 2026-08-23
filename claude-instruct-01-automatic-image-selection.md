# selezione automatica di immagini dai dataset roboflow scaricati e pre-elaborati

La selezione deve rispettare i criteri descritti sotto.  
La procedura non esegue copie di immagini ma compila un file di testo con le immagini candidate.  
La copia effettiva delle immagini in un nuovo dataset di unione è uno step separato che parte dal file di testo con le immagini candidate.

## situazione di partenza
Abbiamo una serie di dataset scaricati da Roboflow ed elaborati eliminando le immagini generate da augmentation (criterio euristico) e convertendo eventuali poligoni in bounding box.  
Dataset modificati si trovano in data/interim e sono censiti in data/dataset.json.

## selezione
- ignorare immagini senza istanze delle classi scooter
- ignorare immagini di primo piano su scooter (dimensione della bounding box copre quasi tutta l'immagine, p.es. soglia 80%)
- ignorare immagini troppo piccole (p.es. soglia 160000 pixel)
- ignorare immagini poco varie: si potrebbe usare un modello YOLO pretrained su COCO per escludere immagini che (oltre allo scooter) non hanno istenze di classi COCO (p.es. soglia almeno 1 istanza COCO).

## deduplicazione
I dataser Roboflow spesso sono frutto di fork e fusioni di dataset esistenti: è opportuno identificare e scartare doppioni esatti e anche immagini molto simili fra loro (p.es. soglia su distanza basata su perceptual hash).




dry run