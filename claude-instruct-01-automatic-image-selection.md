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
- ignorare immagini con scooter troppo lontano (bounding box troppo piccola rispetto all'immagine, p.es. soglia 0.1%)
- ignorare immagini troppo piccole (p.es. soglia 160000 pixel)
- ignorare immagini poco varie: si potrebbe usare un modello YOLO pretrained su COCO per escludere immagini che (oltre allo scooter) non hanno istenze di classi COCO (p.es. soglia almeno 1 istanza COCO).

## deduplicazione
I dataser Roboflow spesso sono frutto di fork e fusioni di dataset esistenti: è opportuno identificare e scartare doppioni esatti e anche immagini molto simili fra loro (p.es. soglia su distanza basata su perceptual hash).

## deduplicazione temporale (opzionale)
Diversi dataset sorgente sono campionamenti fitti di poche riprese video (nomi file tipo `frame_00000`, `frame_00010`, …). Una dedup a soglia singola su questi frame o li tiene tutti o collassa un'intera ripresa a una sola immagine. Opzionalmente (`select_images.py --temporal-dedup`) si raggruppano le immagini per `(dataset, split, clip)` — `clip` e indice di frame ricavati dal nome — e in ogni gruppo, percorso in ordine di indice, si scarta un frame solo se è visivamente vicino all'ultimo frame tenuto (Hamming del pHash sotto soglia) ed entro un intervallo di indici massimo; primo e ultimo frame del gruppo si tengono sempre. Serve a mantenere solo i keyframe di ciascuna ripresa.




dry run