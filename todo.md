ATTENZIONE: messaggio per claude o altri agenti AI: in ogni caso non considerare questo documento; è una raccolta di spunti che non deve contaminare l'attuale stato del progetto..

- modificare build_union_dataset.py in modo che crei contemporaneamente il campione per controllo visivo. Introdurre la possibilità di selezionare i dataset a questo livello (altro campo nel csv? chiedere consiglio a claude)

- lavorare sugli esclusi per inclusione del conducente nelle annotazioni scooter: sarebbe utile poter fare una revisione manuale per recuperare le immagini più belle.  

- mettere in cache anche il perceptual hash

- valutare la possibilità di censire dataset in formato yolo non scaricati da Roboflow (ad esempio quello preso da Ultralytics platform)

- nuovo script apply_selection.py per creare la versione selezionata di ciascun dataset, utile per visione. in interim per ogni dataset una cartella con id-dataset-selected.  

- review_app.py : mostrare anche la risoluzione dell'immagine (se non lo fa gia)