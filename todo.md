ATTENZIONE: messaggio per claude o altri agenti AI: in ogni caso non considerare questo documento; è una raccolta di spunti che non deve contaminare l'attuale stato del progetto..

DOVE SIAMO RIMASTI: stavo esaminando i contaminati e ho scoperto che vengono selezionate immagini con annotazioni di scooter perfette, anziché includenti il conducente.

- **problema** con la determinazione delle annotazioni uniche scooter+conducente: esclude immagini in cui è chiaramente annotato solo lo scooter, ad esempio `e-scooters-detection-base-new-v1__00000013_000_jpg.rf.1b57092c296d6852d2d3d9967658bdd8.jpg`. Per avere contaminazione l'annotazione YOLO per la persona deve sovrapporsi a quella dello scooter+conducente originale. 

- modificare build_union_dataset.py in modo che crei contemporaneamente il campione per controllo visivo. Introdurre la possibilità di selezionare i dataset a questo livello (altro campo nel csv? chiedere consiglio a claude)

- mettere in cache anche il perceptual hash: dura un minuto ma è una rottura.  

- valutare la possibilità di censire dataset in formato yolo non scaricati da Roboflow (ad esempio quello preso da Ultralytics platform): è sufficiente specificare local nello stato e trattarli a parte, assumendo che abbiano lo stesso formato di Roboflow (train/images, valid/images, test/images, train/labels, valid/labels, test/labels) e che siano copiati nella directory raw.

- review_app.py : mostrare la directory del dataset in review