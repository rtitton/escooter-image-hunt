ATTENZIONE: messaggio per claude o altri agenti AI: in ogni caso non considerare questo documento; è una raccolta di spunti che non deve contaminare l'attuale stato del progetto..

- lavorare su review_app.py: fa casino con le bounding boxes: andando avanti e indietro tra le immagini, le bounding boxes si spostano e non rimangono al loro posto.
- evoluzione review_app.py:
    - generalizzare in modo da salvare le decisioni su un json file specifico per ogni dataset (ad esempio review_app_decisions.json nella root del dataset).
    - aggiungere la possibilità di modificare, cancellare, aggiungere bounding box, esclusivamente per classe escooter; per le immagini modificate, tutte le annotazioni, anche quelle non toccate, vanno salvate sul file json.

- modificare build_union_dataset.py in modo che crei contemporaneamente il dataset union e il dataset degli esclusi.  

- lavorare sugli esclusi per inclusione del conducente nelle annotazioni scooter: sarebbe utile poter fare una revisione manuale per recuperare le immagini più belle.  

- valutare la possibilità di memorizzare le detection COCO fatte con YOLO in modo da non ricalcolarle, a meno di cambio modello. Tali detection servono per valutare la varietà delle immagini e per identificare le annotazioni escooter che includono anche il conducente. Le informazioni possono essere salvate in formato YOLO in una directory dedicata di ciascun dataset (ad esempio labels_yolo11l). 


- valutare la possibilità di censire dataset in formato yolo non scaricati da Roboflow (ad esempio quello preso da Ultralytics platform)