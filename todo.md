ATTENZIONE: messaggio per claude o altri agenti AI: in ogni caso non considerare questo documento; è una raccolta di spunti che non deve contaminare l'attuale stato del progetto..

- nuova euristica per contaminazioni scooter-conducente implementata: testare su corpus completo

- cambiato tracciato del csv: due flag, uno di abilitazione del record, altro per download (sostituisce status)

- implementata cache per perceptual hash

- valutare la possibilità di censire dataset in formato yolo non scaricati da Roboflow (ad esempio quello preso da Ultralytics platform): è sufficiente specificare local nello stato e trattarli a parte, assumendo che abbiano lo stesso formato di Roboflow (train/images, valid/images, test/images, train/labels, valid/labels, test/labels) e che siano copiati nella directory raw.

