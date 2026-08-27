ATTENZIONE: messaggio per claude o altri agenti AI: in ogni caso non considerare questo documento; è una raccolta di spunti che non deve contaminare l'attuale stato del progetto..

- aggiungere un parametro per filtrare la dimensione delle bounding box piccole: potrebbe essere una percentuale della dimensione dell'immagine (come già presente per il massimo), oppure una dimensione minima in pixel.  
  fare un giro con il filtro attivo e vedere l'effetto che fa rispetto a prima. configurazione con costante in select_images.py: testato con filtro attivo, cambia pochissimo (passa una manciata di immagini in più, sarebbe comunque carino individuarle e vederle)

- levare dalle palle le rotazioni: sono entrate per beccare le augmented e poi anche per rilevare le contaminazioni.  
  fare un giro con nessuna rotazione e vedere l'effetto che fa rispetto a prima. configurazione con costante in select_images.py: testato con nessuna rotazione, cambia pochissimo (passa una manciata di immagini in più, sarebbe comunque carino individuarle e vederle)

- review_app: aggiungere la possibilità di spostare le immagini in un dataset di riferimento: il dataset di riferimento è  union; il dataset in review può essere union stesso (spostamento disabilitato), oppure un altro dataset (uno di quelli scartati, in questo caso è possibile con un tasto comandare lo spostamento dell'immagine).
  togliere il seleziona con riserva: non ha più senso da quando posso modificare le bounding box.
  aggiungere la possibilità di fare restore delle annotazioni originali (sono salvate su file a parte)

- aggiungere il dataset delle immagini scartate in fase di dedup iniziale e da select finale. Chiedi a claude che individui i punti in cui vengono scartate immagini e che generi un elenco come selected_images e flagged_rider_contamination.txt (cambiare i nomi magari)


ATTENZIONE: il ruolo di questo progetto è ben distinto rispetto a yolo-custom: qui si parte da un corpus esteso per ottenere un dataset union al meglio delle possibilità; yolo-custom è una pipeline per aggiungere a un dataset già stabilito, con tecniche di deduplicazione. questo progetto serve a scremare un grosso corpus di immagini, yolo-custom a fare lo step finale per integrarle in un dataset già esistente (esclusione dei simili, annotazione coco, ecc.).


DEDICARE UNA SESSIONE A METTERE IN ORDINE UN PO DI COSE
- evidenziare meglio i vari step nella console: ogni volta che dica dove sta leggendo, quanti file entrano, quanti escono
- fare un report finale in ogni script meglio delle attuali print.
Quando fai questo, fai un giro completo e salva i numeri; poi lascia la logica ferma, fai le modifiche di cosmesi, giro completo e confronta i numeri.
