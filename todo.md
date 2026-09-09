ATTENZIONE: messaggio per claude o altri agenti AI: in ogni caso non considerare questo documento; è una raccolta di spunti che non deve contaminare l'attuale stato del progetto..


- review_app: la selezione automatica ormai è matura: rifinire la app in vista del passaggio manuale: ordinamento (file, dataset, stato selezionata/scartata, phash) filtro (dataset, stato)
  togliere il seleziona con riserva: solo seleziona/scarta; aggiungere ripristino bbox originali.


ATTENZIONE: il ruolo di questo progetto è ben distinto rispetto a yolo-custom: qui si parte da un corpus esteso per ottenere un dataset union al meglio delle possibilità; yolo-custom è una pipeline per aggiungere a un dataset già stabilito, con tecniche di deduplicazione. questo progetto serve a scremare un grosso corpus di immagini, yolo-custom a fare lo step finale per integrarle in un dataset già esistente (esclusione dei simili, annotazione coco, ecc.).


DEDICARE UNA SESSIONE A METTERE IN ORDINE UN PO DI COSE
- evidenziare meglio i vari step nella console: ogni volta che dica dove sta leggendo, quanti file entrano, quanti escono
- chiedere uno script pipeline complessivo


ATTENZIONE: valutare fiftyone per la rifinitura del dataset (e anche per analisi avanzatem come ad esempio la deduplicazione). Chiedere a Claude di fare una summa dello stato attuale del progetto e poi se si poteva fare con fiftyone.


ATTENZIONE: annotando COCO con yolo pretrained, gemini consiglia confidenza conservativa (0.5), fare un test IOU rispetto alle annotazioni monopattino e di scartare l'annotazione COCO se c'è troppa sovrapposizione. Bisogna però tenere conto del fatto che le persone non possono essere escluse perché il più delle volte conducono il monopattino. Al limite considerare solo box COCO diverse da persona.
