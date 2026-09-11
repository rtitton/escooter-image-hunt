ATTENZIONE: messaggio per claude o altri agenti AI: in ogni caso non considerare questo documento; è una raccolta di spunti che non deve contaminare l'attuale stato del progetto..



ATTENZIONE: il ruolo di questo progetto è ben distinto rispetto a yolo-custom: qui si parte da un corpus esteso per ottenere un dataset union al meglio delle possibilità; yolo-custom è una pipeline per aggiungere a un dataset già stabilito, con tecniche di deduplicazione. questo progetto serve a scremare un grosso corpus di immagini, yolo-custom a fare lo step finale per integrarle in un dataset già esistente (esclusione dei simili, annotazione coco, ecc.).


DEDICARE UNA SESSIONE A METTERE IN ORDINE UN PO DI COSE
- evidenziare meglio i vari step nella console: ogni volta che dica dove sta leggendo, quanti file entrano, quanti escono
- chiedere uno script pipeline complessivo


ATTENZIONE: annotando COCO con yolo pretrained, gemini consiglia confidenza conservativa (0.5), fare un test IOU rispetto alle annotazioni monopattino e di scartare l'annotazione COCO se c'è troppa sovrapposizione. Bisogna però tenere conto del fatto che le persone non possono essere escluse perché il più delle volte conducono il monopattino. Al limite considerare solo box COCO diverse da persona. Ad esempio: il pretrained non rileva il monopattino, se per caso mi rileva una bici che si sovrappone ad un monopattino, allora scarto la bici. L'unica eccezione è la persona: tutte le altre classi non hanno motivo di stare sopra il monopattino. NO non è vero. posso avere un oggetto dietro il monopattino.
