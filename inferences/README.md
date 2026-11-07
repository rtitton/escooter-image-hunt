# inferences/ — inferenze dei modelli su video

Output generati, non versionati (v. `.gitignore`): si rigenerano con gli script.

```
inferences/
  <modello>/<video>/<video>_f<indice>.jpg   frame con le box della classe escooter disegnate
  _confronto/<video>/<video>_f<indice>.jpg  fogli: lo stesso frame di tutti i modelli, affiancati
```

- `python3 scripts/infer_videos.py /mnt/x/media` produce le cartelle dei modelli: un frame ogni 10 del video (`--step`), soglia di confidenza 0,25 (`--conf`), modelli di default in `MODELLI_DEFAULT` o scelti con `--model NOME=PESI[:CLASSE[:IMGSZ]]` (ripetibile). Ad ogni esecuzione le cartelle dei modelli indicati vengono rigenerate.
- `python3 scripts/contact_sheets.py` compone i fogli da tutte le cartelle dei modelli (`--models` per sceglierle e ordinarle, `--cols` per le colonne).
