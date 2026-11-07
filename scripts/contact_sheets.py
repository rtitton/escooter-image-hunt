#!/usr/bin/env python3
"""Fogli affiancati per confrontare a occhio i modelli sugli stessi frame.

COSA FA
  infer_videos.py ha già prodotto, per ogni modello, i frame dei video con le box disegnate:
      inferences/<modello>/<video>/<video>_f000010.jpg
  La cartella di ogni modello contiene gli STESSI nomi di file (stesso frame, stesso video).
  Questo script prende, per ogni frame, le immagini di tutti i modelli e le mette in un'unica
  immagine "foglio", così si vedono insieme e si confrontano a colpo d'occhio (esempio con 6 modelli
  e 3 colonne):

      +-------------------+-------------------+-------------------+
      | nome modello 1    | nome modello 2    | nome modello 3    |   <- barra col nome
      |   frame modello 1 |   frame modello 2 |   frame modello 3 |
      +-------------------+-------------------+-------------------+
      | nome modello 4    | nome modello 5    | nome modello 6    |
      |   frame modello 4 |   frame modello 5 |   frame modello 6 |
      +-------------------+-------------------+-------------------+

  I fogli vanno in inferences/_confronto/<video>/, con lo stesso nome del frame.

QUALI MODELLI
  Di default tutte le sottocartelle di --src che non iniziano per "_" (quindi non _confronto), in ordine
  alfabetico. Con --models si scelgono e si ordinano a mano: l'ORDINE conta, è la posizione nella griglia,
  da sinistra a destra e riga per riga.

COME SI AVVIA (dalla radice del repo, dopo infer_videos.py)
    python3 scripts/contact_sheets.py [--src inferences] [--out-dir inferences/_confronto]
        [--models NOME ...] [--cols 3] [--tile-width 640]
  Ad ogni esecuzione la cartella di output viene cancellata e rigenerata da zero.

DUE FATTI SULLE IMMAGINI CHE SERVONO PER LEGGERE IL CODICE
  - OpenCV (cv2) e numpy rappresentano un'immagine a colori come una tabella di numeri
    ("array") con tre dimensioni: (altezza, larghezza, 3), dove il 3 sono i canali colore,
    in ordine Blu-Verde-Rosso (BGR, non RGB). Ogni numero va da 0 (nero) a 255 (massimo).
    Quindi img.shape[0] è l'altezza in pixel e img.shape[1] è la larghezza.
  - Per attaccare due immagini una sopra l'altra (np.vstack) devono avere la stessa
    larghezza; per attaccarle una accanto all'altra (np.hstack) la stessa altezza.
"""
import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np

BAR_H = 30  # altezza in pixel della barra col nome del modello, sopra ogni riquadro


def tile(path: Path, name: str, tile_w: int):
    """Costruisce il riquadro di UN modello per UN frame: la barra col nome sopra, il frame sotto.

    Esempio con un frame 640x360 e tile_w 640: il frame resta 640x360, la barra è 640x30, il riquadro
    risultante è 640x390. Con un frame 1920x1080 il frame viene prima rimpicciolito a 640x360,
    quindi il riquadro è comunque 640x390."""
    img = cv2.imread(str(path))  # legge il file JPG e lo restituisce come array (altezza, larghezza, 3)
    if img is None:
        raise SystemExit(f"Immagine non leggibile: {path}")

    # Nuova altezza che mantiene le proporzioni dopo aver portato la larghezza a tile_w:
    # altezza_nuova = altezza_vecchia * larghezza_nuova / larghezza_vecchia
    h = round(img.shape[0] * tile_w / img.shape[1])
    # cv2.resize vuole (larghezza, altezza) in quest'ordine, al contrario di shape.
    # INTER_AREA è il metodo di ricampionamento più indicato quando si RIMPICCIOLISCE:
    # fa la media dei pixel vicini invece di saltarne alcuni, e l'immagine resta più nitida.
    img = cv2.resize(img, (tile_w, h), interpolation=cv2.INTER_AREA)

    # Barra del nome: np.full crea un array BAR_H x tile_w x 3 con lo stesso valore (30) ovunque,
    # cioè un rettangolo grigio molto scuro. np.uint8 = interi da 0 a 255, il formato delle immagini.
    bar = np.full((BAR_H, tile_w, 3), 30, np.uint8)
    # Scrive il nome sulla barra: il punto (8, 21) è dove parte il testo (8 pixel da sinistra,
    # 21 dall'alto, cioè circa a metà della barra); 0.65 è la dimensione del carattere,
    # (255, 255, 255) è il colore bianco, 2 lo spessore del tratto.
    cv2.putText(bar, name, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    # vstack impila la barra sopra il frame: hanno entrambi larghezza tile_w, quindi si può fare.
    return np.vstack([bar, img])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, default=Path("inferences"),
                         help="Cartella con una sottocartella per modello (output di infer_videos.py; default: inferences)")
    parser.add_argument("--out-dir", type=Path, default=None,
                         help="Dove scrivere i fogli (default: <src>/_confronto)")
    parser.add_argument("--models", nargs="+", default=None,
                         help="Modelli da affiancare, nell'ordine della griglia (default: tutte le sottocartelle di --src)")
    parser.add_argument("--cols", type=int, default=3, help="Riquadri per riga (default: 3)")
    parser.add_argument("--tile-width", type=int, default=640,
                         help="Larghezza di ogni riquadro in pixel (default: 640): i frame originali hanno "
                              "dimensioni diverse (per esempio 1920x1080 e 640x360), si portano tutti a questa larghezza")
    args = parser.parse_args()

    src = args.src
    out = args.out_dir or src / "_confronto"
    if not src.is_dir():
        raise SystemExit(f"Cartella non trovata: {src}")
    # Modelli: quelli indicati o, di default, le sottocartelle che non iniziano per "_" (_confronto è l'output)
    models = args.models or sorted(p.name for p in src.iterdir() if p.is_dir() and not p.name.startswith("_"))
    if len(models) < 2:
        raise SystemExit(f"Servono almeno due modelli in {src} (trovati: {models})")
    for m in models:
        if not (src / m).is_dir():
            raise SystemExit(f"Modello non trovato: {src / m}")

    # Se ci sono fogli di una esecuzione precedente, si cancellano: così non restano file vecchi
    # di frame che nel frattempo non esistono più.
    if out.exists():
        shutil.rmtree(out)

    # Elenco dei video: le sottocartelle del PRIMO modello. I frame di ogni video sono quelli presenti
    # in TUTTI i modelli (intersezione): se un modello ne ha di meno, i frame mancanti si saltano e si avvisa.
    videos = sorted(p.name for p in (src / models[0]).iterdir() if p.is_dir())
    n = 0  # contatore dei fogli scritti, per il messaggio finale
    for video in videos:
        per_model = [{f.name for f in (src / m / video).glob("*.jpg")} for m in models]
        frames = sorted(set.intersection(*per_model))
        if len(frames) != max(len(s) for s in per_model):
            print(f"ATTENZIONE: {video}: i modelli non hanno tutti gli stessi frame, ne uso {len(frames)}")
        (out / video).mkdir(parents=True)  # una sottocartella di _confronto per ogni video
        for name in frames:
            # Lo stesso frame preso dalla cartella di ciascun modello, già con la sua barra:
            # una lista di riquadri, nell'ordine di models.
            tiles = [tile(src / m / video / name, m, args.tile_width) for m in models]
            # Se i riquadri non riempiono l'ultima riga (per esempio 5 modelli su 3 colonne) si completa con
            # riquadri neri, altrimenti np.hstack non troverebbe righe della stessa larghezza.
            while len(tiles) % args.cols:
                tiles.append(np.zeros_like(tiles[0]))
            # Righe della griglia: tiles[0:cols] è la prima riga, tiles[cols:2*cols] la seconda e così via.
            # hstack affianca i riquadri di una riga: funziona perché hanno la stessa altezza (stessa
            # larghezza e stesse proporzioni, perché sono lo stesso frame).
            rows = [np.hstack(tiles[i:i + args.cols]) for i in range(0, len(tiles), args.cols)]
            # vstack impila le righe una sopra l'altra: stessa larghezza (cols x tile_w), quindi ok.
            # Qualità JPEG 88 su 100: file piccoli, immagine buona.
            cv2.imwrite(str(out / video / name), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 88])
            n += 1
    print(f"{n} fogli ({len(models)} modelli: {', '.join(models)}) in {out}")


if __name__ == "__main__":
    main()
