#!/bin/bash

. scripts/.env

# si sta per cancellare il contenuto della cartella processed e i file di selezione, quindi si chiede conferma all'utente
echo "ATTENZIONE: stai per cancellare il contenuto della cartella 'processed' e i file di selezione. Vuoi continuare? (y/n)"
read answer
if [ "$answer" != "${answer#[Yy]}" ] ;then
    echo "Cancellazione in corso..."
else
    echo "Cancellazione annullata."
    exit 1
fi

# 
rm -rf data/processed/*
rm data/selected_images.txt
rm data/flagged_rider_contamination.txt
