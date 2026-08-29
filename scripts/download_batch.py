#!/usr/bin/env python3
"""Esegue download + dedupe per ogni dataset con enabled=1 e download=1 in
datasets_to_download.csv.

Per ogni riga: scarica il dataset, verifica che i nomi in escooter_class_name
(separati da '|') esistano nel data.yaml, ed esegue la deduplica. Il CSV non
viene mai modificato da questo script: la scelta della versione da fissare
nella colonna `version` resta a discrezione manuale.
"""
import csv

import config
import dedupe_augmented
import download_dataset

CSV_PATH = config.CSV_PATH


def load_rows() -> list[dict]:
    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for i, row in enumerate(rows, start=2):  # riga 1 = header
        if None in row or any(v is None for v in row.values()):
            raise ValueError(
                f"Riga {i} di {CSV_PATH} malformata (numero di campi diverso dall'header): {row}"
            )
    return rows


def main():
    rows = load_rows()
    todo_rows = [r for r in rows if r["enabled"] == "1" and r["download"] == "1"]

    if not todo_rows:
        print("Nessun dataset con enabled=1 e download=1.")
        return

    print(f"{len(todo_rows)} dataset da processare.")
    for row in todo_rows:
        workspace, project = row["workspace_id"], row["project_id"]
        version = int(row["version"]) if row["version"] else None
        class_names = [n for n in row["escooter_class_name"].split("|") if n]
        print(f"\n=== {workspace}/{project} ===")

        try:
            out_dir, version_number, missing = download_dataset.download_dataset(
                workspace, project, version=version, escooter_class_names=class_names
            )
        except Exception as e:
            print(f"ERRORE download {workspace}/{project}: {e}")
            continue

        if missing:
            print(f"Salto la deduplica: classi mancanti nel data.yaml per {workspace}/{project}: {missing}. "
                  f"Correggi il CSV e rilancia questa riga.")
            continue

        try:
            dedupe_augmented.dedupe_dataset(out_dir)
        except Exception as e:
            print(f"ERRORE dedupe {workspace}/{project}: {e}")
            continue

        print(f"{workspace}/{project} completato.")


if __name__ == "__main__":
    main()
