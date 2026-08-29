#!/usr/bin/env python3
"""Confronta datasets_to_download.csv con data/datasets.json (e con i
contenuti effettivi di data/raw/) per verificare che, su questa macchina,
tutti i dataset previsti dal CSV (enabled=1) siano stati effettivamente
scaricati.

Utile lavorando su più macchine: data/ non è versionata, quindi ogni
istanza del progetto può avere scaricato solo un sottoinsieme dei dataset
elencati nel CSV (che invece è specifico dell'istanza ma può essere stato
copiato/aggiornato manualmente da un'altra).

Uscita 0 se non ci sono dataset mancanti o disallineati, 1 altrimenti.
"""
import csv
import sys

import config
import dataset_index

CSV_PATH = config.CSV_PATH
RAW_DIR = config.RAW_DIR


def load_csv_rows() -> list[dict]:
    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> int:
    rows = load_csv_rows()
    index = dataset_index.load_index()
    index_by_id = {e["id"]: e for e in index}

    enabled = [r for r in rows if r["enabled"] == "1"]
    pinned = [r for r in enabled if r["version"]]
    unpinned = [r for r in enabled if not r["version"]]

    ok = True
    referenced_ids: set[str] = set()

    print(f"CSV: {CSV_PATH}")
    print(f"Indice: {dataset_index.INDEX_PATH}\n")

    missing = []
    not_on_disk = []
    workspace_mismatch = []
    for row in pinned:
        expected_id = f"{row['project_id']}-v{row['version']}"
        entry = index_by_id.get(expected_id)
        if entry is None:
            missing.append(row)
            continue
        referenced_ids.add(expected_id)
        if entry["workspace"] != row["workspace_id"]:
            workspace_mismatch.append((row, entry))
        if not (RAW_DIR / expected_id).is_dir():
            not_on_disk.append((row, expected_id))

    if missing:
        ok = False
        print(f"MANCANTI ({len(missing)}) — enabled=1, version fissata nel CSV, assenti dall'indice:")
        for row in missing:
            print(f"  - {row['workspace_id']}/{row['project_id']} v{row['version']}")
        print()

    if not_on_disk:
        ok = False
        print(f"IN INDICE MA NON SU DISCO ({len(not_on_disk)}) — presenti in datasets.json ma "
              f"la cartella in data/raw/ non c'è (rimossa a mano?):")
        for row, expected_id in not_on_disk:
            print(f"  - {expected_id}  (data/raw/{expected_id}/)")
        print()

    if workspace_mismatch:
        ok = False
        print(f"WORKSPACE DISCORDANTE ({len(workspace_mismatch)}) — stesso id ma workspace diverso da quello nel CSV:")
        for row, entry in workspace_mismatch:
            print(f"  - id {entry['id']}: CSV={row['workspace_id']}, indice={entry['workspace']}")
        print()

    unpinned_missing = []
    unpinned_present = []
    for row in unpinned:
        matches = [e for e in index if e["project"] == row["project_id"] and e["workspace"] == row["workspace_id"]]
        if matches:
            referenced_ids.update(e["id"] for e in matches)
            unpinned_present.append((row, matches))
        else:
            unpinned_missing.append(row)

    if unpinned_missing:
        print(f"VERSIONE NON FISSATA E ASSENTE ({len(unpinned_missing)}) — nessun download trovato, "
              f"per nessuna versione:")
        for row in unpinned_missing:
            print(f"  - {row['workspace_id']}/{row['project_id']}")
        print()

    if unpinned_present:
        print(f"VERSIONE NON FISSATA MA PRESENTE ({len(unpinned_present)}) — valuta se fissare la "
              f"versione nella colonna `version` del CSV:")
        for row, matches in unpinned_present:
            versions = ", ".join(f"v{e['version']}" for e in matches)
            print(f"  - {row['workspace_id']}/{row['project_id']}: trovate {versions}")
        print()

    orphans = [e for e in index if e["id"] not in referenced_ids]
    if orphans:
        print(f"NON PIU' PREVISTI DAL CSV ({len(orphans)}) — presenti nell'indice ma non corrispondono "
              f"a nessuna riga enabled=1 del CSV:")
        for e in orphans:
            print(f"  - {e['id']} ({e['workspace']}/{e['project']})")
        print()

    if ok and not unpinned_missing and not orphans:
        print("Tutto allineato: i dataset previsti dal CSV sono tutti presenti nell'indice e su disco.")

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
