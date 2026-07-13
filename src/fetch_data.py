"""Récupération des événements Open Agenda depuis l'API OpenDataSoft.

Périmètre du POC :
    - géographique : commune de Lille
    - temporel     : événements dont la date de fin est postérieure
                     à aujourd'hui moins un an (lastdate_end >= now(years=-1))

Le filtrage est effectué côté serveur via le langage de requête ODSQL,
ce qui évite de télécharger 1,2 million d'enregistrements pour en garder 3 600.
"""

import json
from pathlib import Path

import requests

EXPORT_URL = (
    "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/"
    "evenements-publics-openagenda/exports/json"
)

VILLE = "Lille"
WHERE = f'location_city = "{VILLE}" and lastdate_end >= now(years=-1)'

CHAMPS = [
    "uid",
    "title_fr",
    "description_fr",
    "longdescription_fr",
    "keywords_fr",
    "daterange_fr",
    "firstdate_begin",
    "lastdate_end",
    "location_name",
    "location_address",
    "location_city",
    "location_department",
    "location_region",
    "location_coordinates",
    "canonicalurl",
]

SORTIE = Path("data/raw/events_lille.json")


def fetch_events() -> list[dict]:
    """Interroge l'API et retourne la liste brute des événements du périmètre."""
    reponse = requests.get(
        EXPORT_URL,
        params={"where": WHERE, "select": ",".join(CHAMPS)},
        timeout=120,
    )
    reponse.raise_for_status()
    return reponse.json()


def main() -> None:
    evenements = fetch_events()

    SORTIE.parent.mkdir(parents=True, exist_ok=True)
    SORTIE.write_text(
        json.dumps(evenements, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"{len(evenements)} événements récupérés")
    print(f"Écrits dans {SORTIE}")


if __name__ == "__main__":
    main()