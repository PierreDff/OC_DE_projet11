"""Récupération des événements Open Agenda depuis l'API OpenDataSoft.

Périmètre du POC :
    - géographique : commune de Lille
    - temporel     : événements dont la date de fin est postérieure
                     à aujourd'hui moins un an (lastdate_end >= now(years=-1))

Le filtrage est effectué côté serveur via le langage de requête ODSQL,
ce qui évite de télécharger 1,2 million d'enregistrements pour en garder 3 600.

Un manifeste de collecte (`collecte_manifest.json`) est écrit à côté des données
brutes. Il fige la date de collecte et le seuil temporel évalué ce jour-là.

Pourquoi c'est nécessaire : le seuil `now(years=-1)` est glissant. Un événement
retenu aujourd'hui aura « plus d'un an » demain, alors qu'il reste dans le jeu
figé. Le manifeste permet aux tests unitaires de vérifier le critère contre la
*date de collecte* — un seuil fixe — plutôt que contre `datetime.now()`, qui
ferait échouer le test avec le simple passage du temps, sans changement de code.
"""

import json
from datetime import date, datetime, timedelta
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
MANIFESTE = Path("data/raw/collecte_manifest.json")


def fetch_events() -> list[dict]:
    """Interroge l'API et retourne la liste brute des événements du périmètre."""
    reponse = requests.get(
        EXPORT_URL,
        params={"where": WHERE, "select": ",".join(CHAMPS)},
        timeout=120,
    )
    reponse.raise_for_status()
    return reponse.json()


def ecrire_manifeste(n_evenements: int) -> None:
    """Écrit le manifeste de collecte à côté des données brutes.

    Le seuil temporel est recalculé ici *en Python* pour être figé dans le
    fichier. `now(years=-1)` est évalué côté serveur par l'API : on ne récupère
    pas sa valeur dans la réponse. On la reconstruit donc à la date du jour, ce
    qui correspond exactement au seuil que l'API vient d'appliquer.
    """
    aujourd_hui = date.today()
    seuil = aujourd_hui - timedelta(days=365)

    manifeste = {
        "date_collecte": datetime.now().isoformat(timespec="seconds"),
        "source": "OpenDataSoft Explore v2.1 — evenements-publics-openagenda",
        "where": WHERE,
        "champs": CHAMPS,
        "seuil_temporel": seuil.isoformat(),
        "seuil_note": "lastdate_end >= cette date (now(years=-1) au jour de la collecte)",
        "n_evenements": n_evenements,
    }
    MANIFESTE.parent.mkdir(parents=True, exist_ok=True)
    MANIFESTE.write_text(
        json.dumps(manifeste, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    evenements = fetch_events()

    SORTIE.parent.mkdir(parents=True, exist_ok=True)
    SORTIE.write_text(
        json.dumps(evenements, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    ecrire_manifeste(len(evenements))

    print(f"{len(evenements)} événements récupérés")
    print(f"Écrits dans {SORTIE}")
    print(f"Manifeste de collecte écrit dans {MANIFESTE}")


if __name__ == "__main__":
    main()
