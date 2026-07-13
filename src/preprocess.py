"""Nettoyage et structuration des événements Open Agenda.

Règles appliquées, issues de l'audit qualité (scripts/inspect_data.py,
scripts/diagnose.py) :

1. Suppression des événements dépourvus de tout contenu textuel exploitable.
   Un événement sans titre ni description ne peut pas être vectorisé.
2. Nettoyage du HTML présent dans `longdescription_fr` (balises <p>, <br>,
   entités HTML) et normalisation des espaces.
3. Construction d'un champ `texte` unique, concaténation du titre, du résumé
   et de la description longue, qui servira de support à la vectorisation.

Le dédoublonnage se fait sur `uid`. Les titres répétés (ex. « Mai à Vélo 2026 »,
114 occurrences) correspondent à des événements distincts d'une même campagne,
tenus en des lieux différents avec des descriptions différentes : ils sont conservés.

Entrée  : data/raw/events_lille.json
Sortie  : data/processed/events_lille_clean.json
"""

import json
import re
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

SOURCE = Path("data/raw/events_lille.json")
SORTIE = Path("data/processed/events_lille_clean.json")


def nettoyer_html(valeur: object) -> str:
    """Retire les balises HTML et normalise les espaces d'un texte.

    Args:
        valeur: contenu brut, potentiellement None ou porteur de balises HTML.

    Returns:
        Le texte débarrassé de ses balises, entités décodées, espaces normalisés.
        Chaîne vide si l'entrée est nulle.
    """
    if not isinstance(valeur, str) or not valeur.strip():
        return ""

    texte = BeautifulSoup(valeur, "html.parser").get_text(separator=" ")
    return re.sub(r"\s+", " ", texte).strip()


def construire_texte(ligne: pd.Series) -> str:
    """Assemble le contenu textuel d'un événement en vue de sa vectorisation.

    Concatène le titre, le résumé court et la description longue, en ignorant
    les champs vides.

    Args:
        ligne: une ligne du DataFrame d'événements nettoyés.

    Returns:
        Le texte complet de l'événement, champs séparés par un retour ligne.
    """
    morceaux = [
        ligne["title_fr"],
        ligne["description_fr"],
        ligne["longdescription_fr"],
    ]
    return "\n".join(m for m in morceaux if m)


def main() -> None:
    df = pd.DataFrame(json.loads(SOURCE.read_text(encoding="utf-8")))
    initial = len(df)
    print(f"Événements bruts : {initial}")

    # Règle 2 — nettoyage des champs textuels
    for colonne in ["title_fr", "description_fr", "longdescription_fr"]:
        df[colonne] = df[colonne].apply(nettoyer_html)

    # Règle 3 — texte support de la vectorisation
    df["texte"] = df.apply(construire_texte, axis=1)

    # Règle 1 — suppression des événements sans contenu exploitable
    df = df[df["texte"].str.len() > 0].copy()
    print(f"Supprimés (aucun texte exploitable) : {initial - len(df)}")

    # Dédoublonnage de sécurité
    avant = len(df)
    df = df.drop_duplicates(subset="uid").copy()
    if avant != len(df):
        print(f"Supprimés (uid dupliqués) : {avant - len(df)}")

    SORTIE.parent.mkdir(parents=True, exist_ok=True)
    SORTIE.write_text(
        df.to_json(orient="records", force_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\nÉvénements retenus : {len(df)}")
    print(f"Longueur médiane du texte : {df['texte'].str.len().median():.0f} caractères")
    print(f"Longueur maximale         : {df['texte'].str.len().max():.0f} caractères")
    print(f"\nÉcrit dans {SORTIE}")

    print("\n--- Exemple de texte nettoyé ---")
    print(df["texte"].iloc[0][:400])


if __name__ == "__main__":
    main()