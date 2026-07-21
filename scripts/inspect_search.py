"""Inspection manuelle de la récupération sémantique.

Outil de développement, pas un test automatisé : il n'a pas de verdict
pass/fail, il affiche des résultats qu'un humain juge. C'est le garde-fou avant
de brancher le LLM — si la récupération est mauvaise, aucune qualité de
génération ne la rattrapera.

Toute la logique vit dans ``src/retriever.py``. Ce script se contente de lire
les arguments, d'appeler le retriever et d'afficher. Aucune règle de
récupération ne doit être écrite ici : elle divergerait de celle utilisée par
la chaîne RAG et par l'évaluation.

Usage :
    python scripts/inspect_search.py
    python scripts/inspect_search.py "atelier pour enfants pendant les vacances"
    python scripts/inspect_search.py "un concert de jazz" --k 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Le script est lancé depuis la racine du projet ; on rend `src` importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retriever import (  # noqa: E402
    INDEX_DIR,
    K_DEFAUT,
    Recuperation,
    charger_index,
    rechercher_evenements,
)

REQUETES_DEFAUT = [
    "un concert de musique classique",
    "une exposition de photographie",
    "activité gratuite pour les enfants",
    "spectacle de danse contemporaine",
    "visite guidée du patrimoine lillois",
    "conférence sur l'écologie",
]


def afficher_manifeste() -> None:
    """Affiche les paramètres de construction de l'index interrogé.

    Sans cela, on interprète des résultats sans savoir avec quel découpage ni
    quel modèle l'index a été construit.
    """
    chemin = INDEX_DIR / "manifest.json"
    if not chemin.exists():
        print("(pas de manifest.json — index construit par une version antérieure)\n")
        return

    manifeste = json.loads(chemin.read_text(encoding="utf-8"))
    print("Index interrogé")
    for cle, valeur in manifeste.items():
        print(f"  {cle:<20} {valeur}")
    print()


def afficher_resultats(question: str, recuperation: Recuperation) -> None:
    """Affiche les événements récupérés pour une question, et les métriques."""
    print("=" * 78)
    print(f"❓ {question}")
    print("=" * 78)

    if not recuperation.documents:
        print("\n  Aucun résultat.\n")
        return

    for rang, (doc, score) in enumerate(
        zip(recuperation.documents, recuperation.scores), start=1
    ):
        meta = doc.metadata
        extrait = doc.page_content.replace("\n", " ")[:150]

        print(f"\n{rang}. [{score:.4f}] {meta.get('title')}")
        print(f"   uid     : {meta.get('uid')}")
        print(f"   dates   : {meta.get('daterange')}")
        print(f"   lieu    : {meta.get('location_name')} — {meta.get('location_city')}")
        print(f"   chunk   : {meta.get('chunk_index', 0) + 1}/{meta.get('n_chunks')}")
        print(f"   extrait : {extrait}…")

    print(
        f"\n   ⏱  {recuperation.latence_ms:.0f} ms "
        f"(dont {recuperation.latence_embedding_ms:.0f} ms d'embedding) — "
        f"{recuperation.chunks_examines} chunks examinés, "
        f"{len(recuperation)} événements distincts retenus"
    )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "questions",
        nargs="*",
        help="questions à poser ; à défaut, un jeu de requêtes de référence",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=K_DEFAUT,
        help=f"nombre d'événements distincts à afficher (défaut : {K_DEFAUT})",
    )
    args = parser.parse_args()

    store = charger_index()
    afficher_manifeste()

    for question in args.questions or REQUETES_DEFAUT:
        afficher_resultats(question, rechercher_evenements(store, question, k=args.k))

    print(
        "Lecture : le score est une similarité cosinus (1.0 = identique, 0 = orthogonal).\n"
        "Un top-1 sous ~0.6 sur une requête simple signale un problème de découpage\n"
        "ou de construction du champ `texte`, pas un problème de modèle."
    )


if __name__ == "__main__":
    main()
