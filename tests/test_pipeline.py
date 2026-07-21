"""Tests unitaires du pipeline Puls-Events RAG.

Vérifient les invariants du jeu de données et de l'index, sans appel réseau.

Quatre familles :

1. **Périmètre géographique** — chaque événement est localisé à Lille.
2. **Périmètre temporel** — chaque événement a une date de fin postérieure au
   seuil de collecte (fixe, lu dans le manifeste — pas ``datetime.now()``).
3. **Intégrité du jeu de test** — chaque uid référencé dans ``qa_dataset.json``
   existe dans le jeu de données figé.
4. **Cohérence de l'index** — le nombre de vecteurs correspond au manifest de
   l'index.

Usage :
    pytest tests/test_pipeline.py -v
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

# --------------------------------------------------------------------------- #
# Chemins
# --------------------------------------------------------------------------- #

ROOT = Path(__file__).resolve().parent.parent
EVENTS = ROOT / "data" / "processed" / "events_lille_clean.json"
MANIFESTE_COLLECTE = ROOT / "data" / "raw" / "collecte_manifest.json"
MANIFESTE_INDEX = ROOT / "faiss_index" / "manifest.json"
QA_DATASET = ROOT / "data" / "eval" / "qa_dataset.json"


# --------------------------------------------------------------------------- #
# Fixtures — chargent les données une seule fois par session de test
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="session")
def evenements() -> list[dict]:
    """Charge le jeu de données nettoyé."""
    assert EVENTS.exists(), f"Jeu de données introuvable : {EVENTS}"
    return json.loads(EVENTS.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def manifeste_collecte() -> dict:
    """Charge le manifeste de collecte (date et seuil temporel)."""
    assert MANIFESTE_COLLECTE.exists(), (
        f"Manifeste de collecte introuvable : {MANIFESTE_COLLECTE}. "
        "Le créer avec : python src/fetch_data.py"
    )
    return json.loads(MANIFESTE_COLLECTE.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def manifeste_index() -> dict:
    """Charge le manifeste de l'index FAISS."""
    assert MANIFESTE_INDEX.exists(), (
        f"Manifeste de l'index introuvable : {MANIFESTE_INDEX}. "
        "Construire l'index avec : python src/vectorize.py"
    )
    return json.loads(MANIFESTE_INDEX.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def qa_questions() -> list[dict]:
    """Charge les questions du jeu de test annoté."""
    assert QA_DATASET.exists(), f"Jeu de test introuvable : {QA_DATASET}"
    data = json.loads(QA_DATASET.read_text(encoding="utf-8"))
    return data["questions"]


@pytest.fixture(scope="session")
def uids_existants(evenements) -> set[str]:
    """Ensemble des uids présents dans le jeu de données."""
    return {str(e["uid"]) for e in evenements}


# --------------------------------------------------------------------------- #
# 1. Périmètre géographique
# --------------------------------------------------------------------------- #


class TestPerimetreGeographique:
    """Vérifie que tous les événements sont localisés à Lille."""

    def test_tous_les_evenements_sont_a_lille(self, evenements):
        """Chaque événement doit avoir location_city == 'Lille'."""
        hors_perimetre = [
            (e["uid"], e.get("location_city"))
            for e in evenements
            if e.get("location_city") != "Lille"
        ]
        assert hors_perimetre == [], (
            f"{len(hors_perimetre)} événement(s) hors périmètre : "
            f"{hors_perimetre[:5]}"
        )

    def test_champ_location_city_present(self, evenements):
        """Le champ location_city ne doit jamais être absent ou vide."""
        sans_ville = [
            e["uid"] for e in evenements if not e.get("location_city")
        ]
        assert sans_ville == [], (
            f"{len(sans_ville)} événement(s) sans location_city"
        )


# --------------------------------------------------------------------------- #
# 2. Périmètre temporel
# --------------------------------------------------------------------------- #


class TestPerimetreTemporel:
    """Vérifie que tous les événements respectent le seuil temporel.

    Le seuil est lu dans le manifeste de collecte (date fixe), pas calculé
    depuis datetime.now(). Ainsi le test donne le même verdict aujourd'hui,
    à la soutenance, et dans un an.
    """

    def test_seuil_temporel_present_dans_manifeste(self, manifeste_collecte):
        """Le manifeste doit contenir un seuil_temporel exploitable."""
        assert "seuil_temporel" in manifeste_collecte, (
            "Champ seuil_temporel absent du manifeste de collecte"
        )
        # Vérifie que c'est une date parsable
        seuil = date.fromisoformat(manifeste_collecte["seuil_temporel"])
        assert isinstance(seuil, date)

    def test_tous_les_evenements_posterieurs_au_seuil(
        self, evenements, manifeste_collecte
    ):
        """Chaque événement doit avoir lastdate_end >= seuil_temporel."""
        seuil = manifeste_collecte["seuil_temporel"]  # "2025-07-13"

        anterieurs = []
        for e in evenements:
            lastdate = (e.get("lastdate_end") or "")[:10]  # "2026-01-15T..."→"2026-01-15"
            if not lastdate:
                anterieurs.append((e["uid"], "lastdate_end manquant"))
            elif lastdate < seuil:
                anterieurs.append((e["uid"], lastdate))

        assert anterieurs == [], (
            f"{len(anterieurs)} événement(s) antérieur(s) au seuil {seuil} : "
            f"{anterieurs[:5]}"
        )

    def test_champ_lastdate_end_present(self, evenements):
        """Le champ lastdate_end ne doit jamais être absent ou vide."""
        sans_date = [
            e["uid"] for e in evenements if not e.get("lastdate_end")
        ]
        assert sans_date == [], (
            f"{len(sans_date)} événement(s) sans lastdate_end"
        )


# --------------------------------------------------------------------------- #
# 3. Intégrité du jeu de test
# --------------------------------------------------------------------------- #


class TestIntegriteJeuDeTest:
    """Vérifie que le jeu de test annoté est cohérent avec les données."""

    def test_tous_les_uids_attendus_existent(self, qa_questions, uids_existants):
        """Chaque uid listé dans uids_attendus doit exister dans le jeu figé.

        Un uid absent signifie soit une faute de frappe, soit un événement
        supprimé depuis l'annotation — dans les deux cas, la mesure de recall
        serait faussée.
        """
        manquants = []
        for q in qa_questions:
            for uid in q["uids_attendus"]:
                if str(uid) not in uids_existants:
                    manquants.append((q["id"], uid))

        assert manquants == [], (
            f"{len(manquants)} uid(s) introuvable(s) dans le jeu de données : "
            f"{manquants}"
        )

    def test_questions_ont_un_type_valide(self, qa_questions):
        """Chaque question doit porter un type reconnu."""
        types_valides = {"factuelle", "thematique", "temporelle", "piege"}
        invalides = [
            (q["id"], q.get("type"))
            for q in qa_questions
            if q.get("type") not in types_valides
        ]
        assert invalides == [], f"Type(s) invalide(s) : {invalides}"

    def test_questions_ont_un_ground_truth(self, qa_questions):
        """Chaque question doit avoir un ground_truth non vide."""
        sans_gt = [
            q["id"] for q in qa_questions if not q.get("ground_truth", "").strip()
        ]
        assert sans_gt == [], f"Question(s) sans ground_truth : {sans_gt}"

    def test_pieges_ont_uids_attendus_vides(self, qa_questions):
        """Les questions-pièges ne doivent pas avoir d'uids_attendus.

        Un piège dont uids_attendus n'est pas vide fausserait le calcul de
        recall (on mesurerait la récupération au lieu de l'abstention).
        """
        pieges_avec_uids = [
            q["id"]
            for q in qa_questions
            if q["type"] == "piege" and q.get("uids_attendus")
        ]
        assert pieges_avec_uids == [], (
            f"Piège(s) avec uids_attendus non vides : {pieges_avec_uids}"
        )

    def test_pas_de_doublons_d_id(self, qa_questions):
        """Chaque question doit avoir un id unique."""
        ids = [q["id"] for q in qa_questions]
        doublons = [qid for qid in ids if ids.count(qid) > 1]
        assert doublons == [], f"Id(s) en double : {set(doublons)}"


# --------------------------------------------------------------------------- #
# 4. Cohérence de l'index
# --------------------------------------------------------------------------- #


class TestCoherenceIndex:
    """Vérifie que l'index FAISS est cohérent avec les données sources."""

    def test_nombre_evenements_coherent(self, evenements, manifeste_index):
        """Le nombre d'événements dans l'index doit correspondre au jeu nettoyé."""
        assert manifeste_index["n_events"] == len(evenements), (
            f"Index : {manifeste_index['n_events']} événements, "
            f"jeu nettoyé : {len(evenements)}"
        )

    def test_nombre_chunks_positif(self, manifeste_index):
        """L'index doit contenir au moins autant de chunks que d'événements."""
        assert manifeste_index["n_chunks"] >= manifeste_index["n_events"], (
            f"Moins de chunks ({manifeste_index['n_chunks']}) que "
            f"d'événements ({manifeste_index['n_events']})"
        )

    def test_modele_embedding_coherent(self, manifeste_index):
        """Le modèle d'embedding doit être celui attendu."""
        assert manifeste_index["embed_model"] == "mistral-embed", (
            f"Modèle inattendu : {manifeste_index['embed_model']}"
        )
