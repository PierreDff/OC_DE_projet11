"""Récupération sémantique d'événements dans l'index FAISS.

Ce module est la brique de récupération partagée du POC. Il est consommé par :

- ``src/rag_chain.py``      : contexte envoyé au LLM ;
- ``scripts/evaluate_rag.py`` : mesure du rappel et de la précision ;
- ``scripts/test_search.py``  : inspection manuelle de la récupération.

Deux responsabilités, et deux seulement : charger l'index, et renvoyer les *k*
événements les plus proches d'une question — événements, pas chunks (voir
``rechercher_evenements``).

La génération de réponse ne fait pas partie de ce module. Isoler la récupération
permet de la mesurer seule : si elle est mauvaise, aucune qualité de génération
ne la rattrapera.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_mistralai import MistralAIEmbeddings

load_dotenv()

# --- Constantes ------------------------------------------------------------

RACINE = Path(__file__).resolve().parent.parent
INDEX_DIR = RACINE / "faiss_index"

MODELE_EMBEDDING = "mistral-embed"

K_DEFAUT = 5
"""Nombre d'événements distincts renvoyés à l'appelant."""

FETCH_K_DEFAUT = 20
"""Nombre de chunks demandés à FAISS avant déduplication.

Sur-échantillonnage nécessaire : les chunks d'un même événement se ressemblent
fortement et ressortent groupés. Sans marge, un top-5 de chunks peut ne
contenir que 3 événements distincts.
"""


# --- Structure de résultat -------------------------------------------------


@dataclass
class Recuperation:
    """Résultat d'une recherche, avec ses métriques d'exécution.

    Les champs d'instrumentation ne sont pas décoratifs : ils alimentent le
    rapport technique (latence de récupération) et l'estimation des coûts
    d'exploitation (un appel d'embedding par question posée).

    Attributs
    ---------
    documents:
        Les *k* meilleurs chunks, un par événement, triés par score décroissant.
    scores:
        Similarités cosinus associées, dans le même ordre que ``documents``.
    latence_ms:
        Durée totale de la récupération : embedding de la question + recherche
        FAISS + déduplication.
    latence_embedding_ms:
        Part de ``latence_ms`` imputable à l'appel réseau vers l'API Mistral.
        Isolée car c'est le seul poste coûteux : la recherche FAISS locale sur
        4 527 vecteurs est de l'ordre de la milliseconde.
    chunks_examines:
        Nombre de chunks effectivement remontés par FAISS avant déduplication.
    """

    documents: list[Document] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    latence_ms: float = 0.0
    latence_embedding_ms: float = 0.0
    chunks_examines: int = 0

    def __len__(self) -> int:
        return len(self.documents)

    @property
    def uids(self) -> list[str]:
        """Identifiants des événements récupérés — appariement avec le jeu de test."""
        return [doc.metadata["uid"] for doc in self.documents]


# --- Chargement de l'index -------------------------------------------------


def charger_index(index_dir: Path = INDEX_DIR) -> FAISS:
    """Charge l'index FAISS et le modèle d'embedding associé.

    L'index n'est pas versionné : il est régénéré par ``src/vectorize.py``.
    Un chargement qui échoue signifie donc que le pipeline n'a pas été exécuté,
    et le message d'erreur le dit explicitement plutôt que de laisser remonter
    un ``FileNotFoundError`` opaque.

    Paramètres
    ----------
    index_dir:
        Dossier contenant ``index.faiss``, ``index.pkl`` et ``manifest.json``.

    Retourne
    --------
    FAISS
        Le vector store, prêt à être interrogé.

    Lève
    ----
    FileNotFoundError
        Si l'index n'existe pas.
    RuntimeError
        Si la clé API Mistral est absente de l'environnement.
    """
    if not os.getenv("MISTRAL_API_KEY"):
        raise RuntimeError(
            "MISTRAL_API_KEY absente. Copier .env.example vers .env et y "
            "renseigner la clé (console.mistral.ai)."
        )

    if not index_dir.exists():
        raise FileNotFoundError(
            f"Index introuvable dans {index_dir}. "
            "Le construire d'abord : python src\\vectorize.py"
        )

    embeddings = MistralAIEmbeddings(model=MODELE_EMBEDDING)

    # allow_dangerous_deserialization : l'index est un pickle. Le flag protège
    # contre le chargement d'un index d'origine inconnue. Ici l'index est
    # produit localement par notre propre script — le risque est nul.
    return FAISS.load_local(
        str(index_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )


# --- Récupération ----------------------------------------------------------


def rechercher_evenements(
    vectorstore: FAISS,
    question: str,
    k: int = K_DEFAUT,
    fetch_k: int = FETCH_K_DEFAUT,
) -> Recuperation:
    """Renvoie les *k* événements distincts les plus proches d'une question.

    FAISS ne connaît pas la notion d'événement : il renvoie des chunks. Or les
    chunks d'un même événement long se ressemblent et ressortent groupés. Sans
    traitement, un top-*k* de chunks peut ne contenir que 3 ou 4 événements
    distincts — les places restantes étant perdues pour des doublons, au prix
    d'un contexte redondant envoyé au LLM.

    La fonction sur-échantillonne (``fetch_k`` chunks), ne conserve que le
    meilleur chunk de chaque ``uid``, puis tronque à *k*. FAISS renvoyant ses
    résultats déjà triés par score décroissant, **le premier chunk rencontré
    pour un ``uid`` est nécessairement son meilleur** : une simple lecture
    séquentielle suffit, aucun tri supplémentaire n'est nécessaire.

    Paramètres
    ----------
    vectorstore:
        Index chargé par ``charger_index``.
    question:
        Requête en langage naturel.
    k:
        Nombre d'événements distincts souhaités.
    fetch_k:
        Nombre de chunks demandés à FAISS avant déduplication. Doit rester
        nettement supérieur à *k*, faute de quoi la déduplication peut renvoyer
        moins de *k* événements.

    Retourne
    --------
    Recuperation
        Documents, scores et métriques d'exécution.
    """
    if not question or not question.strip():
        raise ValueError("La question ne peut pas être vide.")

    debut = time.perf_counter()

    # L'embedding de la question est isolé pour être chronométré séparément :
    # c'est le seul appel réseau de la récupération.
    debut_embedding = time.perf_counter()
    vecteur = vectorstore.embedding_function.embed_query(question)
    latence_embedding = (time.perf_counter() - debut_embedding) * 1000

    chunks = vectorstore.similarity_search_with_score_by_vector(vecteur, k=fetch_k)

    documents: list[Document] = []
    scores: list[float] = []
    uids_vus: set[str] = set()

    for doc, score in chunks:
        uid = doc.metadata.get("uid")
        if uid in uids_vus:
            continue
        uids_vus.add(uid)
        documents.append(doc)
        scores.append(float(score))
        if len(documents) == k:
            break

    latence = (time.perf_counter() - debut) * 1000

    return Recuperation(
        documents=documents,
        scores=scores,
        latence_ms=round(latence, 1),
        latence_embedding_ms=round(latence_embedding, 1),
        chunks_examines=len(chunks),
    )
