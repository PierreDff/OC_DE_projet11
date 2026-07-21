"""Étape 3 — Découpage en chunks, vectorisation Mistral et index FAISS.

Lit `data/processed/events_lille_clean.json`, découpe le champ `texte` en chunks,
attache les métadonnées de chaque événement, vectorise via `mistral-embed` et
construit un index FAISS persisté dans `faiss_index/`.

Points de conception :

- **En-tête de contexte** : chaque chunk est préfixé d'une ligne `titre — dates — lieu`.
  Sans cela, les chunks 2..n d'un événement long perdent toute référence à l'événement
  auquel ils appartiennent et deviennent inexploitables en récupération.
- **Similarité cosinus** : les vecteurs sont normalisés (L2) et l'index utilise le
  produit scalaire. Les embeddings Mistral ne sont pas normalisés en sortie ; une
  distance L2 brute pénaliserait les textes longs indépendamment de leur pertinence.
- **Reprise sur erreur** : chaque batch vectorisé est écrit dans `data/processed/emb_cache/`.
  Une interruption (rate limit, coupure réseau) ne fait pas repartir de zéro.
- **Index plat** : à cette volumétrie (quelques milliers de vecteurs), un `IndexFlatIP`
  exhaustif est à la fois exact et instantané. Un index approximatif (IVF, HNSW)
  n'apporterait rien et introduirait une perte de rappel.

Usage :
    python src/vectorize.py
    python src/vectorize.py --force    # ignore le cache et revectorise tout
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_mistralai import MistralAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "data" / "processed" / "events_lille_clean.json"
INDEX_DIR = ROOT / "faiss_index"
CACHE_DIR = ROOT / "data" / "processed" / "emb_cache"

EMBED_MODEL = "mistral-embed"
EMBED_DIM = 1024

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200

BATCH_SIZE = 32          # nombre de chunks envoyés par appel API
MAX_RETRIES = 6          # tentatives par batch avant abandon
BASE_BACKOFF = 2.0       # secondes ; doublées à chaque échec
PAUSE_BETWEEN_BATCHES = 0.5


# --------------------------------------------------------------------------- #
# 1. Chargement
# --------------------------------------------------------------------------- #

def load_events(path: Path) -> list[dict]:
    """Charge le jeu d'événements nettoyé produit par `preprocess.py`."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} introuvable. Exécutez d'abord : python src/preprocess.py"
        )
    with path.open(encoding="utf-8") as f:
        events = json.load(f)
    print(f"[1/5] {len(events)} événements chargés depuis {path.name}")
    return events


# --------------------------------------------------------------------------- #
# 2. Découpage et métadonnées
# --------------------------------------------------------------------------- #

def build_header(event: dict) -> str:
    """Construit la ligne de contexte préfixée à chaque chunk de l'événement.

    Elle garantit qu'un chunk isolé reste rattachable à son événement : titre,
    période et lieu y figurent toujours, même pour le troisième chunk d'une
    description de 9 000 caractères.
    """
    parts = [
        event.get("title_fr") or "Sans titre",
        event.get("daterange_fr") or "",
        " ".join(
            p for p in (event.get("location_name"), event.get("location_city")) if p
        ),
    ]
    return " — ".join(p for p in parts if p)


def build_metadata(event: dict, chunk_index: int, n_chunks: int) -> dict:
    """Métadonnées attachées à chaque chunk.

    `uid` est indispensable : il permet, à l'évaluation, de remonter d'un chunk
    récupéré vers l'événement attendu du jeu de test annoté.
    """
    return {
        "uid": event.get("uid"),
        "title": event.get("title_fr"),
        "daterange": event.get("daterange_fr"),
        "firstdate_begin": event.get("firstdate_begin"),
        "lastdate_end": event.get("lastdate_end"),
        "location_name": event.get("location_name"),
        "location_address": event.get("location_address"),
        "location_city": event.get("location_city"),
        "url": event.get("canonicalurl"),
        "chunk_index": chunk_index,
        "n_chunks": n_chunks,
    }


def build_documents(events: list[dict]) -> list[Document]:
    """Découpe le champ `texte` de chaque événement et produit les Documents LangChain."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    documents: list[Document] = []
    for event in events:
        texte = (event.get("texte") or "").strip()
        if not texte:
            continue  # filet de sécurité : preprocess.py les a déjà écartés

        chunks = splitter.split_text(texte)
        header = build_header(event)

        for i, chunk in enumerate(chunks):
            documents.append(
                Document(
                    page_content=f"{header}\n\n{chunk}",
                    metadata=build_metadata(event, i, len(chunks)),
                )
            )

    n_multi = sum(1 for d in documents if d.metadata["n_chunks"] > 1)
    print(
        f"[2/5] {len(documents)} chunks produits "
        f"({len(documents) / len(events):.2f} chunk/événement en moyenne ; "
        f"{n_multi} chunks issus d'événements découpés en plusieurs morceaux)"
    )
    return documents


# --------------------------------------------------------------------------- #
# 3. Vectorisation par batchs, avec cache et backoff
# --------------------------------------------------------------------------- #

def embed_batch_with_retry(
    embeddings: MistralAIEmbeddings, texts: list[str], batch_no: int
) -> list[list[float]]:
    """Vectorise un batch, avec réessais et attente exponentielle.

    L'API Mistral applique un rate limit ; un échec ponctuel est la norme, pas
    l'exception. On réessaie plutôt que d'interrompre la vectorisation complète.
    """
    for attempt in range(MAX_RETRIES):
        try:
            return embeddings.embed_documents(texts)
        except Exception as exc:  # noqa: BLE001 — on veut rattraper tout échec réseau/API
            wait = BASE_BACKOFF * (2**attempt)
            print(
                f"      batch {batch_no} — échec ({type(exc).__name__}: {exc}). "
                f"Nouvelle tentative dans {wait:.0f} s "
                f"[{attempt + 1}/{MAX_RETRIES}]"
            )
            time.sleep(wait)

    raise RuntimeError(
        f"Batch {batch_no} : {MAX_RETRIES} tentatives échouées. "
        "Vectorisation interrompue — le cache est conservé, relancez le script."
    )


def embed_documents(documents: list[Document], force: bool = False) -> np.ndarray:
    """Vectorise tous les chunks, en reprenant le cache disque si présent."""
    if force and CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    embeddings = MistralAIEmbeddings(model=EMBED_MODEL)
    texts = [d.page_content for d in documents]
    n_batches = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE

    vectors: list[np.ndarray] = []
    for batch_no in range(n_batches):
        cache_file = CACHE_DIR / f"batch_{batch_no:05d}.npy"

        if cache_file.exists():
            vectors.append(np.load(cache_file))
            continue

        start = batch_no * BATCH_SIZE
        batch = texts[start : start + BATCH_SIZE]

        vecs = np.array(
            embed_batch_with_retry(embeddings, batch, batch_no), dtype="float32"
        )
        if vecs.shape[1] != EMBED_DIM:
            raise ValueError(
                f"Dimension inattendue : {vecs.shape[1]} au lieu de {EMBED_DIM}"
            )

        np.save(cache_file, vecs)
        vectors.append(vecs)

        done = min(start + BATCH_SIZE, len(texts))
        print(f"[3/5] {done}/{len(texts)} chunks vectorisés", end="\r", flush=True)
        time.sleep(PAUSE_BETWEEN_BATCHES)

    matrix = np.vstack(vectors)
    if matrix.shape[0] != len(documents):
        raise ValueError(
            f"Incohérence cache/chunks : {matrix.shape[0]} vecteurs pour "
            f"{len(documents)} chunks. Relancez avec --force."
        )

    print(f"[3/5] {matrix.shape[0]} chunks vectorisés (dimension {matrix.shape[1]})")
    return matrix


# --------------------------------------------------------------------------- #
# 4. Index FAISS
# --------------------------------------------------------------------------- #

def build_index(documents: list[Document], matrix: np.ndarray) -> FAISS:
    """Construit l'index FAISS à partir des vecteurs déjà calculés.

    On passe par `from_embeddings` et non `from_documents` : les vecteurs existent
    déjà (cache), il serait absurde de les recalculer. L'objet `MistralAIEmbeddings`
    reste attaché à l'index pour vectoriser les *questions* à l'interrogation.
    """

    embeddings = MistralAIEmbeddings(model=EMBED_MODEL)

    # Normalisation L2 explicite. LangChain ignore silencieusement son propre
    # paramètre `normalize_L2` lorsque la métrique est MAX_INNER_PRODUCT.
    # `mistral-embed` renvoie aujourd'hui des vecteurs déjà unitaires (norme
    # vérifiée à 1.0), mais ce comportement n'est pas garanti par contrat : on ne
    # laisse pas la correction de l'index dépendre d'une propriété non documentée
    # d'une API tierce. Produit scalaire sur vecteurs unitaires = similarité cosinus.
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    matrix = (matrix / norms).astype("float32")

    store = FAISS.from_embeddings(
        text_embeddings=list(
            zip([d.page_content for d in documents], matrix.tolist())
        ),
        embedding=embeddings,
        metadatas=[d.metadata for d in documents],
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
    )

    print(f"[4/5] Index FAISS construit — {store.index.ntotal} vecteurs")
    return store


def save_index(store: FAISS, documents: list[Document]) -> None:
    """Persiste l'index et un manifeste de traçabilité."""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    store.save_local(str(INDEX_DIR))

    manifest = {
        "embed_model": EMBED_MODEL,
        "embed_dim": EMBED_DIM,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "distance_strategy": "cosine (inner product + L2 normalization)",
        "n_chunks": len(documents),
        "n_events": len({d.metadata["uid"] for d in documents}),
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (INDEX_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"[5/5] Index sauvegardé dans {INDEX_DIR}/ (+ manifest.json)")


# --------------------------------------------------------------------------- #
# Entrée
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="vide le cache d'embeddings et revectorise intégralement",
    )
    args = parser.parse_args()

    load_dotenv()

    events = load_events(INPUT_PATH)
    documents = build_documents(events)
    matrix = embed_documents(documents, force=args.force)
    store = build_index(documents, matrix)
    save_index(store, documents)

    print(
        f"\nTerminé : {len(events)} événements → {len(documents)} chunks → "
        f"{store.index.ntotal} vecteurs indexés."
    )


if __name__ == "__main__":
    main()
