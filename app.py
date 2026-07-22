"""Démo Streamlit — Puls-Events RAG.

Interface de démonstration du POC : pose une question en langage naturel,
la chaîne RAG (src/rag_chain.py) répond en s'appuyant sur les événements
lillois indexés dans FAISS.

Ce fichier est volontairement fin : toute la logique métier (récupération,
prompt, abstention, mesures) reste dans src/retriever.py et src/rag_chain.py.
L'UI ne fait que les appeler et afficher le résultat — aucune règle de gestion
ne doit être dupliquée ici.

Usage :
    streamlit run app.py

Placement : ce fichier doit vivre à la racine du dépôt, au même niveau que
le dossier `src/`, pour que `from src.rag_chain import repondre` fonctionne.
"""

from __future__ import annotations

import streamlit as st
from langchain_mistralai import ChatMistralAI

from src.rag_chain import MODELE_CHAT, TEMPERATURE, ReponseRAG, repondre
from src.retriever import FETCH_K_DEFAUT, K_DEFAUT, charger_index

# --------------------------------------------------------------------------- #
# Configuration de page
# --------------------------------------------------------------------------- #

st.set_page_config(
    page_title="Puls-Events — Assistant culturel",
    page_icon="🎭",
    layout="centered",
)

# --------------------------------------------------------------------------- #
# Ressources coûteuses — chargées une seule fois
# --------------------------------------------------------------------------- #
# L'index FAISS et le modèle de chat sont mis en cache par Streamlit : sans
# cela, chaque interaction relance un chargement d'index et une init de client
# API. C'est la même logique que passer `vectorstore=` / `llm=` en boucle
# d'évaluation (src/rag_chain.py) plutôt que de les recréer à chaque appel.


@st.cache_resource(show_spinner="Chargement de l'index FAISS…")
def get_vectorstore():
    return charger_index()


@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatMistralAI(model=MODELE_CHAT, temperature=TEMPERATURE)


# --------------------------------------------------------------------------- #
# État de session
# --------------------------------------------------------------------------- #
# Important : cet historique n'est QUE pour l'affichage. Le POC est stateless
# et mono-tour — chaque appel à `repondre()` ne reçoit que la question
# courante, jamais les tours précédents. La mémoire conversationnelle réelle
# (contexte des échanges précédents injecté dans la génération) est
# explicitement hors périmètre du POC et reportée à la Mission 13.

if "messages" not in st.session_state:
    st.session_state.messages = []  # liste de {"role": ..., "content": ..., "reponse": ReponseRAG | None}

if "erreur_index" not in st.session_state:
    st.session_state.erreur_index = None


# --------------------------------------------------------------------------- #
# Barre latérale
# --------------------------------------------------------------------------- #

with st.sidebar:
    st.header("Réglages de démo")

    k = st.slider(
        "Nombre d'événements récupérés (k)",
        min_value=1,
        max_value=10,
        value=K_DEFAUT,
        help="Nombre d'événements distincts fournis comme contexte au LLM.",
    )

    afficher_metriques = st.checkbox("Afficher les métriques (latence, tokens)", value=True)
    afficher_sources = st.checkbox("Afficher les événements sources", value=True)

    st.divider()
    if st.button("🔄 Réinitialiser la conversation"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption(
        "Le système ne répond qu'à partir des événements récupérés dans "
        "l'index. Il s'abstient plutôt que d'inventer un prix, une date ou "
        "un lieu absent du contexte."
    )
    st.caption(f"fetch_k = {FETCH_K_DEFAUT} · modèle génération = {MODELE_CHAT}")


# --------------------------------------------------------------------------- #
# En-tête
# --------------------------------------------------------------------------- #

st.title("🎭 Puls-Events — Assistant culturel")
st.caption(
    "Posez une question sur les événements culturels à Lille (concerts, "
    "expositions, ateliers…). Le POC répond uniquement à partir des "
    "événements indexés — il ne devine jamais ce qui n'y figure pas."
)

# --------------------------------------------------------------------------- #
# Chargement de l'index (une fois, avec message d'erreur clair)
# --------------------------------------------------------------------------- #

try:
    vectorstore = get_vectorstore()
    llm = get_llm()
except (FileNotFoundError, RuntimeError) as exc:
    st.error(f"Impossible de démarrer la démo : {exc}")
    st.stop()


# --------------------------------------------------------------------------- #
# Affichage de l'historique
# --------------------------------------------------------------------------- #


def afficher_sources_reponse(reponse: ReponseRAG) -> None:
    """Affiche les événements sources dans un expander, un bloc par événement."""
    if not reponse.documents_sources:
        st.caption("Aucun événement source (abstention).")
        return

    with st.expander(f"📍 {len(reponse.documents_sources)} événement(s) source(s)"):
        for i, doc in enumerate(reponse.documents_sources, start=1):
            meta = doc.metadata
            titre = meta.get("title_fr") or meta.get("titre") or f"Événement {i}"
            lieu = meta.get("location_name") or meta.get("location_city") or ""
            dates = meta.get("daterange_fr") or ""
            lien = meta.get("url") or meta.get("canonicalurl") or ""

            st.markdown(f"**{i}. {titre}**")
            details = " · ".join(filter(None, [dates, lieu]))
            if details:
                st.caption(details)
            if lien:
                st.markdown(f"[Voir la fiche]({lien})")
            st.divider()


def afficher_metriques_reponse(reponse: ReponseRAG) -> None:
    """Affiche les métriques de latence et de tokens en petites colonnes."""
    col1, col2, col3 = st.columns(3)
    col1.metric("Récupération", f"{reponse.latence_recuperation_ms:.0f} ms")
    col2.metric("Génération", f"{reponse.latence_generation_ms:.0f} ms")
    col3.metric("Total", f"{reponse.latence_totale_ms:.0f} ms")

    if reponse.tokens_entree is not None or reponse.tokens_sortie is not None:
        st.caption(
            f"Tokens — entrée : {reponse.tokens_entree or '?'} · "
            f"sortie : {reponse.tokens_sortie or '?'}"
        )


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        reponse: ReponseRAG | None = message.get("reponse")
        if reponse is not None:
            if afficher_sources:
                afficher_sources_reponse(reponse)
            if afficher_metriques:
                afficher_metriques_reponse(reponse)


# --------------------------------------------------------------------------- #
# Saisie utilisateur
# --------------------------------------------------------------------------- #

question = st.chat_input("Ex. : Y-a-t-il un concert de jazz ce week-end à Lille ?")

if question:
    st.session_state.messages.append({"role": "user", "content": question, "reponse": None})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Recherche des événements et génération de la réponse…"):
            try:
                reponse = repondre(question, k=k, vectorstore=vectorstore, llm=llm)
            except Exception as exc:  # affichage démo : ne jamais planter la page
                st.error(f"Erreur lors de la génération : {exc}")
                st.stop()

        st.markdown(reponse.texte)
        if afficher_sources:
            afficher_sources_reponse(reponse)
        if afficher_metriques:
            afficher_metriques_reponse(reponse)

    st.session_state.messages.append(
        {"role": "assistant", "content": reponse.texte, "reponse": reponse}
    )
