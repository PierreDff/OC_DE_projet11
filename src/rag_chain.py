"""Étape 4 — Chaîne RAG : récupération, mise en contexte, génération.

Assemble la chaîne complète du POC :

    question → récupération (src/retriever) → construction du contexte
             → génération (mistral-small) → réponse + métriques

Le montage est **explicite**, et non délégué à une abstraction toute faite de
LangChain (type ``RetrievalQA``). Trois raisons :

- **visibilité** : chaque étape est un appel de fonction lisible, débogable ;
- **instrumentation** : latences et volumétrie de tokens sont mesurées à la
  source — ce sont les chiffres qui alimenteront l'estimation des coûts
  d'exploitation (OPEX) du passage en MVP ;
- **réutilisation** : la récupération vient de ``src/retriever.py``, la même
  brique que celle mesurée par l'évaluation et inspectée à la main. Une seule
  logique de récupération dans tout le projet.

Le parti pris central est l'**abstention**. Le système ne doit jamais inventer
un événement ni un détail (prix, horaire) absent du contexte récupéré. C'est la
condition pour qu'un système de recommandation soit digne de confiance ; c'est
aussi ce que mesurent les questions-pièges du jeu de test annoté.

Usage :
    python -c "from src.rag_chain import repondre; print(repondre('un concert de jazz').texte)"
"""

from __future__ import annotations

import datetime
import time
from dataclasses import dataclass, field

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_mistralai import ChatMistralAI

from src.retriever import (
    K_DEFAUT,
    Recuperation,
    charger_index,
    rechercher_evenements,
)

load_dotenv()

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

MODELE_CHAT = "mistral-small-2506"
"""Modèle de génération.

`mistral-small` suffit à une tâche de synthèse à partir d'un contexte fourni :
il n'a pas à *connaître* les événements, seulement à *reformuler* ceux qu'on lui
donne. Un modèle plus gros augmenterait le coût par requête sans gain mesurable
sur cette tâche — arbitrage à réexaminer en production, pas au POC.
"""

TEMPERATURE = 0.2
"""Basse, volontairement.

On ne veut pas de créativité : on veut une reformulation fidèle du contexte. Une
température élevée favoriserait des tournures inventées — exactement ce que
l'abstention cherche à empêcher.
"""


# --------------------------------------------------------------------------- #
# Ancrage temporel
# --------------------------------------------------------------------------- #

JOURS_FR = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
MOIS_FR = [
    "janvier", "février", "mars", "avril", "mai", "juin",
    "juillet", "août", "septembre", "octobre", "novembre", "décembre",
]


def date_du_jour_fr(date: datetime.date | None = None) -> str:
    """Formate une date en français, sans dépendre de la locale système.

    La locale (``locale.setlocale``) n'est pas fiable d'une machine à l'autre —
    notamment sous Windows, où le paquet linguistique n'est pas garanti
    installé. Une table statique suffit ici et évite la dépendance.

    Le paramètre ``date`` est injectable (au lieu d'appeler
    ``datetime.date.today()`` en dur dans le corps) : permet de tester la
    fonction avec une date fixe, sur le même principe que le filtre
    ``now(years=-1)`` de la collecte, jamais figé dans le code lui-même.
    """
    date = date or datetime.date.today()
    return f"{JOURS_FR[date.weekday()]} {date.day} {MOIS_FR[date.month - 1]} {date.year}"


# --------------------------------------------------------------------------- #
# Prompt système
# --------------------------------------------------------------------------- #

PROMPT_SYSTEME = """\
Tu es l'assistant de recommandation culturelle de Puls-Events, pour des \
événements à Lille.

Nous sommes aujourd'hui le {date_du_jour}.

═══════════════════════════════════════════════════════════════════
RÈGLE ABSOLUE, PRIORITAIRE SUR TOUTE AUTRE : NE JAMAIS INVENTER.
═══════════════════════════════════════════════════════════════════

Le CONTEXTE ci-dessous est ta SEULE et UNIQUE source. Tu ne sais rien des \
événements lillois en dehors de lui. Tu ne complètes JAMAIS avec des \
connaissances générales, des suppositions, ou ce qui « semble probable ».

La frontière n'est pas entre les sujets autorisés et interdits : elle est entre \
ce qui est ÉCRIT dans le CONTEXTE (tu peux le rapporter) et ce que tu DÉDUIS \
(tu ne le peux pas). Tu n'écris un fait QUE si tu peux le pointer, mot pour mot, \
dans le CONTEXTE.

- TITRE, LIEU → uniquement s'ils y figurent, à l'identique.
- DATE → rapporte-la exactement comme elle est écrite. N'AJOUTE PAS une année, \
  un jour ou une heure absents. Si le texte dit « vendredi 5 juin » sans année, \
  tu écris « vendredi 5 juin » — pas « 5 juin 2026 ».
- PÉRIODE DEMANDÉE (relative — « ce week-end », « cette semaine », « demain »… \
  — OU explicite — « la semaine du 20 au 27 juillet », « en août 2026 »…) → \
  compare TOUJOURS la date de chaque événement à la période demandée : à la \
  date du jour indiquée plus haut si la période est relative, à la plage \
  donnée par l'utilisateur si elle est explicite. Un événement qui propose \
  PLUSIEURS occurrences (dates répétées, série, festival sur plusieurs \
  semaines) doit être filtré occurrence par occurrence : ne cite QUE les \
  dates qui tombent réellement dans la période demandée, même si d'autres \
  dates du même événement figurent dans le CONTEXTE. Si aucune date ne \
  correspond, dis-le clairement plutôt que de citer un événement ou une \
  occurrence hors période, même s'il porte sur le bon thème.
- PRIX, GRATUITÉ, RÉSERVATION → il n'existe aucun champ prix structuré, donc tu \
  ne CALCULES ni ne DEVINES jamais un tarif. MAIS si la description d'un \
  événement mentionne explicitement une information tarifaire — « gratuit », \
  « entrée libre », « 8 € », « sur réservation », « réservez » — tu peux la \
  rapporter, en la rattachant à CET événement précis et à aucun autre. Citer ce \
  qui est écrit : oui. Inventer ce qui manque : jamais. Si l'information de prix \
  n'est pas dans le texte, tu ne dis rien à son sujet, ou tu indiques qu'elle \
  n'est pas précisée.

Tu ne FUSIONNES JAMAIS deux événements. Chaque événement du CONTEXTE est \
distinct : son titre, ses dates et son lieu vont ensemble et ne se mélangent pas \
avec ceux d'un autre événement. Ne combine pas le nom de l'un avec le lieu d'un \
autre.

Si le CONTEXTE ne permet pas de répondre — aucun événement pertinent, ou détail \
demandé absent — dis-le franchement, par exemple : « Je n'ai pas d'information à \
ce sujet dans les événements disponibles. » S'abstenir est une bonne réponse ; \
inventer ne l'est jamais.

───────────────────────────────────────────────────────────────────
FORME DE LA RÉPONSE
───────────────────────────────────────────────────────────────────

Réponds en français, de façon concise et naturelle, comme un conseiller \
culturel — pas en liste brute. Pour chaque événement recommandé, donne son \
titre, ses dates et son lieu exactement tels qu'ils apparaissent dans le \
CONTEXTE. N'ajoute aucun détail qui n'y est pas.
"""


# --------------------------------------------------------------------------- #
# Structure de réponse
# --------------------------------------------------------------------------- #


@dataclass
class ReponseRAG:
    """Réponse générée, accompagnée de sa traçabilité et de ses métriques.

    Pendant, côté génération, de la ``Recuperation`` côté récupération : la
    réponse ne circule jamais sans les éléments qui permettent de la vérifier et
    d'en mesurer le coût.

    Attributs
    ---------
    texte:
        La réponse en langage naturel produite par le modèle.
    question:
        La question d'origine — utile pour journaliser et pour l'évaluation.
    documents_sources:
        Les événements récupérés qui ont servi de contexte. Permettent de
        vérifier a posteriori si la réponse s'y tient (fidélité) et d'afficher
        les sources dans la démo.
    uids_sources:
        Identifiants des événements sources — appariement avec le jeu de test.
    latence_recuperation_ms:
        Temps passé dans la récupération (embedding + FAISS + déduplication).
    latence_generation_ms:
        Temps passé dans l'appel au LLM de génération. Poste de coût dominant.
    latence_totale_ms:
        Somme des deux — ce que perçoit l'utilisateur.
    tokens_entree, tokens_sortie:
        Volumétrie de tokens de l'appel de génération, si l'API la renvoie.
        Base directe de l'estimation OPEX : coût = f(tokens) par le tarif Mistral.
        ``None`` si l'information n'est pas exposée par le SDK.
    """

    texte: str
    question: str
    documents_sources: list[Document] = field(default_factory=list)
    uids_sources: list[str] = field(default_factory=list)
    latence_recuperation_ms: float = 0.0
    latence_generation_ms: float = 0.0
    latence_totale_ms: float = 0.0
    tokens_entree: int | None = None
    tokens_sortie: int | None = None


# --------------------------------------------------------------------------- #
# Construction du contexte
# --------------------------------------------------------------------------- #


def construire_contexte(recuperation: Recuperation) -> str:
    """Met en forme les événements récupérés en un bloc de texte pour le LLM.

    Chaque événement est présenté avec ses métadonnées structurées (titre,
    dates, lieu, lien) suivies d'un extrait de sa description. Le format est
    volontairement régulier et numéroté : il aide le modèle à référencer les
    événements et à ne pas les confondre.

    On s'appuie sur ``page_content`` (qui porte déjà l'en-tête titre/dates/lieu
    ajouté au chunking) et sur les métadonnées, sans refaire de mise en forme
    coûteuse.
    """
    if not recuperation.documents:
        return "(Aucun événement pertinent trouvé.)"

    blocs: list[str] = []
    for i, doc in enumerate(recuperation.documents, start=1):
        meta = doc.metadata
        # page_content = en-tête (titre — dates — lieu) + corps de la description.
        # On le donne tel quel : c'est le texte le plus fidèle à l'événement.
        contenu = doc.page_content.strip()
        lien = meta.get("url") or ""
        bloc = f"[Événement {i}]\n{contenu}"
        if lien:
            bloc += f"\n(Fiche : {lien})"
        blocs.append(bloc)

    return "\n\n".join(blocs)


def construire_messages(
    question: str,
    contexte: str,
    date: datetime.date | None = None,
) -> list[tuple[str, str]]:
    """Assemble les messages envoyés au modèle de chat.

    Le contexte est placé dans le message utilisateur, encadrant explicitement
    la question. Le prompt système, lui, porte les règles de comportement
    (abstention, citation, ancrage temporel), invariantes d'une question à
    l'autre — à l'exception de la date du jour, interpolée à chaque appel.

    Sans cette date, le modèle ne peut pas évaluer si un événement correspond
    à une période relative (« ce week-end », « cette semaine ») : il ne connaît
    que le texte du CONTEXTE, jamais la date d'exécution. ``date`` est
    injectable pour les tests ; ``None`` retombe sur la date du jour réelle.
    """
    message_utilisateur = (
        f"CONTEXTE — événements disponibles :\n\n{contexte}\n\n"
        f"---\n\n"
        f"QUESTION de l'utilisateur : {question}"
    )
    prompt_systeme = PROMPT_SYSTEME.format(date_du_jour=date_du_jour_fr(date))
    return [
        ("system", prompt_systeme),
        ("human", message_utilisateur),
    ]


# --------------------------------------------------------------------------- #
# Génération
# --------------------------------------------------------------------------- #


def _extraire_tokens(reponse) -> tuple[int | None, int | None]:
    """Extrait la volumétrie de tokens de la réponse du SDK, si présente.

    L'emplacement exact dépend de la version de ``langchain-mistralai`` :
    ``response_metadata['token_usage']`` ou ``usage_metadata``. On tente les
    deux et on renvoie ``(None, None)`` en dernier recours, sans faire échouer
    la génération pour autant — la volumétrie est un bonus d'instrumentation,
    pas une donnée critique.
    """
    # Emplacement 1 : usage_metadata (LangChain récent, normalisé)
    usage = getattr(reponse, "usage_metadata", None)
    if usage:
        entree = usage.get("input_tokens")
        sortie = usage.get("output_tokens")
        if entree is not None or sortie is not None:
            return entree, sortie

    # Emplacement 2 : response_metadata['token_usage'] (remonté du SDK Mistral)
    meta = getattr(reponse, "response_metadata", None) or {}
    usage = meta.get("token_usage") or meta.get("usage") or {}
    entree = usage.get("prompt_tokens")
    sortie = usage.get("completion_tokens")
    return entree, sortie


def repondre(
    question: str,
    k: int = K_DEFAUT,
    vectorstore=None,
    llm: ChatMistralAI | None = None,
) -> ReponseRAG:
    """Répond à une question en s'appuyant sur les événements indexés.

    Orchestre la chaîne complète : récupération, construction du contexte,
    génération. Mesure chaque étape séparément.

    Paramètres
    ----------
    question:
        Question de l'utilisateur, en langage naturel.
    k:
        Nombre d'événements distincts à fournir au modèle comme contexte.
    vectorstore:
        Index déjà chargé. Si ``None``, il est chargé à la volée — pratique pour
        un appel ponctuel, à éviter dans une boucle (l'évaluation charge l'index
        une fois et le passe à chaque appel).
    llm:
        Modèle de chat déjà instancié. Même logique que ``vectorstore`` : on
        évite de le recréer à chaque question dans une boucle d'évaluation.

    Retourne
    --------
    ReponseRAG
        Réponse, événements sources et métriques.
    """
    if not question or not question.strip():
        raise ValueError("La question ne peut pas être vide.")

    if vectorstore is None:
        vectorstore = charger_index()
    if llm is None:
        llm = ChatMistralAI(model=MODELE_CHAT, temperature=TEMPERATURE)

    debut_total = time.perf_counter()

    # 1. Récupération — déléguée au module partagé, avec ses propres métriques.
    recuperation = rechercher_evenements(vectorstore, question, k=k)

    # 2. Mise en contexte.
    contexte = construire_contexte(recuperation)
    messages = construire_messages(question, contexte)

    # 3. Génération — l'appel réseau coûteux, chronométré à part.
    debut_generation = time.perf_counter()
    reponse = llm.invoke(messages)
    latence_generation = (time.perf_counter() - debut_generation) * 1000

    latence_totale = (time.perf_counter() - debut_total) * 1000
    tokens_entree, tokens_sortie = _extraire_tokens(reponse)

    return ReponseRAG(
        texte=reponse.content,
        question=question,
        documents_sources=recuperation.documents,
        uids_sources=recuperation.uids,
        latence_recuperation_ms=recuperation.latence_ms,
        latence_generation_ms=round(latence_generation, 1),
        latence_totale_ms=round(latence_totale, 1),
        tokens_entree=tokens_entree,
        tokens_sortie=tokens_sortie,
    )
