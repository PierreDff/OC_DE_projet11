"""Étape d'évaluation (2/2) — Calcul des métriques sur les réponses générées.

Lit `data/eval/rag_answers.json` (produit par `scripts/generate_answers.py`) et
calcule trois familles de métriques :

1. **Récupération** — ``recall@k`` et ``precision@k`` sur les identifiants
   d'événements (``uid``). Aucun appel réseau.

2. **Abstention** — sur les questions-pièges, le système reconnaît-il qu'il ne
   dispose pas de l'information ? Aucun appel réseau.

3. **Génération** (optionnel, ``--avec-juge``) — ``faithfulness`` et
   ``answer_relevancy`` via un LLM juge (``mistral-small-2506``). Implémenté
   directement plutôt que délégué à RAGAS, pour éviter un conflit de
   dépendances avec ``langchain-community`` 0.4.2 (requise par FAISS).
   Le principe est identique : le juge évalue chaque réponse contre le
   contexte et la question. NB : au POC, le juge utilise le même modèle que
   le générateur ; un juge plus capable (mistral-medium/large) est recommandé
   en production pour ne pas hériter des angles morts du générateur.

Usage :
    python scripts/evaluate_rag.py
    python scripts/evaluate_rag.py --avec-juge
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

REPONSES = ROOT / "data" / "eval" / "rag_answers.json"
RESULTATS = ROOT / "data" / "eval" / "eval_results.json"

PAUSE_ENTRE_APPELS = 3.0
"""Pause en secondes entre deux appels au LLM juge.

On prend une marge pour éviter tout 429 (rate limit). Sur 24 appels (2 par
question × 12), ça ajoute ~72 s au total — acceptable pour un POC.
"""

MODELE_JUGE = "mistral-small-2506"
"""Modèle utilisé pour juger les réponses.

Au POC, le juge utilise le **même modèle que le générateur** (``mistral-small-2506``),
par simplicité de mise en œuvre. Limite assumée : un juge de même capacité que le
modèle évalué peut hériter de ses angles morts, ce qui borne la portée des scores de
faithfulness et d'answer_relevancy. En production, un juge plus capable
(``mistral-medium`` ou ``mistral-large``) serait préférable ; son débit d'appel plus
restreint doit alors être pris en compte.
"""

MARQUEURS_ABSTENTION = [
    r"n'ai pas d'information",
    r"pas d'information",
    r"ne dispose pas",
    r"aucun.{0,40}n'est référencé",
    r"aucun.{0,40}n'est disponible",
    r"n'est pas précisé",
    r"n'est pas indiqué",
    r"n'est pas mentionné",
    r"je ne peux pas répondre",
    r"ne figure pas dans",
]

PAUSE_ENTRE_APPELS = 2.5
"""Pause entre deux appels au juge, marge pour éviter tout 429 (rate limit)."""


# --------------------------------------------------------------------------- #
# Métriques de récupération
# --------------------------------------------------------------------------- #


def recall_at_k(attendus: list[str], recuperes: list[str]) -> float | None:
    """Proportion des événements attendus effectivement récupérés."""
    if not attendus:
        return None
    trouves = set(attendus) & set(recuperes)
    return len(trouves) / len(attendus)


def precision_at_k(attendus: list[str], recuperes: list[str]) -> float | None:
    """Proportion des événements récupérés qui étaient attendus."""
    if not attendus or not recuperes:
        return None
    trouves = set(attendus) & set(recuperes)
    return len(trouves) / len(recuperes)


# --------------------------------------------------------------------------- #
# Métrique d'abstention
# --------------------------------------------------------------------------- #


def est_abstention(reponse: str) -> bool:
    """Détecte si la réponse reconnaît une absence d'information."""
    return any(re.search(m, reponse, re.IGNORECASE) for m in MARQUEURS_ABSTENTION)


# --------------------------------------------------------------------------- #
# Évaluation locale (sans appel réseau)
# --------------------------------------------------------------------------- #


def evaluer_localement(reponses: list[dict]) -> list[dict]:
    """Calcule les métriques de récupération et d'abstention."""
    resultats = []

    for r in reponses:
        attendus = r["uids_attendus"]
        recuperes = r["uids_recuperes"]
        est_piege = r["type"] == "piege" or not attendus

        resultats.append(
            {
                "id": r["id"],
                "type": r["type"],
                "question": r["question"],
                "recall_at_k": recall_at_k(attendus, recuperes),
                "precision_at_k": precision_at_k(attendus, recuperes),
                "abstention": est_abstention(r["answer"]) if est_piege else None,
                "n_attendus": len(attendus),
                "n_recuperes": len(recuperes),
                "uids_trouves": sorted(set(attendus) & set(recuperes)),
                "latence_totale_ms": r["latence_totale_ms"],
                "tokens_entree": r["tokens_entree"],
                "tokens_sortie": r["tokens_sortie"],
            }
        )

    return resultats


# --------------------------------------------------------------------------- #
# LLM-as-judge (remplace RAGAS)
# --------------------------------------------------------------------------- #

PROMPT_FAITHFULNESS = """\
Tu es un évaluateur rigoureux. Tu reçois un CONTEXTE (des événements culturels) \
et une RÉPONSE produite par un système automatique.

Ta tâche : vérifier si CHAQUE affirmation de la RÉPONSE est fidèle au CONTEXTE.

Procédure :
1. Identifie chaque affirmation factuelle dans la RÉPONSE.
2. Pour chaque affirmation, vérifie si elle est soutenue par le CONTEXTE.
3. Compte le nombre total d'affirmations et le nombre d'affirmations soutenues.

Réponds UNIQUEMENT avec un objet JSON (sans backticks, sans texte autour) :
{{"total": <nombre>, "soutenues": <nombre>, "score": <entre 0.0 et 1.0>}}

Le score = soutenues / total. Si la réponse est une abstention ("je n'ai pas \
d'information"), score = 1.0 (l'abstention est fidèle).

CONTEXTE :
{contexte}

RÉPONSE :
{reponse}
"""

PROMPT_RELEVANCY = """\
Tu es un évaluateur rigoureux. Tu reçois une QUESTION posée par un utilisateur \
et une RÉPONSE produite par un système automatique.

Ta tâche : évaluer si la RÉPONSE répond bien à la QUESTION posée.

Critères :
- La réponse aborde-t-elle le sujet de la question ?
- La réponse donne-t-elle l'information demandée (ou indique-t-elle honnêtement \
  qu'elle ne l'a pas) ?
- La réponse est-elle concise et utile ?

Réponds UNIQUEMENT avec un objet JSON (sans backticks, sans texte autour) :
{{"score": <entre 0.0 et 1.0>, "justification": "<1 phrase>"}}

QUESTION :
{question}

RÉPONSE :
{reponse}
"""


def appeler_juge(llm, prompt: str, max_retries: int = 3) -> dict | None:
    """Envoie un prompt au juge et parse la réponse JSON.

    Retry avec backoff exponentiel sur les erreurs transitoires (rate limit).
    """
    for tentative in range(max_retries):
        try:
            reponse = llm.invoke(prompt)
            texte = reponse.content.strip()
            # Nettoyer d'éventuels backticks
            texte = re.sub(r"^```json\s*", "", texte)
            texte = re.sub(r"\s*```$", "", texte)
            return json.loads(texte)
        except json.JSONDecodeError:
            print(f"    [juge] réponse non-JSON (tentative {tentative + 1})")
            if tentative < max_retries - 1:
                time.sleep(PAUSE_ENTRE_APPELS * (tentative + 1))
        except Exception as e:
            print(f"    [juge] erreur: {e} (tentative {tentative + 1})")
            if tentative < max_retries - 1:
                time.sleep(PAUSE_ENTRE_APPELS * (tentative + 1))
    return None


def evaluer_avec_juge(reponses: list[dict]) -> dict | None:
    """Calcule faithfulness et answer_relevancy via un LLM juge.

    Même principe que RAGAS, implémenté directement :
    - faithfulness : chaque affirmation de la réponse est-elle soutenue par le
      contexte fourni ? (détecte les hallucinations)
    - answer_relevancy : la réponse répond-elle bien à la question posée ?

    Au POC, le juge utilise le même modèle que le générateur (mistral-small-2506) ;
    un juge plus capable est recommandé en production pour ne pas hériter de ses
    angles morts.
    """
    try:
        from langchain_mistralai import ChatMistralAI
    except ImportError:
        print("[juge] langchain-mistralai non disponible.")
        return None

    print(f"\n[juge] Évaluation avec {MODELE_JUGE} (LLM-as-judge)…")
    print(f"[juge] 2 appels par question × {len(reponses)} questions = {2 * len(reponses)} appels")
    print(f"[juge] Pause de {PAUSE_ENTRE_APPELS}s entre appels (rate limit).")

    llm = ChatMistralAI(model=MODELE_JUGE, temperature=0)

    scores_par_question = {}
    faithfulness_scores = []
    relevancy_scores = []

    for i, r in enumerate(reponses, start=1):
        qid = r["id"]
        print(f"  [{i}/{len(reponses)}] {qid}…", end=" ", flush=True)

        contexte = "\n\n".join(r["contexts"])

        # --- Faithfulness ---
        prompt_f = PROMPT_FAITHFULNESS.format(contexte=contexte, reponse=r["answer"])
        result_f = appeler_juge(llm, prompt_f)
        faith = result_f.get("score", 0.0) if result_f else None

        time.sleep(PAUSE_ENTRE_APPELS)

        # --- Answer relevancy ---
        prompt_r = PROMPT_RELEVANCY.format(question=r["question"], reponse=r["answer"])
        result_r = appeler_juge(llm, prompt_r)
        relev = result_r.get("score", 0.0) if result_r else None

        time.sleep(PAUSE_ENTRE_APPELS)

        scores_par_question[qid] = {
            "faithfulness": faith,
            "answer_relevancy": relev,
            "faith_detail": result_f,
            "relev_detail": result_r,
        }

        f_str = f"{faith:.2f}" if faith is not None else "ERR"
        r_str = f"{relev:.2f}" if relev is not None else "ERR"
        print(f"faith={f_str}  relev={r_str}")

        if faith is not None:
            faithfulness_scores.append(faith)
        if relev is not None:
            relevancy_scores.append(relev)

    moyennes = {}
    if faithfulness_scores:
        moyennes["faithfulness"] = sum(faithfulness_scores) / len(faithfulness_scores)
    if relevancy_scores:
        moyennes["answer_relevancy"] = sum(relevancy_scores) / len(relevancy_scores)

    return {
        "methode": "LLM-as-judge (mistral-small-2506, implémentation directe)",
        "par_question": scores_par_question,
        "moyennes": moyennes,
    }


# --------------------------------------------------------------------------- #
# Agrégation et affichage
# --------------------------------------------------------------------------- #


def moyenne(valeurs: list[float | None]) -> float | None:
    """Moyenne en ignorant les valeurs non applicables."""
    valides = [v for v in valeurs if v is not None]
    return sum(valides) / len(valides) if valides else None


def fmt(valeur: float | None, largeur: int = 6) -> str:
    return f"{valeur:>{largeur}.2f}" if valeur is not None else f"{'—':>{largeur}}"


def agreger_par_type(resultats: list[dict]) -> dict:
    """Regroupe les scores par typologie de question."""
    par_type = defaultdict(list)
    for r in resultats:
        par_type[r["type"]].append(r)

    agrege = {}
    for typ, items in par_type.items():
        agrege[typ] = {
            "n_questions": len(items),
            "recall_at_k": moyenne([i["recall_at_k"] for i in items]),
            "precision_at_k": moyenne([i["precision_at_k"] for i in items]),
            "taux_abstention": moyenne(
                [1.0 if i["abstention"] else 0.0 for i in items if i["abstention"] is not None]
            ),
        }
    return agrege


def afficher(resultats: list[dict], agrege: dict, juge: dict | None) -> None:
    print("\n" + "=" * 78)
    print("DÉTAIL PAR QUESTION")
    print("=" * 78)
    print(f"{'id':<10} {'type':<12} {'recall':>7} {'prec.':>7} {'abst.':>7}  trouvés")
    print("-" * 78)
    for r in resultats:
        abst = "—" if r["abstention"] is None else ("oui" if r["abstention"] else "NON")
        trouves = f"{len(r['uids_trouves'])}/{r['n_attendus']}" if r["n_attendus"] else "—"
        print(
            f"{r['id']:<10} {r['type']:<12} {fmt(r['recall_at_k'], 7)} "
            f"{fmt(r['precision_at_k'], 7)} {abst:>7}  {trouves}"
        )

    print("\n" + "=" * 78)
    print("SYNTHÈSE PAR TYPOLOGIE")
    print("=" * 78)
    print(f"{'type':<14} {'n':>3} {'recall@k':>10} {'precision@k':>13} {'abstention':>12}")
    print("-" * 78)
    for typ, s in sorted(agrege.items()):
        print(
            f"{typ:<14} {s['n_questions']:>3} {fmt(s['recall_at_k'], 10)} "
            f"{fmt(s['precision_at_k'], 13)} {fmt(s['taux_abstention'], 12)}"
        )

    lat = moyenne([r["latence_totale_ms"] for r in resultats])
    tok_in = sum(r["tokens_entree"] or 0 for r in resultats)
    tok_out = sum(r["tokens_sortie"] or 0 for r in resultats)
    print("\n" + "=" * 78)
    print("COÛT D'EXÉCUTION")
    print("=" * 78)
    print(f"Latence moyenne     : {lat:.0f} ms/question")
    print(f"Tokens cumulés      : {tok_in} entrée / {tok_out} sortie")
    print(f"Tokens par question : {tok_in // len(resultats)} entrée / {tok_out // len(resultats)} sortie")

    if juge:
        print("\n" + "=" * 78)
        print("MÉTRIQUES DE GÉNÉRATION (LLM-as-judge)")
        print("=" * 78)
        for cle, val in juge["moyennes"].items():
            print(f"  {cle:<20} : {val:.3f}")
        print()
        print(f"  {'id':<10} {'faith.':>8} {'relev.':>8}")
        print("  " + "-" * 28)
        for qid, scores in juge["par_question"].items():
            f = scores.get("faithfulness")
            r = scores.get("answer_relevancy")
            print(f"  {qid:<10} {fmt(f, 8)} {fmt(r, 8)}")

    print("\nLecture :")
    print("  recall@k    — part des événements attendus effectivement récupérés.")
    print("  precision@k — part des événements récupérés qui étaient attendus.")
    print("                Plafonne à 0,20 sur les factuelles (1 attendu pour k=5).")
    print("  abstention  — le système reconnaît-il l'absence d'information ? (pièges)")
    print("  faithfulness — la réponse est-elle fidèle au contexte ? (0=hallucination, 1=fidèle)")
    print("  relevancy    — la réponse répond-elle à la question ? (0=hors-sujet, 1=pertinente)")
    print("  —            — métrique non applicable à ce type de question.")


# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--avec-juge",
        action="store_true",
        help="ajoute faithfulness et answer_relevancy via LLM juge (appels API)",
    )
    args = parser.parse_args()

    if not REPONSES.exists():
        raise FileNotFoundError(
            f"{REPONSES} introuvable. Lancer d'abord :\n"
            "    python scripts/generate_answers.py"
        )

    reponses = json.loads(REPONSES.read_text(encoding="utf-8"))
    print(f"{len(reponses)} réponses chargées depuis {REPONSES.name}")

    resultats = evaluer_localement(reponses)
    agrege = agreger_par_type(resultats)
    juge = evaluer_avec_juge(reponses) if args.avec_juge else None

    afficher(resultats, agrege, juge)

    RESULTATS.write_text(
        json.dumps(
            {"par_question": resultats, "par_type": agrege, "juge": juge},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nRésultats détaillés écrits dans {RESULTATS}")


if __name__ == "__main__":
    main()
