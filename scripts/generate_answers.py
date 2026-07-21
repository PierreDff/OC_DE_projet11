"""Étape d'évaluation (1/2) — Génération des réponses du système RAG.

Lit le jeu de test annoté, lance la chaîne RAG sur chaque question, et
sauvegarde les réponses générées avec leur contexte et leurs métriques dans
`data/eval/rag_answers.json`.

Pourquoi un script séparé de l'évaluation ?

- **Économie d'API.** C'est ici, et seulement ici, qu'on appelle Mistral pour
  générer. Le calcul des métriques (recall, precision, abstention) se fait
  ensuite sans aucun appel réseau, et l'évaluation RAGAS peut être relancée
  autant que nécessaire sans re-générer.
- **Reproductibilité.** Les réponses évaluées sont figées sur disque, comme les
  données et l'index. On sait exactement quelles réponses ont été notées.
- **Débogage.** Un score surprenant se comprend en relisant la réponse et le
  contexte qui l'ont produit — présents dans le fichier de sortie.

Le LLM n'étant pas déterministe, deux exécutions ne produisent pas des réponses
identiques. Ce fichier fige *une* exécution : c'est elle qui sera évaluée.

Usage :
    python scripts/generate_answers.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Le script est lancé depuis la racine du projet ; on rend `src` importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag_chain import ChatMistralAI, MODELE_CHAT, TEMPERATURE, repondre  # noqa: E402
from src.retriever import charger_index  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
QA_PATH = ROOT / "data" / "eval" / "qa_dataset.json"
SORTIE = ROOT / "data" / "eval" / "rag_answers.json"

# Pause entre deux questions : marge de sécurité vis-à-vis du rate limit.
# mistral-small-2506 autorise 5 req/s ; on est très en dessous, mais une petite
# pause lisse la charge et évite tout pic accidentel.
PAUSE_ENTRE_QUESTIONS = 0.5


def charger_jeu_de_test(path: Path) -> list[dict]:
    """Charge les questions du jeu de test annoté."""
    if not path.exists():
        raise FileNotFoundError(
            f"Jeu de test introuvable : {path}. "
            "Vérifiez data/eval/qa_dataset.json."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    questions = data.get("questions", [])
    if not questions:
        raise ValueError("Le jeu de test ne contient aucune question.")
    return questions


def generer_reponses(questions: list[dict]) -> list[dict]:
    """Lance la chaîne RAG sur chaque question et collecte les résultats.

    L'index et le modèle de chat sont chargés une seule fois, puis réutilisés à
    chaque appel : les recréer à chaque question rechargerait l'index (lent) et
    n'apporterait rien.
    """
    print(f"Chargement de l'index et du modèle {MODELE_CHAT}…")
    vectorstore = charger_index()
    llm = ChatMistralAI(model=MODELE_CHAT, temperature=TEMPERATURE)

    resultats: list[dict] = []
    total = len(questions)

    for i, item in enumerate(questions, start=1):
        qid = item["id"]
        question = item["question"]
        print(f"[{i}/{total}] {qid} — {question[:60]}…")

        reponse = repondre(question, vectorstore=vectorstore, llm=llm)

        resultats.append(
            {
                # Rappel de l'annotation, pour que l'évaluation ait tout sous la main
                "id": qid,
                "question": question,
                "type": item["type"],
                "uids_attendus": item["uids_attendus"],
                "ground_truth": item["ground_truth"],
                # Produits par le système RAG
                "answer": reponse.texte,
                "contexts": [doc.page_content for doc in reponse.documents_sources],
                "uids_recuperes": reponse.uids_sources,
                # Métriques d'exécution — alimentent le rapport et l'estimation OPEX
                "latence_recuperation_ms": reponse.latence_recuperation_ms,
                "latence_generation_ms": reponse.latence_generation_ms,
                "latence_totale_ms": reponse.latence_totale_ms,
                "tokens_entree": reponse.tokens_entree,
                "tokens_sortie": reponse.tokens_sortie,
            }
        )

        time.sleep(PAUSE_ENTRE_QUESTIONS)

    return resultats


def main() -> None:
    questions = charger_jeu_de_test(QA_PATH)
    print(f"{len(questions)} questions chargées depuis {QA_PATH.name}\n")

    resultats = generer_reponses(questions)

    SORTIE.parent.mkdir(parents=True, exist_ok=True)
    SORTIE.write_text(
        json.dumps(resultats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Bilan rapide des métriques d'exécution
    lat_moy = sum(r["latence_totale_ms"] for r in resultats) / len(resultats)
    tokens_in = sum(r["tokens_entree"] or 0 for r in resultats)
    tokens_out = sum(r["tokens_sortie"] or 0 for r in resultats)

    print(f"\n{len(resultats)} réponses générées et écrites dans {SORTIE}")
    print(f"Latence totale moyenne : {lat_moy:.0f} ms/question")
    print(f"Tokens cumulés : {tokens_in} en entrée, {tokens_out} en sortie")
    print("\nÉtape suivante : python scripts/evaluate_rag.py")


if __name__ == "__main__":
    main()
