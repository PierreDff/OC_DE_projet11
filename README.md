# Puls-Events RAG — POC de recommandation d'événements culturels

Proof of Concept d'un système de génération augmentée par récupération (RAG)
capable de répondre à des questions en langage naturel sur des événements
culturels, à partir des données publiques Open Agenda.

Réalisé dans le cadre d'une mission pour Puls-Events : démontrer la faisabilité
d'un chatbot de recommandation avant un éventuel déploiement à plus grande échelle.

---

## Objectifs

- Collecter et nettoyer les données d'événements publics Open Agenda
- Vectoriser les descriptions d'événements avec les embeddings Mistral
- Indexer ces vecteurs dans une base FAISS interrogeable par similarité sémantique
- Orchestrer, via LangChain, une chaîne de récupération et de génération de réponses
- Évaluer la qualité des réponses sur un jeu de test annoté

---

## Périmètre du POC

Deux critères, appliqués de façon identique dans le code, les tests et le rapport.

### Périmètre géographique

Commune de **Lille** (`location_city = "Lille"`).

Le périmètre est volontairement resserré : le POC vise à démontrer la faisabilité
technique, non l'exhaustivité de la couverture. Le passage à l'échelle (métropole,
département, région) se fait en modifiant une constante du script de collecte.

Volumétrie constatée à la mise en place :

| Périmètre | Événements de moins d'un an |
|---|---|
| Région Hauts-de-France | ~28 000 |
| Département du Nord | > 6 000 |
| **Commune de Lille** | **3 642** |
| Lille + 10 km | ~6 300 |
| Lille + 15 km | ~8 900 |

Le choix de Lille place le volume dans une fenêtre compatible avec les quotas
d'embedding de l'API Mistral, tout en laissant assez de matière pour que la
recherche sémantique soit pertinente.

### Périmètre temporel

Sont retenus les événements dont **la date de fin de la dernière occurrence est
postérieure à la date du jour moins un an** :

```
lastdate_end >= now(years=-1)
```

Ce critère conserve les événements récemment terminés, ceux en cours, et ceux à
venir — ce qui correspond au besoin exprimé (un an d'historique et événements à
venir). La date de début n'entre pas dans le filtre : un événement de longue durée
reste pertinent tant qu'il ne s'est pas achevé il y a plus d'un an.

Le seuil est calculé **côté serveur** par la fonction ODSQL `now(years=-1)`, évaluée
au moment de la requête. Aucune date n'est écrite en dur dans le code : le filtre
reste correct quelle que soit la date d'exécution.

**Conséquence :** le volume de données récupérées varie selon le jour d'exécution.
Les tests unitaires vérifient donc le *critère*, jamais un nombre d'événements figé.

---

## Source de données

Jeu de données [`evenements-publics-openagenda`](https://public.opendatasoft.com/explore/dataset/evenements-publics-openagenda)
publié sur la plateforme OpenDataSoft (~1,18 million d'enregistrements au niveau national).

L'accès se fait par l'API Explore v2.1 d'OpenDataSoft, sans authentification.
Le filtrage géographique et temporel est appliqué côté serveur (langage ODSQL) :
seuls les enregistrements du périmètre sont téléchargés. La collecte passe par
l'endpoint `/exports/json`, non soumis à la limite de 10 000 enregistrements de
l'endpoint `/records`.

---

## Stack technique

| Composant | Choix |
|---|---|
| Langage | Python 3 |
| Embeddings | Mistral AI — `mistral-embed` (dimension 1024) |
| Génération | Mistral AI — modèle de chat |
| Base vectorielle | FAISS (`faiss-cpu`) |
| Orchestration | LangChain (`langchain-community` **épinglé**, voir *Dette technique*) |
| Manipulation de données | pandas, numpy |
| Nettoyage HTML | BeautifulSoup (`beautifulsoup4`) |
| Tests | pytest |

---

## Installation

### Prérequis

- Python 3.10, 3.11 ou 3.12
- Une clé API Mistral ([console.mistral.ai](https://console.mistral.ai))

### Mise en place

```powershell
git clone <url-du-repo>
cd puls-events-rag

python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows PowerShell
# source .venv/bin/activate       # macOS / Linux

python -m pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` liste les dépendances directes : il se lit d'un coup d'œil.
Les versions n'y sont pas figées, à une exception près (`langchain-community`,
voir *Dette technique*). `requirements-lock.txt`, généré par `pip freeze`, fige
l'arbre complet des dépendances et garantit une installation strictement
reproductible.

### Configuration

Copier `.env.example` vers `.env` et y renseigner la clé API :

```
MISTRAL_API_KEY=votre_cle
```

Le fichier `.env` est ignoré par git et ne doit jamais être versionné.

### Vérification de l'environnement

```powershell
python scripts\check_env.py
```

Le script doit afficher une dimension d'embedding de **1024**, confirmant que
FAISS, LangChain et la clé Mistral sont opérationnels.

---

## Pipeline

### 1. Collecte

```powershell
python src\fetch_data.py
```

Interroge l'API OpenDataSoft dans le périmètre défini et écrit le résultat brut
dans `data/raw/events_lille.json` (**3 642 événements**). Les dossiers manquants
sont créés automatiquement.

**`data/raw/` ne doit jamais être modifié à la main.** Ce dossier est l'image
fidèle de la réponse de l'API ; toute transformation passe par un script versionné.

### 2. Audit qualité

```powershell
python scripts\inspect_data.py    # volumétrie, complétude, doublons, longueurs
python scripts\diagnose.py        # élucidation des anomalies repérées
```

Ces scripts ne modifient rien. Ils documentent l'état du jeu brut et justifient
les règles de nettoyage appliquées ensuite.

### 3. Nettoyage

```powershell
python src\preprocess.py
```

Produit `data/processed/events_lille_clean.json` : **3 641 événements** nettoyés,
dotés d'un champ `texte` prêt à être vectorisé.

### 4. Vectorisation et indexation

```powershell
python src\vectorize.py           # reprend le cache d'embeddings s'il existe
python src\vectorize.py --force   # vide le cache et revectorise intégralement
```

Découpe le champ `texte` en chunks, attache les métadonnées, vectorise via
`mistral-embed` et construit l'index FAISS dans `faiss_index/`.
Produit **4 527 vecteurs** pour 3 641 événements.

Un `manifest.json` est écrit à côté de l'index : modèle d'embedding, dimension,
paramètres de découpage, métrique, volumétrie, horodatage. Il rend l'index
auto-descriptif et permet de vérifier a posteriori avec quels réglages il a été
construit.

**Reprise sur erreur.** Chaque batch vectorisé est mis en cache sur disque
(`data/processed/emb_cache/`). Une interruption — rate limit de l'API, coupure
réseau — ne fait pas repartir de zéro : relancer le script reprend là où il s'était
arrêté. C'est aussi ce qui permet de reconstruire l'index en quelques secondes
lorsque seuls les paramètres d'indexation changent, sans repayer d'appels API.

### 5. Vérification de la récupération

```powershell
python scripts\inspect_search.py                               # jeu de requêtes par défaut
python scripts\inspect_search.py "une exposition de photo"     # requête libre
python scripts\inspect_search.py "un concert de jazz" --k 10  # top-10
```

Affiche les événements les plus proches d'une requête, avec leur score de
similarité, leurs métadonnées et les métriques d'exécution (latence d'embedding,
latence FAISS). Ce script ne teste pas la chaîne RAG : il isole la
**récupération**. Si elle est mauvaise, aucune qualité de génération ne la
rattrapera — c'est le garde-fou avant de brancher le LLM.

### 6. Chaîne RAG (génération de réponses)

```powershell
python -c "from src.rag_chain import repondre; r = repondre('un concert de jazz'); print(r.texte)"
```

La chaîne complète : récupération (via `src/retriever.py`) → construction du
contexte → génération (`mistral-small-2506`, température 0.2). Le montage est
**explicite** (pas de `RetrievalQA` de LangChain) : chaque étape est un appel
lisible, instrumenté (latences, tokens).

La réponse est encapsulée dans une `ReponseRAG` qui porte le texte, les
événements sources, les uids récupérés et les métriques d'exécution. La réponse
ne circule jamais sans sa traçabilité.

**Prompt d'abstention.** Le système ne doit jamais inventer. Le prompt a été
durci en trois itérations après observation de hallucinations concrètes :

1. **V1** (naïve) : « ne jamais inventer » — le modèle a répondu « l'entrée est
   gratuite sur réservation » alors que l'information n'existait pas, et a
   fusionné deux événements distincts.
2. **V2** : interdiction en tête, fusion nommément interdite, prix interdit en
   bloc — le « gratuite » disparaît, mais le prix est alors interdit même quand
   il est explicitement écrit dans la source.
3. **V3** (finale) : distinction **citer / déduire**. Le modèle peut rapporter
   une gratuité explicitement écrite ; il ne peut pas la déduire quand elle
   manque. Dates reportées telles qu'écrites, sans année ajoutée.

### 7. Évaluation

En deux temps, découplés :

```powershell
python scripts\generate_answers.py         # génère les réponses (appels API, une seule fois)
python scripts\evaluate_rag.py             # calcule les métriques (local, gratuit, instantané)
python scripts\evaluate_rag.py --avec-juge # ajoute fidélité et pertinence (LLM-as-judge)
```

`generate_answers.py` lance la chaîne RAG sur les 12 questions du jeu de test
annoté et sauvegarde les réponses dans `data/eval/rag_answers.json`. Les réponses
sont figées : l'évaluation porte sur un artefact stable et reproductible.

`evaluate_rag.py` calcule trois familles de métriques :

- **Récupération** : `recall@k` et `precision@k` sur les identifiants
  d'événements (`uid`). Implémentées ici (pas RAGAS) parce que la structure est
  événementielle.
- **Abstention** : sur les questions-pièges, le système reconnaît-il qu'il ne
  dispose pas de l'information ? Détection lexicale, relisible à la main.
- **Génération** (optionnel, `--avec-juge`) : `faithfulness` et
  `answer_relevancy` via LLM-as-judge (`mistral-small-2506`).

### 8. Tests unitaires

```powershell
pytest tests\test_pipeline.py -v
```

13 tests, aucun appel réseau, exécution en 0,24 s :

- **Périmètre géographique** — tous les événements sont à Lille
- **Périmètre temporel** — tous les événements postérieurs au seuil du manifeste
  de collecte (date fixe, pas `datetime.now()`)
- **Intégrité du jeu de test** — chaque uid de `qa_dataset.json` existe dans le
  jeu figé, types valides, ground_truth présents, pièges à uids vides
- **Cohérence de l'index** — volumétrie conforme au manifest

---

## Constats de l'audit qualité

| Constat | Décision |
|---|---|
| `longdescription_fr` contient du HTML brut (`<p>`, `<br>`, entités) | Nettoyage par BeautifulSoup |
| 1 événement sans titre ni description (`uid` 70189310) | **Supprimé** — invectorisable |
| 153 événements sans `longdescription_fr` (95,8 % de remplissage) | **Conservés** — titre et résumé suffisent |
| 929 titres dupliqués, dont « Mai à Vélo 2026 » ×114 | **Conservés** — `uid` uniques, lieux et descriptions distincts : événements réels d'une même campagne, non des doublons |
| 0 `uid` dupliqué | Aucune action ; dédoublonnage de sécurité conservé dans le code |
| 16 événements sans `location_region` | **Conservés** — adresses vérifiées, toutes à Lille (59000/59777). Champ vide à la source, sans incidence sur le périmètre |
| `keywords_fr` rempli à 34 % seulement | **Écarté** de la vectorisation — trop lacunaire |
| Caractères Unicode `U+2028`/`U+2029` (copier-coller depuis un traitement de texte) | Absorbés par la normalisation des espaces (`re.sub(r"\s+", " ", …)`) |
| 1 date aberrante (« Auberge de jeunesse HI Lille », 01/01/2032) | **Conservée** — cas isolé, ne justifie pas une règle |

### Choix du nettoyage HTML

BeautifulSoup est préféré à une expression régulière pour trois raisons :

- **décodage des entités** : `&eacute;` → `é`, `&amp;` → `&` — une regex les laisserait telles quelles dans l'index ;
- **séparation des blocs** : `<p>Concert</p><p>Théâtre</p>` donnerait `ConcertThéâtre` avec une regex naïve ;
- **tolérance au HTML mal formé**, inévitable sur des fiches saisies par des centaines de contributeurs.

### Limite identifiée du jeu de données

Le dataset s'intitule « événements **publics** », non « événements culturels ».
Il contient de fait un volume notable d'annonces d'emploi et de formation
(« Recrutement garde d'enfants » ×32, « Jobdating Conseiller Clientèle » ×11,
« Formation Conducteur de bus » ×10…).

Aucun champ ne permet de les écarter de façon fiable : `category` est vide,
`keywords_fr` n'est rempli qu'à 34 %. Ces événements sont donc **conservés** dans
le POC. Une classification thématique — par LLM ou modèle supervisé — serait
nécessaire en production ; ce point figure parmi les recommandations du rapport
technique.

---

## Résultat du pré-processing

| Indicateur | Valeur |
|---|---|
| Événements bruts | 3 642 |
| Supprimés (aucun texte exploitable) | 1 |
| **Événements retenus** | **3 641** |
| Longueur médiane du champ `texte` | 716 caractères |
| Longueur maximale | 9 606 caractères |

L'écart entre médiane et maximum motive la stratégie de découpage : la majorité
des événements tient dans un seul chunk, une minorité demande un découpage.

---

## Vectorisation et index FAISS

Quatre décisions structurent cette étape. Chacune répond à un défaut constaté, non
à une convention.

### Découpage calibré, non par défaut

Un premier découpage à 1000 / 150 caractères produisait 1,58 chunk par événement —
incohérent avec une médiane de 716 caractères, et le signe que des fiches courtes
étaient coupées inutilement. Le calibrage à 1500 / 200 :

| Indicateur | 1000 / 150 | **1500 / 200** |
|---|---|---|
| Chunks produits | 5 769 | **4 527** |
| Chunks par événement | 1,58 | **1,24** |
| Chunks issus d'événements découpés | 3 145 (55 %) | **1 344 (30 %)** |

Le troisième indicateur est le plus parlant : on passe d'un index majoritairement
composé de fragments à un index composé à 70 % de fiches entières. Effet
secondaire non négligeable : 1 242 vecteurs de moins, donc autant d'appels
d'embedding économisés à chaque reconstruction.

### En-tête de contexte sur chaque chunk

Chaque chunk est préfixé d'une ligne `titre — dates — lieu`.

Sans cela, les chunks 2..n d'un événement long perdent toute référence à
l'événement dont ils proviennent : ils deviennent des orphelins, introuvables sur
une requête portant sur le titre, la date ou le lieu. Le coût est une légère
redondance sur le premier chunk ; le bénéfice est qu'aucun fragment n'est perdu
pour la recherche.

### Similarité cosinus, garantie par construction

Les vecteurs sont **normalisés explicitement** (L2) avant indexation, et l'index
utilise le produit scalaire (`MAX_INNER_PRODUCT`) — produit scalaire sur vecteurs
unitaires = similarité cosinus.

La normalisation n'est pas déléguée à LangChain : son paramètre `normalize_L2` est
silencieusement ignoré lorsque la métrique est `MAX_INNER_PRODUCT`. Elle n'est pas
non plus déléguée au modèle : `mistral-embed` renvoie aujourd'hui des vecteurs déjà
unitaires (norme vérifiée à 1,000), mais ce comportement n'est garanti par aucun
contrat d'API. La correction de l'index ne doit pas reposer sur une propriété non
documentée d'un service tiers.

Le choix du cosinus plutôt que de la distance euclidienne tient à l'hétérogénéité
des textes (716 à 9 606 caractères) : une distance L2 brute pénaliserait les textes
longs indépendamment de leur pertinence.

### Déduplication par `uid` à la récupération

FAISS ne connaît pas la notion d'événement : il renvoie les *k* chunks les plus
proches. Or les chunks d'un même événement se ressemblent fortement et ressortent
donc **groupés**. Sans traitement, un top-5 pouvait ne contenir que 3 ou 4
événements distincts — les places restantes étant occupées par des doublons, au
détriment d'événements différents et au prix d'un contexte redondant envoyé au LLM.

La récupération sur-échantillonne (`fetch_k = 20`), ne conserve que le meilleur
chunk de chaque `uid`, puis tronque à *k*. Les résultats de FAISS étant déjà triés,
le premier chunk rencontré pour un `uid` est nécessairement son meilleur.

### Index plat, assumé

À cette volumétrie (4 527 vecteurs), un `IndexFlatIP` exhaustif est à la fois exact
et instantané. Un index approximatif (IVF, HNSW) n'apporterait aucun gain mesurable
et introduirait une perte de rappel. Le choix se reposerait à partir de quelques
centaines de milliers de vecteurs.

---

## Résultats d'évaluation

| Dimension | Score | Interprétation |
|---|---|---|
| Factuelles recall@5 | **1,00** (4/4) | Le bon événement est toujours retrouvé |
| Pièges abstention | **3/3** (100 %) | Zéro invention (durée, PMR, baseball) |
| Faithfulness | **0,986** | Quasi aucune hallucination (LLM-as-judge) |
| Answer relevancy | **0,933** | Les réponses répondent aux questions |
| Thématiques recall | 0,24 | Moyenne trompeuse — voir analyse ci-dessous |
| Temporelles recall | 0,29 | Limite confirmée (pas de filtrage par date) |

### Analyse par typologie

**Factuelles (recall 1,00).** Quand la question nomme un événement, le système le
retrouve systématiquement. Les noms propres sont des signaux forts pour les
embeddings.

**Pièges (abstention 3/3).** Trois modes d'abstention testés et réussis :
attribut absent (durée du concert de Voulzy), attribut absent sur un lieu (PMR
de Malika Aït Gherbi), événement absent avec contexte trompeur (match de baseball
→ le système reçoit palet breton, Urban Game, « À vous de jouer » et les rejette).

**Thématiques (0,24 en moyenne, mais trompeur).** theme_01 (expositions) à 0,60 ;
theme_02 (rock en août) à 0,00 — échec temporel, pas thématique ; theme_03
(metal) à 0,12 — le système ne discrimine pas les sous-genres musicaux.

**Temporelles (0,29).** Démontré sur « concerts de rock en août 2026 » : 5 concerts
de rock ramenés (juin, octobre, novembre, février) — aucun d'août. Le système
capte le thème et ignore la date. La recherche est purement sémantique ; le
filtrage par métadonnées temporelles est la recommandation principale pour le MVP.

### Limites identifiées

**Pas de filtrage temporel.** La recherche est purement sémantique. « Août 2026 »
pèse quasi-zéro dans la requête. Solution MVP : recherche hybride (filtre par
métadonnées de date + sémantique). Les métadonnées nécessaires (`firstdate_begin`,
`lastdate_end`) sont déjà dans l'index — c'est le mécanisme de requête qui manque.

**Pas de discrimination fine des sous-genres.** « Metal » et « pop rock » se
confondent sémantiquement (recall 1/8). Enrichissement taxonomique nécessaire.

**Qualité des données source.** Titres trompeurs (« Théâtre de rues » = atelier
d'arts plastiques pour enfants), champs vides (`keywords_fr` à 34 %),
incohérences internes (jeudi/vendredi sur un même événement). Inhérent au
crowdsourcing Open Agenda.

**Date du jour absente du prompt.** Le modèle ne sait pas si un événement est passé
ou à venir. Correctif simple (injection de la date dans le prompt), identifié mais
non implémenté dans le POC.

**Recall thématique non mesurable exhaustivement.** 275 expositions dans le corpus :
impossible d'annoter exhaustivement. Le recall@k sur les thématiques denses est un
recall sur cible d'ancrage, pas un recall absolu. Complété par les métriques
sémantiques du LLM-as-judge.

**k=5 insuffisant pour les thèmes denses.** Certaines requêtes thématiques (visites
patrimoine) ont 15+ événements pertinents ; le recall plafonne mécaniquement à
5/15 même si le système fait un travail parfait. Recommandation MVP : k adaptatif
ou pagination.

**Absence d'information tarifaire structurée.** Le jeu Open Agenda ne comporte
aucun champ prix. 23 % des événements mentionnent un tarif dans du texte libre
(« entrée libre », « 8 € »), 77 % n'ont aucune information. Le prompt V3 autorise
la citation d'un prix explicitement écrit et interdit la déduction — vérifié sur
la paire fact_04 (gratuité écrite → citée) / piege_01 (durée absente → abstention).

**Pas de mémoire conversationnelle.** Le POC est mono-tour : chaque question est
traitée indépendamment. La mémoire relève du MVP (mission 13).

---

## Structure du projet

```
puls-events-rag/
├── .env / .env.example
├── README.md
├── requirements.txt / requirements-lock.txt
│
├── data/
│   ├── raw/
│   │   ├── events_lille.json          # données brutes collectées (API Open Agenda)
│   │   └── collecte_manifest.json     # traçabilité de la collecte (date, seuil temporel)
│   ├── processed/
│   │   ├── events_lille_clean.json    # données nettoyées
│   │   └── emb_cache/                 # cache d'embeddings (142 batches .npy)
│   ├── eval/
│   │   ├── qa_dataset.json            # jeu de questions/réponses de référence (12 questions)
│   │   ├── rag_answers.json           # réponses générées par la chaîne RAG (figées)
│   │   └── eval_results.json          # résultats détaillés de l'évaluation
│   └── events_lille_soutenance.json   # jeu de données figé pour la soutenance (versionné)
│
├── faiss_index/
│   ├── index.faiss                    # index vectoriel FAISS
│   ├── index.pkl
│   └── manifest.json                  # paramètres de construction de l'index
│
├── src/
│   ├── fetch_data.py                  # collecte des données (API) + manifeste de collecte
│   ├── preprocess.py                  # nettoyage / audit qualité
│   ├── vectorize.py                   # découpage + construction de l'index FAISS
│   ├── retriever.py                   # récupération sémantique (FAISS + déduplication)
│   └── rag_chain.py                   # chaîne RAG complète (retriever + prompt + LLM)
│
├── scripts/
│   ├── check_env.py                   # test de fumée de l'environnement
│   ├── count_events.py                # comptage par périmètre géographique
│   ├── diagnose.py                    # élucidation des anomalies
│   ├── explore_api.py                 # inspection du schéma du jeu de données
│   ├── inspect_data.py                # audit qualité du jeu brut
│   ├── inspect_search.py              # inspection manuelle du retriever
│   ├── generate_answers.py            # génère les réponses RAG sur le dataset d'éval
│   └── evaluate_rag.py                # évalue les réponses générées
│
├── tests/
│   └── test_pipeline.py               # 13 tests unitaires (géo, temporel, intégrité, index)
│
└── notebooks/
```

`data/raw/`, `data/processed/` et `faiss_index/` sont ignorés par git : leur
contenu est volumineux et intégralement régénérable par les scripts. `data/eval/`
est versionné : il contient le jeu de test annoté et les résultats de référence.

---

## Champs exploités

| Champ | Usage |
|---|---|
| `uid` | identifiant unique — dédoublonnage, déduplication à la récupération, appariement avec le jeu de test annoté |
| `title_fr` | titre — vectorisé, en-tête de chunk, métadonnée |
| `description_fr` | résumé court — vectorisé |
| `longdescription_fr` | description longue (HTML) — vectorisée après nettoyage |
| **`texte`** | **champ construit : titre + résumé + description longue — support de la vectorisation** |
| `daterange_fr` | libellé de dates lisible — en-tête de chunk, métadonnée |
| `firstdate_begin`, `lastdate_end` | dates — filtrage temporel et métadonnées |
| `location_name`, `location_address`, `location_city` | lieu — en-tête de chunk, métadonnées |
| `location_coordinates` | coordonnées — métadonnées, filtrage par rayon |
| `canonicalurl` | lien vers la fiche d'origine — métadonnée |

---

## Reproductibilité et stabilité des données

Le jeu de données Open Agenda est vivant : des événements y sont ajoutés, modifiés
et supprimés en continu. Combiné au seuil temporel glissant (`now(years=-1)`), cela
implique que **deux exécutions du pipeline à des dates différentes ne produisent pas
un index identique**. C'est le comportement attendu d'un système connecté à une
source de production, non un défaut.

Le jeu collecté est figé dans `data/events_lille_soutenance.json`, versionné : c'est
lui qui sert de référence à l'index FAISS, au jeu de test annoté et à la
démonstration. Ces trois éléments sont solidaires et ne doivent pas être
désynchronisés par une nouvelle collecte.

**Note sur les scores de similarité.** Le vecteur d'une requête est recalculé par
l'API à chaque exécution, et `mistral-embed` ne renvoie pas des flottants
strictement identiques d'un appel à l'autre. Les scores absolus varient donc au
millième près, de façon uniforme sur l'ensemble des résultats : **le classement,
lui, est stable**. Les métriques d'évaluation portent sur l'ordre, jamais sur la
valeur absolue d'un score.

---

## Dette technique

**`langchain-community` a été sunsetté par LangChain**, au profit d'un modèle
« un fournisseur = un package » (`langchain-openai`, `langchain-chroma`…). FAISS,
n'ayant pas d'éditeur commercial derrière lui, ne dispose à ce jour d'aucun package
d'intégration officiel : `langchain-community` reste la seule voie d'accès
maintenue par LangChain à l'intégration FAISS.

Le POC y reste donc, mais avec la version **épinglée** dans `requirements.txt` :

```
langchain-community==0.4.2
```

C'est la seule dépendance épinglée du projet. La raison est explicite : un package
non maintenu ne doit plus voir son comportement évoluer sous le projet.
Le `DeprecationWarning` affiché à l'exécution n'est volontairement pas masqué — il
documente une réalité de la stack.

*(Un package `langchain-faiss` existe sur PyPI ; il s'agit d'une publication tierce
non officielle, sans description ni maintenance depuis octobre 2024. Il n'est pas
utilisé.)*

**Recommandation pour la version finale.** Deux voies : appeler `faiss` directement
derrière une interface interne — seules `save_local` / `load_local` et la recherche
par similarité sont utilisées, soit une trentaine de lignes — ou basculer vers une
base vectorielle disposant d'un package maintenu (Chroma, Qdrant, pgvector). La
seconde s'impose dès que l'index doit être partagé entre plusieurs processus ou
machines, ce que FAISS ne permet pas.


---

## État d'avancement

- [x] **Étape 1** — Environnement de travail : venv, dépendances, FAISS opérationnel,
      clé Mistral validée, structure du dépôt, protection des secrets
- [x] **Étape 2** — Pré-processing des données Open Agenda
  - [x] Exploration du schéma du jeu de données
  - [x] Choix et validation du périmètre géographique et temporel
  - [x] Collecte des données brutes (3 642 événements)
  - [x] Audit qualité chiffré
  - [x] Nettoyage HTML, suppression des textes vides, dédoublonnage
  - [x] Structuration du jeu final (3 641 événements, champ `texte`)
  - [x] Manifeste de collecte (`collecte_manifest.json`)
- [x] **Étape 3** — Découpage en chunks, vectorisation et index FAISS
  - [x] Découpage calibré sur la distribution réelle des textes (1500 / 200)
  - [x] En-tête de contexte sur chaque chunk
  - [x] Métadonnées attachées, `uid` inclus
  - [x] Vectorisation par batchs, avec cache disque et backoff exponentiel
  - [x] Index FAISS cosinus (4 527 vecteurs) + `manifest.json`
  - [x] Déduplication par `uid` à la récupération
  - [x] Vérification de la récupération sur jeu de requêtes
- [x] **Étape 4** — Chaîne RAG
  - [x] Module de récupération partagé (`src/retriever.py`) avec instrumentation
  - [x] Chaîne RAG explicite (`src/rag_chain.py`) — récupération → contexte → génération
  - [x] Prompt d'abstention en 3 itérations (V1 naïve → V2 stricte → V3 citer/déduire)
  - [x] Dataclasses `Recuperation` et `ReponseRAG` avec métriques embarquées
- [x] **Évaluation**
  - [x] Jeu de test annoté (12 questions, 4 typologies : factuelle, thématique, temporelle, piège)
  - [x] Script de génération des réponses (`generate_answers.py`) — découplé
  - [x] Script d'évaluation (`evaluate_rag.py`) — recall@k, precision@k, abstention + LLM-as-judge
  - [x] Résultats : recall factuel 1,00, abstention 3/3, faithfulness 0,986
- [x] **Tests unitaires**
  - [x] 13 tests pytest (périmètre géo, temporel, intégrité jeu de test, cohérence index)
  - [x] Exécution instantanée (0,24 s), sans appel réseau
- [ ] Rapport technique
- [ ] PowerPoint + démo live
