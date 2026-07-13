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
| Orchestration | LangChain |
| Manipulation de données | pandas |
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

`requirements.txt` liste les dépendances directes, sans versions : il se lit d'un
coup d'œil. `requirements-lock.txt`, généré par `pip freeze`, fige l'arbre complet
des dépendances et garantit une installation strictement reproductible.

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

## Pipeline de données

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

L'écart entre médiane et maximum motive la stratégie de découpage en chunks de
l'étape 3 : la majorité des événements tient dans un seul chunk, une minorité
demande un découpage.

---

## Structure du projet

```
puls-events-rag/
├── .env                    # clé API Mistral (non versionné)
├── .env.example            # modèle de configuration
├── .gitignore
├── README.md
├── requirements.txt        # dépendances directes
├── requirements-lock.txt   # arbre complet des dépendances (pip freeze)
├── data/
│   ├── events_lille_soutenance.json   # jeu de référence figé (versionné)
│   ├── raw/                # données brutes issues de l'API (non versionné)
│   └── processed/          # données nettoyées, prêtes à vectoriser (non versionné)
├── notebooks/              # exploration
├── scripts/
│   ├── check_env.py        # test de fumée de l'environnement
│   ├── explore_api.py      # inspection du schéma du jeu de données
│   ├── count_events.py     # comptage par périmètre géographique
│   ├── inspect_data.py     # audit qualité du jeu brut
│   └── diagnose.py         # élucidation des anomalies
├── src/
│   ├── __init__.py
│   ├── fetch_data.py       # collecte des données Open Agenda
│   └── preprocess.py       # nettoyage et structuration
└── tests/
    └── __init__.py
```

`data/raw/` et `data/processed/` sont ignorés par git : leur contenu est
volumineux et intégralement régénérable par les scripts.

---

## Champs exploités

| Champ | Usage |
|---|---|
| `uid` | identifiant unique, dédoublonnage |
| `title_fr` | titre — vectorisé |
| `description_fr` | résumé court — vectorisé |
| `longdescription_fr` | description longue (HTML) — vectorisée après nettoyage |
| **`texte`** | **champ construit : titre + résumé + description longue — support de la vectorisation** |
| `daterange_fr` | libellé de dates lisible — métadonnée |
| `firstdate_begin`, `lastdate_end` | dates — filtrage temporel et métadonnées |
| `location_name`, `location_address`, `location_city` | lieu — métadonnées |
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
- [ ] **Étape 3** — Découpage en chunks, vectorisation et index FAISS
- [ ] **Étape 4** — Chaîne RAG LangChain + Mistral
- [ ] Jeu de test annoté et évaluation
- [ ] Tests unitaires
- [ ] Rapport technique
