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
| **Commune de Lille** | **~3 640** |
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
Le filtrage géographique et temporel est appliqué côté serveur : seuls les
enregistrements du périmètre sont téléchargés.

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

`requirements.txt` liste les dépendances directes. `requirements-lock.txt`,
généré par `pip freeze`, fige l'arbre complet des dépendances et garantit une
installation strictement reproductible.

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

## Utilisation

### Collecte des données

```powershell
python src\fetch_data.py
```

Interroge l'API OpenDataSoft dans le périmètre défini et écrit le résultat brut
dans `data/raw/events_lille.json`. Le dossier est créé automatiquement s'il
n'existe pas.

### Inspection du jeu brut

```powershell
python scripts\inspect_data.py
```

Affiche la volumétrie, le taux de remplissage des champs, les doublons et la
longueur des textes. Sert à documenter les problèmes de qualité avant nettoyage.

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
│   ├── raw/                # données brutes issues de l'API (non versionné)
│   └── processed/          # données nettoyées, prêtes à vectoriser (non versionné)
├── notebooks/              # exploration
├── scripts/
│   ├── check_env.py        # test de fumée de l'environnement
│   ├── explore_api.py      # inspection du schéma du jeu de données
│   ├── count_events.py     # comptage par périmètre géographique
│   └── inspect_data.py     # audit qualité du jeu brut
├── src/
│   ├── __init__.py
│   └── fetch_data.py       # collecte des données Open Agenda
└── tests/
    └── __init__.py
```

Les dossiers `data/raw/` et `data/processed/` sont ignorés par git : leur contenu
est volumineux et intégralement régénérable par les scripts.

---

## Champs exploités

| Champ | Usage |
|---|---|
| `uid` | identifiant unique, dédoublonnage |
| `title_fr` | titre de l'événement — vectorisé |
| `description_fr` | résumé court — vectorisé |
| `longdescription_fr` | description longue (contient du HTML) — vectorisé après nettoyage |
| `daterange_fr` | libellé de dates lisible — métadonnée |
| `firstdate_begin`, `lastdate_end` | dates — filtrage temporel et métadonnées |
| `location_name`, `location_address`, `location_city` | lieu — métadonnées |
| `location_coordinates` | coordonnées — métadonnées, filtrage par rayon |
| `canonicalurl` | lien vers la fiche d'origine — métadonnée |

---

## Qualité des données — points identifiés

- **`longdescription_fr` contient du HTML brut** (balises, entités). Un nettoyage
  est nécessaire avant vectorisation.
- **De nombreux champs sont vides** (`keywords_fr`, `category`, `conditions_fr`…).
  Le taux de remplissage réel est mesuré par `scripts/inspect_data.py`.
- La présence d'une valeur non nulle dans `description_fr` ne garantit pas qu'elle
  soit exploitable : le nettoyage porte sur le contenu, pas sur la nullité.

---

## Reproductibilité et stabilité des données

Le jeu de données Open Agenda est vivant : des événements y sont ajoutés, modifiés
et supprimés en continu. Combiné au seuil temporel glissant (`now(years=-1)`), cela
implique que **deux exécutions du pipeline à des dates différentes ne produisent pas
un index identique**. C'est le comportement attendu d'un système connecté à une
source de production.

Les données collectées à un instant donné sont en revanche figées dans
`data/raw/`, et l'index FAISS qui en dérive l'est également.

---

## État d'avancement

- [x] **Étape 1** — Environnement de travail : venv, dépendances, FAISS opérationnel,
      clé Mistral validée, structure du dépôt, protection des secrets
- [ ] **Étape 2** — Pré-processing des données Open Agenda *(en cours)*
  - [x] Exploration du schéma du jeu de données
  - [x] Choix et validation du périmètre géographique et temporel
  - [x] Collecte des données brutes
  - [ ] Nettoyage (HTML, textes vides, doublons)
  - [ ] Structuration du jeu final
- [ ] **Étape 3** — Vectorisation et index FAISS
- [ ] **Étape 4** — Chaîne RAG LangChain + Mistral
- [ ] Jeu de test annoté et évaluation
- [ ] Tests unitaires
- [ ] Rapport technique
