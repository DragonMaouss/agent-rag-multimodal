# Multimodal RAG Agent 

Application RAG (Retrieval-Augmented Generation) multimodale pour l'analyse et l'interrogation de documents complexes (PDF, schémas, tableaux, Markdown).

## Fonctionnalités Clés

- Multimodalité : Extraction et traitement du contenu des documents PDF au format Markdown via `Docling`.
- Recherche Hybride : Couplage de la recherche vectorielle sémantique et de la recherche textuelle par BM25.
- Re-Classement (Re-ranking) : Intégration d'un modèle Cross-Encoder pour affiner, filtrer et trier les documents les plus pertinents avant la soumission au LLM.
- Réécriture et contextualisation des requêtes : Reformulation automatique des questions de l'utilisateur en tenant compte de l'historique de conversation pour optimiser la recherche.
- Mémoire conversationnelle : Gestion d'un historique de dialogue persistant pour traiter les questions suivi.
- Indexation vectorielle de productio : Migration vers **Qdrant** pour le stockage et la recherche vectorielle haure performance.
- Evaluation : Scripts dédiés à l'évaluation de la qualité du retrieval (Hit Rate, MRR, MAP, NDCG) et des réponses générées sur un golden dataset.
- Architecture API : Backend asynchrone avec **FastAPI** avec prise en charge de l'indexation de fichiers en arrière-plan.
- Interface Web : Application interactive fait avec **Strealit** permettant le chat et l'upload de fichiers PDF.
- Conteneurisation : Orchestration via **Docker**.


## Roadmap

### Phase 1, 2 & 3 : Validées
- Multimodalité (Docling)
- Recherche Hybride (BM25 + Vectoriel Qdrant)
- Re-ranking (Cross-Encoder)
- Réécriture de requête & Conservation de Mémoire
- Evaluation (Hit Rate, MRR, MAP, NDCG)

### Phase 4 :
1. Chunking adaptatif et sémantique : Découpage basé sur la cohérance sémantique du texte pour préserver la stucture.
2. Oberservabilité et Monitoring : Traçabilité complète de la pipeline RAG pour mesurer latence, coûts et qualité des réponses.
3. Support multiformat : Prise en chage des fichiers Word (`.docx`), présentations (`.pptx`), tableaurs (`.xlsx`).

### Phase 5 : 
4. Compression du contexte : Réduction du volume de tokens du contexte avant envoi.
5. Mise en cache sémantique : Cache vectoriel pour réutiliser les réponses aux questions similaires.


## Stack technique 

- Langage : Python 3.10+
- RAG : LangChain, LangChain-Google-GenAI, LangChain-Chroma,HuggingFaceEmbeddings
- IA et Vision : Google Gemini (Gemini Flash / Gemini Vision)
- Stockage Vectoriel : Qdrant
- Backend : FastAPI, Uvicorn
- Frontend : Streamlit
- DevOps : Docker


## Structure du projet

```text
agent-rag-multimodal/
│
├── data/                           # Dossier contenant les PDF
│
├── src/
│   ├── api.py                      # Backend FastAPI
│   ├── app.py                      # Interface utilisateur Streamlit
│   └── rag_multimodal.py           # Pipeline d'extraction et d'indexation multimodale
│
├── evaluation/                     # Datasets et scripts d'évaluation
│   ├── golden_dataset.json     
│   ├── evaluation.py               # Evaluation global                  
│   └── evalutation_retrieval.py#   # Evaluation du retrieval
│   
├── Dockerfile                      # Config image Docker
├── docker-compose.yml              # Orchestration des services (API + Streamlit)
├── requirements.txt                # Dépendances du projet
└── .env                            # Variables d'environnement (Clé API Gemini)

## Installation et Lancement

### Option 1 : Lancement via Docker Compose

1. Créer un ficher .env à la racine contenant votre clé API : 
```text
GOOGLE_API_KEY=votre_cle_api_gemini_ici
````

2. Lancez les conteneurs :
```bash
docker compose up --build
```

3. Accédez aux servies : 
- Interface Steamlit : http://localhost:8501
- Documentation FastAPI : http://localhost:8000/docs

### Option 2 : lancement en local

1. Clone le projet et créer un environemment virtuel :
```bash
python -m venv venv
source venv/bin/activate # Sur Windows : venv\Scripts\activate
```

2. Installer les dépendaces :
```bash
pip install -r requirements.txt
````

3. Créer un ficher .env à la racine contenant votre clé API : 
```text
GOOGLE_API_KEY=votre_cle_api_gemini_ici
```

4. Lancer le Backend (Terminal 1) :
```bash
uvicorn src.api:app --reload
```

5. Lancer le Frontend (Terminal 2) :
```bash
streamlit run src/app.py
```

