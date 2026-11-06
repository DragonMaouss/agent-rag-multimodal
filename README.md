# Multimodal RAG Agent 

Application RAG (Retrieval-Augmented Generation) multimodale pour l'analyse  l'indexation et l'interrogation de documents complexes (PDF, Word `.docx`, PowerPoint `.pptx`, Excel `.xlsx`, HTML `.html`, Markdown `.md`).

## Fonctionnalités Clés

- **Multimodalité** : Extraction et traitement au format Markdown via `Docling` pour tous les formats bureautiques et techniques courants.
- **Recherche Hybride** : Couplage de la recherche vectorielle sémantique **Qdrant** et de la recherche textuelle par mots-clés **BM25**.
- **Re-Classement (Re-ranking)** : Filtrage et tri de précision via un modèle Cross-Encoder (`BAAI/bge-reranker-base`) pour ne soumettre au LLM que les extraits les plus pertinents.
- **Chunking Adaptatif Sémantique** : Découpage intelligent du texte basé sur les ruptures de cohérence sémentique (`SemanticChunker`). 
- **Pruning de Contexte** : Elimination du bruit au niveau de la phrase pour réduire l'utilisation de jetons et maximiser la précision des réponses. 
- **Cashe Sémantique Vectoriel et Suive de Fréquence** : Stockage et réutilisation automatique des réponses des questions fréquemment posées. 
- **Réécriture et Contextualisation des Eequêtes** : Reformulation automatique des questions de l'utilisateur en tenant compte de l'historique de conversation pour optimiser la recherche.
- **Observabilité et Monitoring** : Traçabilité de la pipeline RAG via **LangSmith** (`@traceable`). 
- **Evaluation** : Benchmarks pour mesurer les métriques du retrieval (Hit Rate@K, MRR, MAP, NDCG) et de génération (LLM-as-a-Judge). 
- **Architecture** : Backend asynchrone **FastAPI**, interface interactive **Streamlit** et conteneurisation complète avec **Docker Compose**. 


## Roadmap

### Phase Validées (1 à 5)
- [x] Ingestion multiformat & multimodale (PDF, DOCX, PPTX, XLSX, HTML, MD via Docling)
- [x] Recherche Hybride (Qdrant + BM25) & Re-ranking (Cross-Encoder)
- [x] Mémoire conversationnelle & Réécriture de requête
- [x] Découpage sémantique adaptatif & Pruning de contexte
- [x] Cache sémantique vectoriel & Suivi de fréquence des requêtes
- [x] Monitoring, traçabilité et observabilité avec LangSmith
- [x] Benchmark et métriques d'évaluation (*Hit Rate, MRR, MAP, NDCG, LLM-as-a-Judge*)

### Phase 6 : Agentic RAG, Sécurité & Routing (En cours)
- [ ] **Tools & MCP** : Intégration du Tool Calling et connexion au protocole MCP (Model Context Protocol).
- [ ] **Access & RBAC** : Authentification OAuth2 / JWT et filtrage des documents au niveau vecteur Qdrant selon les rôles utilisateurs.
- [ ] **Guardrails** : Anonymisation PII et détection des injections de prompt en entrée.
- [ ] **Intelligent Router** : Routage dynamique des requêtes selon la complexité (Modèle rapide/économique vs Modèle raisonneur avancé).

## Stack technique 

- **Langage** : Python 3.10+
- **RAG** : LangChain, LangChain-Google-GenAI, LangChain-Community
- **Modèles LLM & Embeddings** : Google Gemini (`gemini-3.5-flash-lite`), HuggingFace Embeddings (`sentence-transformers/all-mpnet-base-v2`)
- **Modèle de Re-ranking** : `BAAI/bge-reranker-base`
- **Vector Database** : Qdrant
- **Monitoring** : LangSmith
- **Backend API** : FastAPI, Uvicorn
- **Interface Utilisateur** : Streamlit
- **DevOps** : Docker


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
└── .env                            # Variables d'environnement (Clé API Gemini et LangSmith)
```

## Installation et Lancement

### Option 1 : Lancement via Docker Compose

1. **Configurer les variables d'environnement dans un fichier `.env` à la racine** : 
```text
GOOGLE_API_KEY=votre_cle_api_gemini_ici
QDRANT_URL=http://qdrant:6333
LANGCHAIN_TRACING_V2=true
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_API_KEY=votre_cle_langsmith
LANGCHAIN_PROJECT=rag-multimodal-agent
````

2. **Lancez les conteneurs** :
```bash
docker compose up --build
```

3. **Accédez aux servies** : 
- Interface Steamlit : http://localhost:8501
- Documentation FastAPI : http://localhost:8000/docs
- Tableau de bord Qdrant : http://localhost:6333/dashboard

### Option 2 : lancement en local

1. **Clone le projet et créer un environemment virtuel** :
```bash
python -m venv venv
source venv/bin/activate # Sur Windows : venv\Scripts\activate
```

2. **Installer les dépendaces** :
```bash
pip install -r requirements.txt
```

3. **Lancer Qdrant via Docker** :
```bash
docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
```

4. **Lancer le Backend (Terminal 1)** :
```bash
uvicorn src.api:app --reload --port 8000
```

5. **Lancer le Frontend (Terminal 2)** :
```bash
streamlit run src/app.py --server.port 8501
```

## Évaluation et Benchmarking

Pour exécuter la suite de tests automatique sur le jeu de données de référence (`golden_dataset.json`) :

```bash
# Évaluation des performances du Retrieval (Hit Rate, MRR, MAP, NDCG)
python evaluation/evaluate_retrieval.py

# Évaluation globale des réponses générées (Mots-clés + LLM-as-a-Judge)
python evaluation/evaluate.py
```