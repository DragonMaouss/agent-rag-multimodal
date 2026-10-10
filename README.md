# Multimodal RAG Agent 

Application RAG (Retrieval-Augmented Generation) multimodale pour l'analyse et l'interrogation de documents complexes (PDF, schémas, tableaux, Markdown).

## Fonctionnalités Clés

- Multimodalité : Extraction de texte et d'images intégrées à partir de fichiers PDF via `PyMuPDF`.
- Vision par IA : Analyse automatique et génération de description textuelles détaillés des schémas/images par **Gemini Vision**.
- Indexation vectorielle : Utilisation de **ChromaDB** et d'embeddings.
- Architecture API : Backend asynchrone avec **FastAPI**.
- Interface Web : Application interactive fait avec **Strealit**.
- Conteneurisation : Utilisation de **Docker**


## Stack technique 

- Langage : Python 3.10+
- RAG : LangChain, LangChain-Google-GenAI, LangChain-Chroma,HuggingFaceEmbeddings
- IA et Vision : Google Gemini (Gemini Flash / Gemini Vision)
- Stockage Vectoriel : ChromaDB
- Backend : FastAPI, Uvicorn
- Frontend : Streamlit
- DevOps : Docker


## Structure du projet

```text
agent-rag-multimodal/
│
├── data/                       # Dossier contenant les PDF
│   └── sample_multimodal.pdf
│
├── src/
│   ├── api.py                  # Backend FastAPI
│   ├── app.py                  # Interface utilisateur Streamlit
│   ├── rag_base.py             # Script RAG textuel de base
│   └── rag_multimodal.py       # Pipeline d'extraction et d'indexation multimodale
│   
├── chroma_multimodal_db/       # Base vectorielle ChromaDB 
├── Dockerfile                  # Config image Docker
├── docker-compose.yml          # Orchestration des services (API + Streamlit)
├── requirements.txt            # Dépendances du projet
└── .env                        # Variables d'environnement (Clé API Gemini)

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

