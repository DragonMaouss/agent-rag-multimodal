# agent-rag-multimodal

Application RAG (Retrieval-Augmented Generation) multimodale pour l'analyse et l'interrogation de documents complexes (PDF, schémas, tableaux, Markdown).


## 1. Objectif du projet

Permettre à l'utilisateur d'interroger une base de connaissances complexe contenant du texte, des structures tabulaires et des éléments visuels (images, schélas).

La première version se concentre sur un parcours simple : 

1. charger un document (PDF ou Markdown) ;
2. extraire et indexer le texte et les éléments visuels ;
3. poser une question en langage naturel ;
4. obtenir une réponse précise générée par un LLM avec sources associées ;
5. consulter l'historique et la progression des échanges.


## 2. Roadmap

### Etape 1 - Base du projet et RAG textuel

- initialiser l'environnement et la structure du projet ;
- charger un fichier PDF/Markdown simple ;
- découper le texte en chunks ;
- générer des embeddings et les stoker dans une base vectorielle locale (ChromaDB) ;
- exécuter une première requête de réponse textuelle avec un LLM.

### Etape 2 - Prise en charge multimodale

- intégrer l'extraction d'images et de schémas depuis les PDF ;
- générer des description textuelles d'image via un modèle Vision ;
- migrer vers Qdrant pour la gestion des vecteurs complexes ;
- lier les explications visuelles aux réponses RAG.

### Etape 3 - API Backend

- créer l'application FastAPI ;
- définir l'endpoint `/upload` pour l'ingestion de documents ;
- définir l'endpoint `/quesry` pour le traitement des questions.

### Etape 4 - Interface Utilisateur 

- créer l'interface web interactive avec Streamlit ;
- ajouter le composant de chat et l'upload de fichiers ;
- afficher les sources et aperçus d'images dans l'interface.

### Etape 5 - Optimisation et Conteneurisation

- dockeriser l'application ;
- optimiser les temps de réponse et la précision de recherche.


## 3. Architecture prévue

```text
Interface Stramlit
    ↓
API FastAPI
    ↓
Framework RAG (LangChain / LlamaIndex)
    ↓
Base Vectorielle (Qdrant / ChromaDB) + Modèle LLM / Vision
```

## 4. Stack technique envisagée

- Python ;
- LangChain / LlamaIndex ;
- Qdrant / ChromaDB ;
- FastAPI ;
- StreamLit ;
- Docker.