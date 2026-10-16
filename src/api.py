import os
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import List, Optional
from src.rag_multimodal import process_and_index_file, get_rag_chain, rewrite_query
from langchain_core.messages import HumanMessage, AIMessage

# Chargement des variables d'environnement
load_dotenv()

# Initialisation de l'application FastAPI
app = FastAPI(
    title="Agent RAG Multimodal API",
    description="API pour l'Agent RAG Multimodal utilisant FastAPI, LangChain et Qdrant.",
    version="1.0.0",
)

# Dossier pour stocker les fichiers téléchargés
UPLOAD_DIR = "data"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialisation globale du RAG au démarrage de l'application
print("--- Initialisation du RAG Multimodal ---")

rag_chain = get_rag_chain()

print("--- RAG Multimodal initialisé avec succès ---")

class SourceInfo(BaseModel):
    source: str
    page: int
    image_index : Optional[int] = None

class QueryRequest(BaseModel):
    query: str
    chat_history : Optional[List[dict]] = None

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceInfo]


@app.get("/")
def read_root():
    return {"message": "Bienvenue sur l'API de l'Agent RAG Multimodal. Utilisez le point de terminaison /query pour poser vos questions."}


@app.post("/upload")
async def upload_pdf(file : UploadFile = File(...), background_tasks: BackgroundTasks = BackgroundTasks()) :
    """
    Endpoint de telechargement et d'indexation d'un fichier PDF
    """
    try :
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        contents = await file.read()
        with open(file_path, "wb") as f :
            f.write(contents)

        # Traitement asynchrone des documents
        background_tasks.add_task(process_and_index_file, file_path)

        return {"filename" : file.filename, "message" : f"Fichier {file.filename} reçu et en cours d'indexation en arrière-plan."}
    except Exception as e :
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    """
    Endpoint principal permettant de poser une question a l'agent RAG.
    """
    try : 
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="La requête ne peut pas être vide.")

        # Transformation de l'historation en langchain
        langchain_history = []
        if request.chat_history :
            for msg in request.chat_history:
                if msg["role"] == "user":
                    langchain_history.append(HumanMessage(content=msg["content"]))
                else:
                    langchain_history.append(AIMessage(content=msg["content"]))

        
        # Exécution de la chaîne RAG
        optimized_request = rewrite_query(request.query)
        response = rag_chain.invoke({
            "input": optimized_request,
            "chat_history" : langchain_history
        })

        # Extraction des sources
        sources_list = []
        seen = set()

        for doc in response.get("context", []):
            metadata = doc.metadata
            source_file = os.path.basename(metadata.get("source","Inconnu"))
            page_num = metadata.get("page",1)
            img_idx = metadata.get("image_index", None)

            # Clé unique pour éviter les doublons si plusieurs chunks proviennet de la même page/image
            identifier = (source_file, page_num, img_idx)
            if identifier not in seen:
                seen.add(identifier)
                sources_list.append({
                    "source" : source_file,
                    "page" : page_num,
                    "image_index" : img_idx
                })

        return {
            "question": request.query,
            "answer": response["answer"],
            "sources": sources_list
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))