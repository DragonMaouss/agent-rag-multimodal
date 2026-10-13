import os
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import List, Optional
from src.rag_multimodal import process_and_index_file, get_rag_chain


load_dotenv()

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

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceInfo]

@app.get("/")
def read_root():
    return {"message": "Bienvenue sur l'API de l'Agent RAG Multimodal. Utilisez le point de terminaison /query pour poser vos questions."}


@app.post("/upload")
async def upload_pdf(file : UploadFile = File(...)) :
    try :
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        contents = await file.read()
        with open(file_path, "wb") as f :
            f.write(contents)

        num_docs = process_and_index_file(file_path)
        return {"filename" : file.filename, "message" : f"Fichier indexé avec succès ({num_docs} éléments extraits)."}
    except Exception as e :
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    try : 
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="La requête ne peut pas être vide.")

        # Exécution de la chaîne RAG
        response = rag_chain.invoke({"input": request.query})

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