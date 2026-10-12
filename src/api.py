import os
import io
import base64
from PIL import Image
import pymupdf
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from dotenv import load_dotenv
from src.rag_multimodal import process_and_index_file, get_rag_chain

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter


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

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    question: str
    answer: str

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

        response = rag_chain.invoke({"input": request.query})

        return {
            "question": request.query,
            "answer": response["answer"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))