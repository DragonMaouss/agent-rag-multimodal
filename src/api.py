import os
import logging
from contextlib import asynccontextmanager
from typing import List, Optional
from fastapi import FastAPI, Form, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.params import Depends
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from fastapi.security import OAuth2PasswordRequestForm

from src.config import settings
from src.core.database import QdrantManager
from src.core.indexer import DocumentIndexer
from src.core.retriever import HybridRetrieverBuilder
from src.pipeline import AgenticRAGPipeline
from src.auth import User, create_access_token, get_current_user, verify_password, USERS_DB
from src.guardrails import GuardrailSecurityError

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("rag_api")

os.makedirs(settings.upload_dir, exist_ok=True)

# Container pour le partage propre des composants au sein de l'application
app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("--- Initialisation des composants du RAG ---")
    embedding = HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        encode_kwargs={"normalize_embeddings": True}
    )
    db_manager = QdrantManager(config=settings, embedding_function=embedding)
    indexer = DocumentIndexer(config=settings, embedding_function=embedding, qdrant_client=db_manager.client)
    retriever_builder = HybridRetrieverBuilder(config=settings, indexer=indexer)
    pipeline = AgenticRAGPipeline(config=settings, db_manager=db_manager, indexer=indexer, retriever_builder=retriever_builder)

    app_state["indexer"] = indexer
    app_state["pipeline"] = pipeline
    logger.info("--- Agent RAG Prêt pour Production ---")
    yield
    app_state.clear()

app = FastAPI(
    title="Agent RAG Multimodal API",
    description="API modulaire et scalable pour l'Agent RAG Multimodal.",
    version="2.0.0",
    lifespan=lifespan
)

class SourceInfo(BaseModel):
    source: str
    page: int
    image_index : Optional[int] = None

class QueryRequest(BaseModel):
    query: str
    chat_history : Optional[List[dict]] = None

class ToolCallInfo(BaseModel):
    name: str
    args: dict

class SecurityEnvelope(BaseModel):
    status: str
    detected_pii: List[str]
    guardrails_applied: List[str]

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceInfo]
    tool_calls: Optional[List[ToolCallInfo]] = []
    security_envelope: Optional[SecurityEnvelope] = None

class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    username: str
    roles: List[str]

@app.get("/")
def read_root():
    return {"message": "API Agent RAG Multimodal opérationnelle."}

@app.post("/login", response_model=TokenResponse)
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = USERS_DB.get(form_data.username)
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=401,
            detail="Nom d'utilisateur ou mot de passe incorrect",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(data={"sub": user.username, "roles": user.roles})
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "username": user.username,
        "roles": user.roles
    }

@app.post("/upload")
async def upload_file(file: UploadFile = File(...), access_roles: Optional[str] = Form("public"), background_tasks: BackgroundTasks = BackgroundTasks()):
    try:
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in settings.supported_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Format '{file_ext}' non supporté. Formats valides : {', '.join(settings.supported_extensions)}"
            )
        
        file_path = os.path.join(settings.upload_dir, file.filename)
        contents = await file.read()
        with open(file_path, "wb") as f :
            f.write(contents)

        # Découpage et nettoyage de la liste des rôles d'accès
        roles_list = [r.strip() for r in access_roles.split(",") if r.strip()] if access_roles else ["public"]

        indexer: DocumentIndexer = app_state["indexer"]
        background_tasks.add_task(indexer.index_file, file_path,access_roles=roles_list)
        return {"filename": file.filename, "access_roles": roles_list, "message": f"Fichier {file.filename} reçu et indexation lancée."}
    except Exception as e:
        logger.error("Erreur d'upload : %s", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest, current_user: User = Depends(get_current_user)):
    try:
        pipeline: AgenticRAGPipeline = app_state["pipeline"]
        res = pipeline.run_query(query=request.query, raw_chat_history=request.chat_history, user_roles=current_user.roles)
        return res
    except GuardrailSecurityError as e:
        logger.warning("Violation de sécurité détectée : %s", e)
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        logger.error("Erreur d'exécution query : %s", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/tools")
def list_available_tools():
    """Retourne la liste des outils (Tools & MCP) enregistrés."""
    pipeline: AgenticRAGPipeline = app_state["pipeline"]
    tools_info = [{"name": t.name, "description": t.description} for t in pipeline.tools]
    return {"tools_count": len(tools_info), "tools": tools_info}