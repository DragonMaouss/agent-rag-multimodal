import os
from fastapi import FastAPI, HTTPException 
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

app = FastAPI(
    title="Agent RAG Multimodal API",
    description="API pour l'Agent RAG Multimodal utilisant FastAPI, LangChain et Qdrant.",
    version="1.0.0",
)

# Initialisation globale du RAG au démarrage de l'application
print("--- Initialisation du RAG Multimodal ---")

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    encode_kwargs={"normalize_embeddings": True},
)

# Chargement de la base Chroma
vectorstore = Chroma(
    persist_directory="./chroma_multimodal_db",
    embedding_function=embedding,
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Congiguration du modèle LLM
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0)

system_prompt = (
    "Vous êtes un assistant expert capable d'analyser à la fois du texte et des descriptions de schémas ou d'images. "
    "Répondez aux questions en vous basant uniquement sur le contexte fourni ci-dessous.\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    (
        ("system", system_prompt),
        ("human", "{input}")
    )
)

question_answer_chain = create_stuff_documents_chain(
    llm=llm,
    prompt=prompt
)

rag_chain = create_retrieval_chain(
    retriever=retriever,
    combine_docs_chain=question_answer_chain,
)

print("--- RAG Multimodal initialisé avec succès ---")

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    question: str
    answer: str

@app.get("/")
def read_root():
    return {"message": "Bienvenue sur l'API de l'Agent RAG Multimodal. Utilisez le point de terminaison /query pour poser vos questions."}

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