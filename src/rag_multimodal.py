import os
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import BM25Retriever
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams


# Chargement des variables d'environnement
load_dotenv()

# Initialisation globale du RAG au démarrage de l'application
print("--- Initialisation du RAG Multimodal ---")


QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "multimodal_collection"
UPLOAD_DIR = "data"
GEMINI_MODEL = "gemini-3.8-flash"

# Connexion au client Qdrant
client = QdrantClient(url=QDRANT_URL)

# Creation de la collection si elle n'existe pas
try : 
    client.get_collection(collection_name=COLLECTION_NAME)
    print(f"Collection Qdrant {COLLECTION_NAME} trouve")
except Exception:
    print(f"Creation de la collection Qdrant {COLLECTION_NAME} ...")
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE) # 768 pour all-mpnet-base-v2
    )


# Initialisation du modele d'embeddings HuggingFace
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    encode_kwargs={"normalize_embeddings": True},
)

# Chargement de la base Qdrant
vectorstore = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=embedding,
)

# Congiguration du modèle LLM
llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0)


def process_and_index_file(pdf_path : str) :
    """
    Extrait le contenu d'un ficher PDF au format Markdown, decoupe en chunks, enrichis les metatdonnes et les ajoute dans la base ChromaDB.
    """
    if not os.path.exists(pdf_path):
        print(f"Le fichier {pdf_path} n'existe pas.")
        return []

    print(f"Analyse du PDF avec Dockling : {pdf_path} ...")

    # Chargement du PDF avec Docling au format Markdown
    loader = DoclingLoader(
        file_path=pdf_path,
        export_type=ExportType.MARKDOWN
    )
    raw_docs = loader.load()

    # Decoupage en chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    docs = text_splitter.split_documents(raw_docs)

    # Préfixe chaque chunk avec le nom du ficher pour garder le contexte
    file_name = os.path.basename(pdf_path)
    for doc in docs :
        doc.page_content = f"Document source : {file_name} {doc.page_content}"

    # Ajoute les PDF dans la base Qdrant
    vectorstore.add_documents(filter_complex_metadata(docs))

    return len(docs)


def get_all_corpus_documents():
    """
    Parcourt le dossier d'upload et charge tout les PDF pour BM25
    """
    all_docs = []
    if os.path.exists(UPLOAD_DIR):
        for file_name in os.listdir(UPLOAD_DIR):
            if file_name.endswith(".pdf"):
                file_path = os.path.join(UPLOAD_DIR, file_name)
                try :
                    loader = DoclingLoader(
                        file_path=file_path,
                        export_type=ExportType.MARKDOWN
                    )
                    raw_docs = loader.load()
                    
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200
                    )
                    docs = text_splitter.split_documents(raw_docs)

                    for doc in docs :
                        doc.page_content = f"Document source : {file_name} {doc.page_content}"
                    all_docs.extend(filter_complex_metadata(docs))
                except Exception as e:
                    print(f"Erreur lors du chargement de {file_name} pour BM25 : {e}")
    return all_docs


def get_retriever():
    """"
    Construit un retriever avec recherche hybride (vector/semantic) -> Reranker (cross encoder)
    """
    # Retriever Vectoriel 
    vectorstore_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k":20}
    )

    # Retriever BM25
    corpus_docs = get_all_corpus_documents()
    if corpus_docs:
        bm25_retriever = BM25Retriever.from_documents(corpus_docs)
        bm25_retriever.k = 25

        # Ensemble retriever (50/50)
        ensemble_retriever = EnsembleRetriever(
            retrievers=[vectorstore_retriever,bm25_retriever],
            weights=[0.5,0.5]
        )
    else :
        ensemble_retriever = vectorstore_retriever
    
    # Model de reranking Cross Encoder
    reranker_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-large")

    # Compresseur qui retient les top_n documents les plus pertinants
    compressor = CrossEncoderReranker(
        model=reranker_model,
        top_n=5
    )

    # Retriever final avec le re-ranker et l'ensemble 
    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble_retriever
    )    


def get_rag_chain():
    """
    Retourne une chaîne RAG actualisée avec le retriever pointant sur la base
    """
    retriever = get_retriever()

    system_prompt = (
        "Vous êtes un assistant expert capable d'analyser à la fois du texte et des descriptions de schémas ou d'images. "
        "Répondez aux questions en vous basant uniquement sur le contexte fourni ci-dessous.\n\n"
        "{context}"
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}")
        ]
    )

    question_answer_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt
    )

    return create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=question_answer_chain
    )


def rewrite_query(user_query: str):
    """
    Reecrit la requete de l'utilisateur pour l'optimiser avant la requete
    """
    rewrite_prompt = ChatPromptTemplate.from_messages([
        ("system", "Vous êtes un expert en recherche documentaire. "
                   "Votre rôle est de réécrire la question de l'utilisateur pour qu'elle soit claire, "
                   "autonome et maximise la pertinence d'une recherche par similarité dans des documents techniques. "
                   "Ne répondez pas à la question, donnez uniquement la requête reformulée."),
        ("human", "Question originale : {query}")
    ])

    rewriter_chain = rewrite_prompt | llm | StrOutputParser()
    rewritten = rewriter_chain.invoke({"query" : user_query})
    return rewritten