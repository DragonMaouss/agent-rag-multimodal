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

load_dotenv()

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

UPLOAD_DIR = "data"
GEMINI_MODEL = "gemini-3.5-flash-lite"

# Congiguration du modèle LLM
llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0)


def extract_pdf_multimodal(pdf_path : str) :
    """
    Extrait le texte et les images d'un PDF, puis décrit les images avec Gemini Vision
    """
    if not os.path.exists(pdf_path):
        print(f"Le fichier {pdf_path} n'existe pas.")
        return []

    print(f"Analyse du PDF avec Dockling : {pdf_path} ...")

    loader = DoclingLoader(
        file_path=pdf_path,
        export_type=ExportType.MARKDOWN
    )
    raw_docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    docs = text_splitter.split_documents(raw_docs)

    file_name = os.path.basename(pdf_path)
    for doc in docs :
        doc.page_content = f"Document source : {file_name} {doc.page_content}"

    return filter_complex_metadata(docs)


def process_and_index_file(pdf_path : str) :
    """
    Extrait, découpe et ajoute les documents du PDF dans la base Chroma existante.
    """
    docs = extract_pdf_multimodal(pdf_path=pdf_path)
    if not docs :
        return 0

    vectorstore.add_documents(docs)
    return len(docs)


def get_all_corpus_documents():
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


def get_rag_chain():
    """
    Créer et retourne une chaîne RAG actualisée avec le retriever pointant sur la base
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

    retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble_retriever
    )

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