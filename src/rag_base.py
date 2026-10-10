import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Charger les variables d'environnement à partir du fichier .env
load_dotenv()

def run_basic_rag() : 
    # Création d'un petit fichier texte pour le test
    sample_file_path = "data/sample.txt"
    if not os.path.exists(sample_file_path):
        os.makedirs(os.path.dirname(sample_file_path), exist_ok=True)
        with open(sample_file_path, "w") as f:
            f.write(
                "Le projet Agent RAG Multimodal utilise Python, FastAPI, Streamlit et Qdrant."     
                "Il a pour objectif d'analyser des documents complexes incluant du texte et des images."
                "L'intelligence artificielle générative permet d'interroger ces sources de manière fluide."
            )

    print("--- 1. Chargement du document ---")
    loader = TextLoader(sample_file_path)
    docs = loader.load()

    print("--- 2. Découpage du document en chunks ---")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=(50))
    split_docs = text_splitter.split_documents(docs)

    print("--- 3. Création de l'index Chroma ---")
    embedding = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

    vectorstore = Chroma.from_documents(
        documents=split_docs, 
        embedding=embedding,
        persist_directory="./chroma_db"
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    print("--- 4. Création de la chaîne RAG ---")
    llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0)

    system_prompt = (
        "Vous êtes un assistant utile et précis. "
        "Répondez aux questions en utilisant uniquement sur le context fourni ci dessous. \n\n"
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
    rag_chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=question_answer_chain
    )

    print("--- 5. Test de la chaîne RAG ---")
    query = "Quelles technologies sont utilisées pour le projet Agent RAG Multimodal ?"
    print(f"Question: {query}")

    responde = rag_chain.invoke({"input": query})
    print(f"Réponse: {responde['answer']}")

if __name__ == "__main__":
    run_basic_rag()