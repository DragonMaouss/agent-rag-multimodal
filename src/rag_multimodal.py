import os
import io
import pymupdf 
import base64
from PIL import Image
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_huggingface import HuggingFaceEmbeddings

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

# Congiguration du modèle LLM
llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0)

def extract_pdf_multimodal(pdf_path : str) :
    """
    Extrait le texte et les images d'un PDF, puis décrit les images avec Gemini Vision
    """
    if not os.path.exists(pdf_path):
        print(f"Le fichier {pdf_path} n'existe pas.")
        return []

    doc = pymupdf.open(pdf_path)
    documents = []

    # Modèle multimodal pour la description des images
    vision_llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash", temperature=0)

    print(f"Analyse du PDF : {pdf_path} ({len(doc)} pages) ...")

    for page_num in range(len(doc)):
        page = doc[page_num]

        # 1. Extraction du texte
        text = page.get_text()
        if text.strip():
            documents.append(
                Document(
                    page_content=text, 
                    metadata={"source": pdf_path, "page": page_num + 1}
                    )
                )

        # 2. Extraction des images
        image_list = page.get_images(full=True)
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]

            try: 
                # Conversion des bytes en image PIL
                image = Image.open(io.BytesIO(image_bytes))

                print(f'Image trouvée à la page {page_num + 1}, index {img_index + 1} : {image_ext}, taille: {image.size}')

                # Encode l'image en base64 pour l'envoyer à Gemini Vision
                buffered = io.BytesIO()
                image.save(buffered, format="JPEG" if image.format != "PNG" else "PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                mime_type = "image/png" if image.format == "PNG" else "image/jpeg"

                # 3. Description de l'image avec Gemini Vision
                message = HumanMessage(
                    content=[
                        {"type": "text", "text": "Décris précisément ce schéma, graphique ou cette image en détaillant les informations clés, les titres et les données affichées pour un usage documentaire."},
                        {
                            "type": "image_url",
                            "image_url": f"data:{mime_type};base64,{img_base64}"
                        },
                    ]
                )
                
                response = vision_llm.invoke([message])
                image_description = response.content 

                # Ajout de la description de l'image comme document
                img_doc_content = f"Description de l'image à la page {page_num + 1}, index {img_index + 1} : {image_description}"
                documents.append(
                    Document(
                        page_content=img_doc_content, 
                        metadata={"source": pdf_path, "page": page_num + 1, "image_index": img_index + 1}
                    )
                )
            except Exception as e:
                print(f"Erreur lors de la description de l'image à la page {page_num + 1}, index {img_index + 1} : {e}")

    return documents

def process_and_index_file(pdf_path : str) :
    """
    Extrait, découpe et ajoute les documents du PDF dans la base Chroma existante.
    """
    docs = extract_pdf_multimodal(pdf_path=pdf_path)
    if not docs :
        return 0

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    split_docs = text_splitter.split_documents(docs)
    vectorstore.add_documents(split_docs)

    return len(docs)


def get_rag_chain():
    """
    Créer et retourne une chaîne RAG actualisée avec le retriever pointant sur la base
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k":10})

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
