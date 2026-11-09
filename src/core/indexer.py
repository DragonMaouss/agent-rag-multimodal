import os
import logging
from typing import List, Optional
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from langsmith import traceable

logger = logging.getLogger(__name__)

class DocumentIndexer:
    def __init__(self, config, embedding_function, qdrant_client):
        self.config = config
        self.embedding = embedding_function
        self.qdrant_client = qdrant_client
        self.semantic_splitter = SemanticChunker(
            embeddings=self.embedding,
            breakpoint_threshold_type="percentile"
        )
        self.vectorstore = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=self.config.collection_name,
            embedding=self.embedding,
        )
        self._corpus_cache: List[Document] = []
        self._is_corpus_loaded = False

    @traceable(name="INDEX_FILE", tags=["ingestion"])
    def index_file(self, file_path: str, access_roles: Optional[List[str]] = None):
        """
        Extrait et indexe un document en lui assignant des rôles d'accès RBAC (ex: ['rh', 'admin']).
        """
        if not os.path.exists(file_path):
            logger.warning("Fichier introuvable : %s", file_path)
            return 0

        ext = os.path.splitext(file_path)[1].lower()
        if ext not in self.config.supported_extensions:
            logger.warning("Format non supporté : %s", ext)
            return 0

        roles = access_roles or ["public"]
        file_name = os.path.basename(file_path)

        try:
            loader = DoclingLoader(file_path=file_path, export_type=ExportType.MARKDOWN)
            raw_docs = loader.load()
            docs = self.semantic_splitter.split_documents(raw_docs)

            # Préfixe chaque chunk avec le nom du fichier pour garder le contexte
            file_name = os.path.basename(file_path)
            for doc in docs:
                doc.page_content = f"Document source : {file_name}\n{doc.page_content}"
                doc.metadata["source"] = file_name
                doc.metadata["access_roles"] = roles  # Stockage des rôles dans le payload Qdrant

            filtered_docs = filter_complex_metadata(docs)
            self.vectorstore.add_documents(filtered_docs)
            
            # Mise à jour immédiate du cache BM25
            self._corpus_cache.extend(filtered_docs)
            logger.info("%d chunks indexés pour %s avec rôles : %s", len(docs), file_name, roles)
            return len(docs)
        except Exception as e:
            logger.error("Erreur lors de l'indexation de %s : %s", file_path, e)
            return 0

    def get_all_corpus_documents(self) -> List[Document]:
        """
        Récupère tous les documents déjà indexés directement depuis la collection Qdrant
        pour construire l'index BM25 sans ré-exécuter Docling sur le disque au démarrage.
        """
        if self._is_corpus_loaded and self._corpus_cache:
            return self._corpus_cache

        logger.info("Chargement instantané du corpus depuis Qdrant pour BM25...")
        all_docs = []
        offset = None
        batch_size = 500

        try:
            while True:
                records, next_offset = self.qdrant_client.scroll(
                    collection_name=self.config.collection_name,
                    limit=batch_size,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False
                )

                for record in records:
                    payload = record.payload or {}
                    # Récupération du contenu textuel du payload Qdrant
                    content = payload.get("page_content") or payload.get("text") or ""
                    metadata = payload.get("metadata") or {
                        k: v for k, v in payload.items() if k not in ("page_content", "text")
                    }
                    if content:
                        all_docs.append(Document(page_content=content, metadata=metadata))

                if next_offset is None:
                    break
                offset = next_offset

            logger.info("%d documents chargés depuis Qdrant pour BM25 en quelques millisecondes.", len(all_docs))
        except Exception as e:
            logger.error("Erreur lors de la récupération du corpus depuis Qdrant : %s", e)

        self._corpus_cache = all_docs
        self._is_corpus_loaded = True
        return self._corpus_cache