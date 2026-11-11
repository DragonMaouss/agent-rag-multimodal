import logging
from typing import List, Optional
from langchain_classic.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import BM25Retriever
from qdrant_client.http import models as qdrant_models

logger = logging.getLogger(__name__)

class HybridRetrieverBuilder:
    def __init__(self, config, indexer):
        self.config = config
        self.indexer = indexer
        self.reranker_model = HuggingFaceCrossEncoder(model_name=self.config.reranker_model)
        self.compressor = CrossEncoderReranker(model=self.reranker_model, top_n=3)

    def build(self, user_roles: Optional[List[str]] = None) -> ContextualCompressionRetriever:        
        """
        Construit un retriever avec recherche hybride (vector/semantic) -> Reranker (cross encoder)
        """
        roles = user_roles or ["public"]
        logger.info("Construction du Retriever Hybride RBAC pour les rôles : %s", roles)

        # 1. Filtre Qdrant selon les privilèges RBAC
        qdrant_filter = qdrant_models.Filter(
            must=[
                qdrant_models.FieldCondition(
                    key="metadata.access_roles",
                    match=qdrant_models.MatchAny(any=roles)
                )
            ]
        )

        # 2. Retriever Vectoriel filtré par les rôles
        logger.info("Initialisation du Retriever Hybride + BGE CrossEncoder...")
        vectorstore_retriever = self.indexer.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 15, "filter": qdrant_filter}
        )

        # 3. Retriever BM25 filtré par rôles
        corpus_docs = self.indexer.get_all_corpus_documents()
        allowed_corpus = [
            d for d in corpus_docs
            if any(role in d.metadata.get("access_roles", ["public"]) for role in roles)
        ]

        if allowed_corpus:
            bm25_retriever = BM25Retriever.from_documents(allowed_corpus)
            bm25_retriever.k = 15
            # Ensemble retriever (50/50)
            ensemble_retriever = EnsembleRetriever(
                retrievers=[vectorstore_retriever, bm25_retriever],
                weights=[0.5, 0.5]
            )
        else:
            ensemble_retriever = vectorstore_retriever

        # 4. Retriever final avec Re-ranker
        return ContextualCompressionRetriever(
            base_compressor=self.compressor,
            base_retriever=ensemble_retriever
        )