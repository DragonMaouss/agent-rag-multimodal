import logging
from langchain_classic.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import BM25Retriever

logger = logging.getLogger(__name__)

class HybridRetrieverBuilder:
    def __init__(self, config, indexer):
        self.config = config
        self.indexer = indexer

    def build(self):
        """
        Construit un retriever avec recherche hybride (vector/semantic) -> Reranker (cross encoder)
        """
        # Retriever Vectoriel 
        logger.info("Initialisation du Retriever Hybride + BGE CrossEncoder...")
        vectorstore_retriever = self.indexer.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 15}
        )

        # Retriever BM25
        corpus_docs = self.indexer.get_all_corpus_documents()
        if corpus_docs:
            bm25_retriever = BM25Retriever.from_documents(corpus_docs)
            bm25_retriever.k = 15
            # Ensemble retriever (50/50)
            ensemble_retriever = EnsembleRetriever(
                retrievers=[vectorstore_retriever, bm25_retriever],
                weights=[0.5, 0.5]
            )
        else:
            ensemble_retriever = vectorstore_retriever

        reranker_model = HuggingFaceCrossEncoder(model_name=self.config.reranker_model)
        compressor = CrossEncoderReranker(model=reranker_model, top_n=3)

        # Retriever final avec le re-ranker et l'ensemble 
        return ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=ensemble_retriever
        )