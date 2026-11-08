import logging
import os
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langsmith import traceable

logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(self, config, db_manager, indexer, retriever_builder):
        self.config = config
        self.db_manager = db_manager
        self.indexer = indexer
        self.llm = ChatGoogleGenerativeAI(model=config.gemini_model, temperature=0)
        self.retriever = retriever_builder.build()
        self.rag_chain = self._build_rag_chain()

    def _build_rag_chain(self):
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", "Étant donné un historique de chat et la dernière question de l'utilisateur "
                       "qui pourrait faire référence à des contextes précédents dans l'historique, "
                       "formulez une question autonome qui peut être comprise sans l'historique. "
                       "Ne répondez pas à la question, reformulez-la simplement si nécessaire ou renvoyez-la telle quelle."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        history_aware_retriever = create_history_aware_retriever(
            self.llm, self.retriever, contextualize_q_prompt
        )

        system_prompt = (
            "Vous êtes un assistant expert capable d'analyser du texte et des schémas. "
            "Répondez aux questions en vous basant uniquement sur le contexte fourni ci-dessous.\n\n"
            "{context}"
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        question_answer_chain = create_stuff_documents_chain(llm=self.llm, prompt=prompt)
        return create_retrieval_chain(retriever=history_aware_retriever, combine_docs_chain=question_answer_chain)

    @traceable(name="REWRITE_QUERY", tags=["optimization"])
    def rewrite_query(self, user_query: str):
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
        rewriter_chain = rewrite_prompt | self.llm | StrOutputParser()
        return rewriter_chain.invoke({"query": user_query})

    @traceable(name="PRUNE_CONTEXT", tags=["optimization", "pruning"])
    def prune_context(self, docs: List[Document], query: str, max_sentences_per_chunk: int = 3):
        """
        Élimine le bruit à l'intérieur des chunks retenus en ne gardant que 
        les phrases ayant la plus forte similarité sémantique avec la requête.
        """
        if not docs:
            return []
        query_words = set(query.lower().split())
        pruned_docs = []
        for doc in docs:
            sentences = [s.strip() for s in doc.page_content.split('.') if len(s.strip()) > 15]
            if len(sentences) <= max_sentences_per_chunk:
                pruned_docs.append(doc)
                continue

            scored = [(sum(1 for w in s.lower().split() if w in query_words), s) for s in sentences]
            top_sentences = set(s for _, s in sorted(scored, key=lambda x: x[0], reverse=True)[:max_sentences_per_chunk])
            pruned_text = ". ".join([s for s in sentences if s in top_sentences]) + "."
            pruned_docs.append(Document(page_content=pruned_text, metadata=doc.metadata))

        return pruned_docs

    @traceable(name="PIPELINE_RUN", tags=["production"])
    def run_query(self, query: str, raw_chat_history: Optional[List[Dict[str, Any]]] = None):
        if not query.strip():
            raise ValueError("La requête ne peut pas être vide.")

        if not raw_chat_history:
            cached_res = self.db_manager.check_semantic_cache(query)
            if cached_res:
                return cached_res

        query_hits = self.db_manager.track_query_frequency(query)

        langchain_history = []
        if raw_chat_history:
            for msg in raw_chat_history:
                if msg["role"] == "user":
                    langchain_history.append(HumanMessage(content=msg["content"]))
                else:
                    langchain_history.append(AIMessage(content=msg["content"]))

        optimized_query = self.rewrite_query(query)
        response = self.rag_chain.invoke({
            "input": optimized_query,
            "chat_history": langchain_history
        })

        raw_docs = response.get("context", [])
        pruned_docs = self.prune_context(raw_docs, query=optimized_query)

        sources_list = []
        seen = set()
        for doc in pruned_docs:
            meta = doc.metadata
            source_file = os.path.basename(meta.get("source", "Inconnu"))
            page_num = meta.get("page", 1)
            img_idx = meta.get("image_index", None)
            identifier = (source_file, page_num, img_idx)
            if identifier not in seen:
                seen.add(identifier)
                sources_list.append({"source": source_file, "page": page_num, "image_index": img_idx})

        answer = response["answer"]
        if not raw_chat_history and answer and query_hits >= self.config.min_frequency_threshold:
            self.db_manager.store_in_semantic_cache(query, answer, sources_list)

        return {
            "question": query,
            "answer": answer,
            "sources": sources_list,
            "cache_hit": False
        }