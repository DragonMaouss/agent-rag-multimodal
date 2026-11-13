import logging
import numpy as np
import re
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langsmith import traceable

from src.tools.mcp_tools import create_rag_retrieval_tool, web_search_tool, fetch_web_page_tool, calculator_tool, MCPToolClient
from src.guardrails import InputGuardrails, RetrievalGuardrails, OutputGuardrails, ToolGuardrails, FinalResponseGuardrails

logger = logging.getLogger(__name__)

class AgenticRAGPipeline:
    def __init__(self, config, db_manager, indexer, retriever_builder):
        self.config = config
        self.db_manager = db_manager
        self.indexer = indexer
        self.llm = ChatGoogleGenerativeAI(model=config.gemini_model, temperature=0)
        self.retriever_builder = retriever_builder
        
        # Instanciation de la liste avant d'initialiser les outils
        self.last_retrieved_sources: List[Dict[str, Any]] = []
        self.active_user_roles: List[str] = ["public"]

        self.input_guardrails = InputGuardrails()
        self.retrieval_guardrails = RetrievalGuardrails()
        self.tool_guardrails = ToolGuardrails()
        self.output_guardrails = OutputGuardrails()
        self.final_response_guardrails = FinalResponseGuardrails()

        self.tools = self._initialize_tools()
        self.llm_with_tools = self.llm.bind_tools(self.tools) if self.tools else self.llm

    def _initialize_tools(self) -> List[Any]:
        tools = []
        if self.config.enable_tools:
            # Passage explicite de last_sources_store
            rag_tool = create_rag_retrieval_tool(
                retriever_builder=self.retriever_builder,
                prune_context_fn=self.prune_context,
                last_sources_store=self.last_retrieved_sources,
                current_user_roles_fn=lambda: self.active_user_roles
            )
            tools.extend([rag_tool, web_search_tool, fetch_web_page_tool, calculator_tool])
            mcp_client = MCPToolClient(server_urls=self.config.mcp_server_urls)
            tools.extend(mcp_client.get_tools_sync())
            
        logger.info("%d outils (Tools & MCP) enregistrés dans l'agent.", len(tools))
        return tools

    def _parse_llm_output(self, content: Any) -> str:
        """
        Extrait et convertit le contenu du LLM en chaîne de caractères (str).
        """
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            extracted_text = []
            for item in content:
                if isinstance(item, str):
                    extracted_text.append(item)
                elif isinstance(item, dict) and "text" in item:
                    extracted_text.append(item["text"])
                elif hasattr(item, "text"):
                    extracted_text.append(item.text)
            return "\n".join(extracted_text)
        return str(content) if content is not None else ""

    @traceable(name="REWRITE_QUERY", tags=["optimization"])
    def rewrite_query(self, user_query: str):
        rewrite_prompt = ChatPromptTemplate.from_messages([
            ("system", "Vous êtes un expert en recherche documentaire. "
                       "Réécrivez la question utilisateur pour qu'elle soit claire et précise."),
            ("human", "Question originale : {query}")
        ])
        rewriter_chain = rewrite_prompt | self.llm
        res = rewriter_chain.invoke({"query": user_query})
        return self._parse_llm_output(res.content)

    @traceable(name="PRUNE_CONTEXT", tags=["optimization", "pruning"])
    def prune_context(self, docs: List[Document], query: str, max_sentences_per_chunk: int = 3):
        """
        Élimine le bruit à l'intérieur des chunks retenus en ne gardant que 
        les phrases ayant la plus forte similarité sémantique avec la requête.
        """
        safe_docs = self.retrieval_guardrails.validate_and_sanitize_docs(docs)
        if not safe_docs:
            return []
        
        query_vec = np.array(self.indexer.embedding.embed_query(query), dtype=np.float32)
        pruned_docs = []
        for doc in safe_docs:
            # Découpage sur ponctuation forte ou sauts de ligne (adapté au Markdown Docling)
            sentences = [
                s.strip()
                for s in re.split(r"(?<=[.!?])\s+|\n+", doc.page_content)
                if len(s.strip()) > 15
            ]
            if len(sentences) <= max_sentences_per_chunk:
                pruned_docs.append(doc)
                continue
            
            sentence_vecs = np.array(self.indexer.embedding.embed_documents(sentences), dtype=np.float32)
            norms = np.linalg.norm(sentence_vecs, axis=1)
            valid_mask = norms > 0
            
            similarities = np.zeros(len(sentences), dtype=np.float32)
            similarities[valid_mask] = (sentence_vecs[valid_mask] @ query_vec) / (norms[valid_mask] * query_norm)
            
            # Sélection des meilleures phrases puis tri des indices pour préserver l'ordre du texte
            top_indices = sorted(np.argsort(similarities)[::-1][:max_sentences_per_chunk])
            pruned_text = " ".join(sentences[i] for i in top_indices)
            pruned_docs.append(Document(page_content=pruned_text, metadata=doc.metadata))

        return pruned_docs

    @traceable(name="PIPELINE_RUN", tags=["production", "agentic_rag"])
    def run_query(self, query: str, user_roles: Optional[List[str]] = None, raw_chat_history: Optional[List[Dict[str, Any]]] = None):
        self.active_user_roles = user_roles or ["public"]
               
        if not query.strip():
            raise ValueError("La requête ne peut pas être vide.")

        self.last_retrieved_sources.clear()

        # Input Guardrails
        sanitized_query, detected_pii = self.input_guardrails.process(query)

        if not raw_chat_history:
            cached_res = self.db_manager.check_semantic_cache(sanitized_query)
            if cached_res:
                return cached_res

        query_hits = self.db_manager.track_query_frequency(sanitized_query)

        messages = [
            SystemMessage(content=(
                "Vous êtes un assistant IA autonome doté d'outils réels :\n"
                "1. `search_internal_documents` : Utilisez cet outil pour toute question relative aux documents internes de l'entreprise.\n"
                "2. `web_search_tool` : Utilisez cet outil pour chercher sur Internet si la réponse est absente des documents internes ou concerne l'actualité.\n"
                "3. `fetch_web_page_tool` : Utilisez cet outil pour lire le contenu textuel complet d'une URL web spécifique.\n"
                "4. `calculator_tool` : Utilisez cet outil pour réaliser des calculs arithmétiques exacts.\n"
                "Vous pouvez enchaîner plusieurs appels d'outils si nécessaire avant de rédiger une réponse finale claire, précise et structurée."
            ))
        ]
        
        if raw_chat_history:
            for msg in raw_chat_history:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                else:
                    messages.append(AIMessage(content=msg["content"]))

        optimized_query = self.rewrite_query(sanitized_query)
        messages.append(HumanMessage(content=optimized_query))

        tool_calls_executed = []
        raw_answer = None
        tools_by_name = {t.name: t for t in self.tools}

        # Boucle d'agent multi-tours bornée
        for iteration in range(self.config.max_agent_iterations):
            ai_msg = self.llm_with_tools.invoke(messages)
            messages.append(ai_msg)

            if not getattr(ai_msg, "tool_calls", None):
                raw_answer = ai_msg.content
                break

            for tool_call in ai_msg.tool_calls:
                tool_name = tool_call["name"]
                raw_args = tool_call["args"]
                tool_call_id = tool_call.get("id", f"{tool_name}_{iteration}")

                # Validation RBAC et schéma d'arguments à chaque tour
                validated_args = self.tool_guardrails.validate_tool_call(
                    tool_name=tool_name,
                    tool_args=raw_args,
                    user_roles=self.active_user_roles
                )

                tool_calls_executed.append({"name": tool_name, "args": validated_args})
                selected_tool = tools_by_name.get(tool_name)

                if selected_tool:
                    try:
                        tool_output = selected_tool.invoke(validated_args)
                    except Exception as e:
                        logger.error("Erreur lors de l'exécution de l'outil %s : %s", tool_name, e)
                        tool_output = f"Erreur d'exécution de l'outil {tool_name} : {str(e)}"
                else:
                    tool_output = f"Outil '{tool_name}' indisponible."

                messages.append(ToolMessage(content=str(tool_output), tool_call_id=tool_call_id))

        # Synthèse forcée si le nombre maximal d'itérations est atteint
        if raw_answer is None:
            logger.warning("Plafond de %d itérations atteint, synthèse finale forcée.", self.config.max_agent_iterations)
            messages.append(HumanMessage(
                content="Limite d'appels d'outils atteinte. Rédigez maintenant la réponse finale complète à partir des résultats déjà obtenus."
            ))
            final_response = self.llm.invoke(messages)
            raw_answer = final_response.content

        # Output Guardrails
        parsed_answer = self._parse_llm_output(raw_answer)
        sources_list = list(self.last_retrieved_sources)
        sanitized_answer = self.output_guardrails.process(parsed_answer, sources_list)

        if not raw_chat_history and sanitized_answer and query_hits >= self.config.min_frequency_threshold:
            self.db_manager.store_in_semantic_cache(sanitized_query, sanitized_answer, sources_list)

        return self.final_response_guardrails.seal_response(
            question=sanitized_query,
            answer=sanitized_answer,
            sources=sources_list,
            tool_calls=tool_calls_executed,
            detected_pii=detected_pii,
            cache_hit=False
        )