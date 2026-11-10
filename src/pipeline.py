import logging
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langsmith import traceable

from src.tools.mcp_tools import (
    create_rag_retrieval_tool,
    web_search_tool,
    fetch_web_page_tool,
    calculator_tool,
    MCPToolClient
)
from src.guardrails import InputGuardrails, RetrievalGuardrails, OutputGuardrails, GuardrailSecurityError

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
        self.output_guardrails = OutputGuardrails()

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
        
        query_words = set(query.lower().split())
        pruned_docs = []
        for doc in safe_docs:
            sentences = [s.strip() for s in doc.page_content.split('.') if len(s.strip()) > 15]
            if len(sentences) <= max_sentences_per_chunk:
                pruned_docs.append(doc)
                continue

            scored = [(sum(1 for w in s.lower().split() if w in query_words), s) for s in sentences]
            top_sentences = set(s for _, s in sorted(scored, key=lambda x: x[0], reverse=True)[:max_sentences_per_chunk])
            pruned_text = ". ".join([s for s in sentences if s in top_sentences]) + "."
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
                "Basez votre réponse finale sur le retour direct des outils."
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

        ai_msg = self.llm_with_tools.invoke(messages)
        tool_calls_executed = []

        if hasattr(ai_msg, "tool_calls") and ai_msg.tool_calls:
            tools_by_name = {t.name: t for t in self.tools}
            messages.append(ai_msg)
            
            tool_outputs_text = []
            for tool_call in ai_msg.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_call_id = tool_call.get("id", tool_name)
                tool_calls_executed.append({"name": tool_name, "args": tool_args})
                
                selected_tool = tools_by_name.get(tool_name)
                if selected_tool:
                    tool_output = selected_tool.invoke(tool_args)
                    tool_outputs_text.append(f"Output [{tool_name}]: {tool_output}")
                    messages.append(ToolMessage(content=str(tool_output), tool_call_id=tool_call_id))

            synth_prompt = (
                f"Question originale : '{query}'\n\n"
                f"Résultats d'exécution des outils :\n"
                + "\n\n".join(tool_outputs_text)
                + "\n\nRédigez une réponse claire, précise et structurée à l'utilisateur."
            )
            final_response = self.llm.invoke([HumanMessage(content=synth_prompt)])
            raw_answer = final_response.content
        else:
            raw_answer = ai_msg.content

        parsed_answer = self._parse_llm_output(raw_answer)
        sources_list = list(self.last_retrieved_sources)
        final_answer = self.output_guardrails.process(parsed_answer, sources_list)

        if not raw_chat_history and final_answer and query_hits >= self.config.min_frequency_threshold:
            self.db_manager.store_in_semantic_cache(sanitized_query, final_answer, sources_list)

        return {
            "question": sanitized_query,
            "answer": final_answer,
            "sources": sources_list,
            "tool_calls": tool_calls_executed,
            "cache_hit": False
        }