import logging
import asyncio
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup
import httpx
from langchain_core.tools import tool, BaseTool
from langchain_core.documents import Document

try:
    from duckduckgo_search import DDGS
    HAS_DDG = True
except ImportError:
    HAS_DDG = False

logger = logging.getLogger(__name__)

class RAGSearchToolInput:
    query: str

def create_rag_retrieval_tool(retriever_builder, prune_context_fn, last_sources_store: list):
    """
    Transforme la recherche hybride RAG en outil pour l'agent.
    """
    retriever = retriever_builder.build()

    @tool("search_internal_documents")
    def search_internal_documents(query: str) -> str:
        """
        Recherche des informations dans les documents internes indexés (PDF, Word, Excel, PowerPoint, Markdown).
        À utiliser lorsque la question porte sur la documentation de l'entreprise, des schémas ou des données métiers.
        """
        try:
            docs: List[Document] = retriever.invoke(query)
            pruned_docs = prune_context_fn(docs, query=query)
            if not pruned_docs:
                return "Aucun document pertinent trouvé dans la base interne."
            
            # Réinitialisation et capture des sources
            last_sources_store.clear()
            seen = set()
            formatted_results = []
            
            for doc in pruned_docs:
                meta = doc.metadata or {}
                source_file = meta.get("source", "Inconnu")
                page_num = meta.get("page", 1)
                img_idx = meta.get("image_index", None)
                identifier = (source_file, page_num, img_idx)
                
                if identifier not in seen:
                    seen.add(identifier)
                    last_sources_store.append({
                        "source": source_file,
                        "page": page_num,
                        "image_index": img_idx
                    })
                
                formatted_results.append(f"Extrait [{source_file} p.{page_num}]:\n{doc.page_content}")
                
            return "\n\n---\n\n".join(formatted_results)
        except Exception as e:
            logger.error("Erreur lors de l'exécution du tool RAG : %s", e)
            return f"Erreur lors de la recherche dans les documents : {str(e)}"

    return search_internal_documents


@tool("web_search_tool")
def web_search_tool(query: str, max_results: int = 4) -> str:
    """
    Effectue une recherche en temps réel sur Internet via DuckDuckGo.
    À utiliser lorsque l'information demandée concerne l'actualité, le web externe ou si la documentation interne est insuffisante.
    """
    if not HAS_DDG:
        return "Module `duckduckgo_search` non disponible."
    try:
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append(f"Titre: {r.get('title')}\nURL: {r.get('href')}\nExtrait: {r.get('body')}")
        
        if not results:
            return "Aucun résultat trouvé sur le web."
        return "\n\n---\n\n".join(results)
    except Exception as e:
        logger.error("Erreur Web Search : %s", e)
        return f"Erreur de recherche Web : {str(e)}"


@tool("fetch_web_page_tool")
def fetch_web_page_tool(url: str) -> str:
    """
    Télécharge et extrait le texte brut nettoyé d'une page Web à partir d'une URL spécifique.
    À utiliser pour lire en profondeur le contenu d'un lien trouvé lors d'une recherche web.
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        response = httpx.get(url, headers=headers, timeout=10.0, follow_redirects=True)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, "html.parser")
        for element in soup(["script", "style", "nav", "footer", "header", "noscript"]):
            element.decompose()
            
        text = soup.get_text(separator="\n", strip=True)
        lines = [line for line in text.splitlines() if len(line.strip()) > 20]
        cleaned_text = "\n".join(lines[:60])
        
        return f"Contenu extrait de {url} :\n\n{cleaned_text}"
    except Exception as e:
        return f"Erreur de lecture de la page {url} : {str(e)}"

    
@tool("calculator_tool")
def calculator_tool(expression: str) -> str:
    """
    Calcule des expressions mathématiques complexes ou financières.
    Exemple d'entrée : '1500 * (1 + 0.20)'
    """
    try:
        # Évaluation sécurisée d'expressions mathématiques
        allowed_names = {"abs": abs, "round": round}
        code = compile(expression, "<string>", "eval")
        for name in code.co_names:
            if name not in allowed_names:
                raise NameError(f"Utilisation de '{name}' non autorisée.")
        result = eval(code, {"__builtins__": {}}, allowed_names)
        return str(result)
    except Exception as e:
        return f"Erreur de calcul : {str(e)}"


class MCPToolClient:
    """
    Client de connexion au protocole MCP (Model Context Protocol).
    Permet de récupérer les outils exposés dynamiquement par des serveurs MCP (stdio ou SSE).
    """
    def __init__(self, server_urls: List[str]):
        self.server_urls = server_urls

    async def fetch_mcp_tools(self) -> List[BaseTool]:
        """
        Découvre et convertit les outils MCP distants en outils LangChain.
        """
        mcp_tools: List[BaseTool] = []
        for url in self.server_urls:
            try:
                # Simulation de la résolution JSON-RPC / SSE avec le protocole MCP
                logger.info("Connexion au serveur MCP à l'adresse : %s", url)
                # Note : En environnement complet, utiliser le SDK officiel `mcp` (mcp.client.sse)
            except Exception as e:
                logger.warning("Impossible de charger les outils MCP depuis %s : %s", url, e)
        return mcp_tools

    def get_tools_sync(self) -> List[BaseTool]:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                return []
            return loop.run_until_complete(self.fetch_mcp_tools())
        except Exception:
            return []