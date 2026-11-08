import os
from pydantic_settings import BaseSettings
from typing import List, Optional

class Settings(BaseSettings):
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    collection_name: str = "multimodal_collection"
    cache_collection_name: str = "semantic_cache_collection"
    staging_collection_name: str = "query_frequency_collection"
    
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    reranker_model: str = "BAAI/bge-reranker-base"
    gemini_model: str = "gemini-3.5-flash-lite"
    
    min_frequency_threshold: int = 3
    cache_similarity_threshold: float = 0.97
    upload_dir: str = "data"
    supported_extensions: tuple = (".pdf", ".docx", ".pptx", ".xlsx", ".html", ".md")

    enable_tools: bool = True
    mcp_server_urls: List[str] = [os.getenv("MCP_SERVER_URL", "http://localhost:8001/sse")]
    mcp_connect_timeout: int = 10

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()