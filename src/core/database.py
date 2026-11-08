import logging
import uuid
from typing import Optional, Dict, Any, List
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.models import PointStruct
from langsmith import traceable

logger = logging.getLogger(__name__)

class QdrantManager:
    def __init__(self, config, embedding_function):
        self.config = config
        self.embedding = embedding_function
        self.client = QdrantClient(url=config.qdrant_url)
        self._init_collections()

    def _init_collections(self):
        for coll in [self.config.collection_name, self.config.cache_collection_name, self.config.staging_collection_name]:
            try:
                self.client.get_collection(collection_name=coll)
                logger.info("Collection Qdrant active : %s", coll)
            except Exception:
                logger.info("Création de la collection Qdrant : %s", coll)
                self.client.create_collection(
                    collection_name=coll,
                    vectors_config=VectorParams(size=768, distance=Distance.COSINE)
                )

    @traceable(name="CHECK_SEMANTIC_CACHE", tags=["cache"])
    def check_semantic_cache(self, user_query: str):
        """
        Recherche dans Qdrant une question déjà posée sémantiquement similaire.
        """
        try:
            query_vector = self.embedding.embed_query(user_query)
            # Prise en charge des nouvelles versions (query_points) et anciennes (search) de qdrant-client
            if hasattr(self.client, "query_points"):
                response = self.client.query_points(
                    collection_name=self.config.cache_collection_name,
                    query=query_vector,
                    limit=1,
                    score_threshold=self.config.cache_similarity_threshold
                )
                results = response.points
            else:
                results = self.client.search(
                    collection_name=self.config.cache_collection_name,
                    query_vector=query_vector,
                    limit=1,
                    score_threshold=self.config.cache_similarity_threshold
                )

            if results:
                hit = results[0]
                payload = hit.payload
                return {
                    "question": user_query,
                    "answer": payload.get("answer"),
                    "sources": payload.get("sources", []),
                    "cache_hit": True,
                    "similarity_score": float(hit.score)
                }
        except Exception as e:
            logger.error("Erreur vérification cache sémantique : %s", e)
        return None

    @traceable(name="STORE_SEMANTIC_CACHE", tags=["cache"])
    def store_in_semantic_cache(self, user_query: str, answer: str, sources: List[Dict[str, Any]]):
        """
        Stocke le vecteur de la question et la réponse générée dans le cache.
        """
        try:
            query_vector = self.embedding.embed_query(user_query)
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, user_query))
            self.client.upsert(
                collection_name=self.config.cache_collection_name,
                points=[
                    PointStruct(
                        id=point_id,
                        vector=query_vector,
                        payload={"query": user_query, "answer": answer, "sources": sources}
                    )
                ]
            )
        except Exception as e:
            logger.error("Erreur stockage cache sémantique : %s", e)

    @traceable(name="TRACK_QUERY_FREQUENCY", tags=["cache"])
    def track_query_frequency(self, user_query: str):
        """
        Vérifie si la question (ou sa reformulation sémantique) a déjà été posée.
        Incrémente son compteur et retourne la fréquence totale.
        """
        try:
            query_vector = self.embedding.embed_query(user_query)
            if hasattr(self.client, "query_points"):
                response = self.client.query_points(
                    collection_name=self.config.staging_collection_name,
                    query=query_vector,
                    limit=1,
                    score_threshold=0.95
                )
                results = response.points
            else:
                results = self.client.search(
                    collection_name=self.config.staging_collection_name,
                    query_vector=query_vector,
                    limit=1,
                    score_threshold=0.95
                )

            if results:
                hit = results[0]
                current_count = hit.payload.get("hit_count", 1) + 1
                self.client.set_payload(
                    collection_name=self.config.staging_collection_name,
                    payload={"hit_count": current_count},
                    points=[hit.id]
                )
                return current_count
            else:
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, user_query))
                self.client.upsert(
                    collection_name=self.config.staging_collection_name,
                    points=[
                        PointStruct(
                            id=point_id,
                            vector=query_vector,
                            payload={"query": user_query, "hit_count": 1}
                        )
                    ]
                )
                return 1
        except Exception as e:
            logger.error("Erreur suivi fréquence : %s", e)
            return 1