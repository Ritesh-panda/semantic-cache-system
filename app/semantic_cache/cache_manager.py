"""
Semantic cache with cluster-aware routing for fast semantic query reuse.

Justification:
1. Cluster routing reduces cache search space, improving lookup scalability.
2. Cosine similarity on embeddings detects semantically equivalent queries.
3. In-memory cache provides millisecond latency without external infrastructure.
"""

import time
import numpy as np
from typing import Optional, Dict, List

from app.embedding_layer.embedding_model import load_embeddings
from app.clustering_layer.fuzzy_cluster import get_memberships
from app.vector_store.faiss_index import load_index


SIM_THRESHOLD = 0.85
MAX_CACHE_SIZE = 1000


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


class SemanticCache:

    def __init__(self):

        self.entries: List[Dict] = []
        self.stats = {"total": 0, "hits": 0, "misses": 0}

        self.eviction_count = 0

        self.memberships = get_memberships()
        self.index = load_index()
        self.embeddings = load_embeddings()

    def add_entry(self, query: str, embedding: np.ndarray, cluster_id: int, result: str):
        """
        Add new cache entry with LRU eviction.
        """

        # LRU eviction
        if len(self.entries) >= MAX_CACHE_SIZE:
            self.entries.pop(0)
            self.eviction_count += 1

        entry = {
            "query": query,
            "embedding": embedding.astype(np.float32),
            "cluster_id": cluster_id,
            "result": result,
            "timestamp": time.time()
        }

        self.entries.append(entry)
        self.stats["total"] += 1

    def lookup(self, query_embedding: np.ndarray, cluster_id: int) -> Optional[Dict]:
        """
        Lookup semantic cache for similar query inside same cluster.
        """

        query_embedding = query_embedding.astype(np.float32)

        best_sim = 0
        best_entry = None

        for entry in self.entries:

            if entry["cluster_id"] != cluster_id:
                continue

            sim = cosine_sim(query_embedding, entry["embedding"])

            if sim > best_sim:
                best_sim = sim
                best_entry = entry

        if best_entry and best_sim > SIM_THRESHOLD:

            self.stats["hits"] += 1

            return {
                "query": best_entry["query"],
                "similarity_score": best_sim,
                "result": best_entry["result"],
                "dominant_cluster": best_entry["cluster_id"]
            }

        self.stats["misses"] += 1
        return None

    def get_stats(self) -> Dict:
        """
        Return cache statistics.
        """

        total = self.stats["total"]
        hits = self.stats["hits"]
        misses = self.stats["misses"]

        hit_rate = hits / (hits + misses) if (hits + misses) > 0 else 0

        return {
            "total_entries": total,
            "hit_count": hits,
            "miss_count": misses,
            "hit_rate": round(hit_rate, 3),
            "cache_size": len(self.entries),
            "max_size": MAX_CACHE_SIZE,
            "evictions": self.eviction_count
        }

    def clear_cache(self):
        """
        Reset cache and statistics.
        """

        self.entries = []
        self.stats = {"total": 0, "hits": 0, "misses": 0}
        self.eviction_count = 0


if __name__ == "__main__":

    print("\n--- Semantic Cache Demo ---\n")

    cache = SemanticCache()

    embeddings = load_embeddings()

    query_id = 1200
    query_embedding = embeddings[query_id]

    memberships = get_memberships()
    cluster_id = int(np.argmax(memberships[query_id]))

    print(f"Query cluster: {cluster_id}")

    print("\nFirst query (expected MISS)")
    result = cache.lookup(query_embedding, cluster_id)

    if result is None:

        print("Cache MISS")

        result_text = f"Top documents related to doc {query_id}"

        cache.add_entry(
            query="space mission failure causes",
            embedding=query_embedding,
            cluster_id=cluster_id,
            result=result_text
        )

    print("\nSecond query (different wording, expected HIT)")

    similar_query_embedding = query_embedding + np.random.normal(0, 0.001, query_embedding.shape)

    hit = cache.lookup(similar_query_embedding, cluster_id)

    if hit:
        print("Cache HIT")
        print(hit)

    print("\nCache Stats")
    print(cache.get_stats())