"""
Cluster-aware FAISS search.

Idea:
1) Predict the dominant cluster for the query.
2) Restrict candidate docs to those with membership > 0.1 in that cluster.
3) Run FAISS once (oversampled) and filter to cluster docs → return top-k.

This reduces the effective search space (~16k → ~1–2k) and improves latency.
"""

import time
import numpy as np
from app.vector_store.faiss_index import load_index
from app.clustering_layer.fuzzy_cluster import get_memberships

# Load once at import
index = load_index()
memberships = get_memberships()  # shape: [num_docs, num_clusters]

# Precompute cluster → doc_ids mapping (membership > 0.1)
CLUSTER_THRESHOLD = 0.1
cluster_to_docs = {
    cid: set(np.where(memberships[:, cid] > CLUSTER_THRESHOLD)[0].tolist())
    for cid in range(memberships.shape[1])
}


def cluster_search(query_embedding: np.ndarray, cluster_id: int, k: int = 10):
    """
    FAISS search restricted to docs strongly belonging to `cluster_id`.

    Args:
        query_embedding: np.ndarray of shape (1, d), float32
        cluster_id: int
        k: number of results

    Returns:
        distances: np.ndarray shape (k,)
        indices: np.ndarray shape (k,)
    """

    # Oversample FAISS results to ensure enough after filtering
    oversample = min(200, index.ntotal)
    distances, indices = index.search(query_embedding, oversample)

    allowed = cluster_to_docs.get(cluster_id, set())

    filtered_ids = []
    filtered_dists = []

    for dist, idx in zip(distances[0], indices[0]):
        if idx in allowed:
            filtered_ids.append(idx)
            filtered_dists.append(dist)
            if len(filtered_ids) == k:
                break

    # Fallback: if too few results in cluster, pad with FAISS top results
    if len(filtered_ids) < k:
        for dist, idx in zip(distances[0], indices[0]):
            if idx not in filtered_ids:
                filtered_ids.append(idx)
                filtered_dists.append(dist)
                if len(filtered_ids) == k:
                    break

    return np.array(filtered_dists), np.array(filtered_ids)


# ----------------------------
# Speed test
# ----------------------------
if __name__ == "__main__":
    from app.embedding_layer.embedding_model import load_embeddings

    embeddings = load_embeddings()
    q = embeddings[100].reshape(1, -1).astype(np.float32)

    # Full FAISS search
    t0 = time.time()
    d_full, i_full = index.search(q, 10)
    full_ms = (time.time() - t0) * 1000

    # Cluster-aware search
    cid = int(np.argmax(memberships[100]))
    t1 = time.time()
    d_clu, i_clu = cluster_search(q, cid, 10)
    clu_ms = (time.time() - t1) * 1000

    print("\nSpeed Comparison")
    print("----------------")
    print(f"Full FAISS search:    {full_ms:.3f} ms")
    print(f"Cluster-aware search: {clu_ms:.3f} ms")