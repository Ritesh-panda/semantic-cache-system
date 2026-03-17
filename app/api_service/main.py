"""
FastAPI service exposing the semantic search system with semantic cache.

This service loads the FAISS index, embeddings, cluster memberships, and cache
at startup to ensure low-latency queries. It implements the exact API contract
required by the assignment with semantic cache routing.
"""

from typing import List
import numpy as np
import time
from collections import OrderedDict, deque, Counter
from sentence_transformers import SentenceTransformer
from app.vector_store.cluster_search import cluster_search
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.semantic_cache.cache_manager import SemanticCache
from app.embedding_layer.embedding_model import load_embeddings
from app.data_layer.data_loader import load_corpus
from app.vector_store.faiss_index import load_index
from app.clustering_layer.fuzzy_cluster import get_memberships


# ----------------------------
# Pydantic Models
# ----------------------------

class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    query: str
    cache_hit: bool
    matched_query: str | None
    similarity_score: float | None
    result: str
    dominant_cluster: int


class CacheStatsResponse(BaseModel):
    total_entries: int
    hit_count: int
    miss_count: int
    hit_rate: float


# ----------------------------
# FastAPI Setup
# ----------------------------

app = FastAPI(title="Semantic Search Cache API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------
# Global State (Loaded Once)
# ----------------------------

cache = SemanticCache()

corpus = []
embeddings = None
memberships = None
index = None

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

query_embedding_cache = OrderedDict()
MAX_QUERY_CACHE = 1000

embedding_cache_hits = 0
embedding_cache_total = 0


# ----------------------------
# Metrics Tracking
# ----------------------------

query_latencies = deque(maxlen=100)
query_timestamps = deque(maxlen=1000)

cluster_counter = Counter()


# ----------------------------
# Startup Event
# ----------------------------

@app.on_event("startup")
def startup_event():

    global corpus, embeddings, memberships, index

    print("\nLoading system components...")

    corpus = load_corpus()
    embeddings = load_embeddings()
    memberships = get_memberships()
    index = load_index()

    print("System ready.")


# ----------------------------
# Metrics Middleware
# ----------------------------

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):

    start_time = time.time()

    response = await call_next(request)

    if request.url.path == "/query":

        latency_ms = (time.time() - start_time) * 1000

        query_latencies.append(latency_ms)
        query_timestamps.append(time.time())

    return response


# ----------------------------
# Utility Functions
# ----------------------------

def embed_query(query: str) -> np.ndarray:

    global embedding_cache_hits, embedding_cache_total

    embedding_cache_total += 1

    if query in query_embedding_cache:

        embedding_cache_hits += 1

        query_embedding_cache.move_to_end(query)

        return query_embedding_cache[query]

    vec = model.encode([query])[0].astype(np.float32)

    query_embedding_cache[query] = vec

    if len(query_embedding_cache) > MAX_QUERY_CACHE:
        query_embedding_cache.popitem(last=False)

    return vec


def get_dominant_cluster(query_embedding: np.ndarray) -> int:

    doc_sims = np.dot(embeddings, query_embedding)

    doc_id = int(np.argmax(doc_sims))

    return int(np.argmax(memberships[doc_id]))


def format_results(indices: List[int]) -> str:

    results = []

    for idx in indices[:3]:

        text = corpus[idx]["text"]

        results.append(f"- {text[:200].replace('\n',' ')}...")

    return "\n".join(results)


# ----------------------------
# POST /query
# ----------------------------

@app.post("/query", response_model=QueryResponse)
def query_endpoint(req: QueryRequest):

    query_embedding = embed_query(req.query)

    cluster_id = get_dominant_cluster(query_embedding)

    cluster_counter[cluster_id] += 1

    cache_hit = cache.lookup(query_embedding, cluster_id)

    if cache_hit:

        return QueryResponse(
            query=req.query,
            cache_hit=True,
            matched_query=cache_hit["query"],
            similarity_score=round(cache_hit["similarity_score"], 3),
            result=cache_hit["result"],
            dominant_cluster=cluster_id
        )

    query_embedding = query_embedding.reshape(1, -1).astype(np.float32)

    distances, indices = cluster_search(query_embedding, cluster_id, 10)

    doc_ids = indices.tolist()

    result_text = format_results(doc_ids)

    cache.add_entry(
        query=req.query,
        embedding=query_embedding[0],
        cluster_id=cluster_id,
        result=result_text
    )

    return QueryResponse(
        query=req.query,
        cache_hit=False,
        matched_query=None,
        similarity_score=0.0,
        result=result_text,
        dominant_cluster=cluster_id
    )
# ----------------------------
# GET /cache/stats
# ----------------------------

@app.get("/cache/stats", response_model=CacheStatsResponse)
def cache_stats():

    stats = cache.get_stats()

    return CacheStatsResponse(
        total_entries=stats["total_entries"],
        hit_count=stats["hit_count"],
        miss_count=stats["miss_count"],
        hit_rate=stats["hit_rate"]
    )


# ----------------------------
# DELETE /cache
# ----------------------------

@app.delete("/cache", status_code=204)
def clear_cache():

    cache.clear_cache()

    return


# ----------------------------
# GET /metrics
# ----------------------------

@app.get("/metrics")
def get_metrics():

    if len(query_latencies) == 0:
        return {
            "p50_latency_ms": 0,
            "p95_latency_ms": 0,
            "cache_hit_rate": 0,
            "embedding_cache_hit_rate": 0,
            "queries_per_minute": 0,
            "top_clusters": []
        }

    p50 = float(np.percentile(list(query_latencies), 50))
    p95 = float(np.percentile(list(query_latencies), 95))

    cache_stats = cache.get_stats()

    cache_hit_rate = cache_stats["hit_rate"]

    embedding_hit_rate = (
        embedding_cache_hits / embedding_cache_total
        if embedding_cache_total > 0 else 0
    )

    now = time.time()

    last_minute = [t for t in query_timestamps if now - t <= 60]

    qpm = len(last_minute)

    top_clusters = [cluster for cluster, _ in cluster_counter.most_common(3)]

    return {
        "p50_latency_ms": round(p50, 2),
        "p95_latency_ms": round(p95, 2),
        "cache_hit_rate": round(cache_hit_rate, 2),
        "embedding_cache_hit_rate": round(embedding_hit_rate, 2),
        "queries_per_minute": qpm,
        "top_clusters": top_clusters
    }