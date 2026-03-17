"""
Production embedding generator for the semantic search corpus.

Justification:
1. SentenceTransformer MiniLM-L6-v2 provides strong semantic embeddings while remaining CPU efficient.
2. Batch processing with torch.no_grad() ensures memory efficiency and avoids gradient overhead.
3. Float32 numpy storage keeps embeddings FAISS-compatible while minimizing memory footprint.
"""

import time
import psutil
import numpy as np
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer

from app.data_layer.data_loader import load_corpus


EMBEDDINGS_PATH = Path("data/embeddings.npy")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 128
SEED = 42


def get_embeddings(corpus) -> np.ndarray:
    """
    Generate embeddings for the full corpus.
    Returns numpy array [N, 384] float32.
    """

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    texts = [doc["text"] for doc in corpus]

    model = SentenceTransformer(MODEL_NAME, device="cpu")

    start_time = time.time()
    process = psutil.Process()

    with torch.no_grad():

        embeddings = model.encode(
            texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=False
        )

    embeddings = embeddings.astype(np.float32)

    total_time = time.time() - start_time
    avg_ms = (total_time / len(texts)) * 1000
    peak_ram = process.memory_info().rss / (1024 ** 3)

    print("\nEmbedding Generation Stats")
    print("--------------------------")
    print(f"Documents: {len(texts)}")
    print(f"Embedding Dimension: {embeddings.shape[1]}")
    print(f"Total Time: {total_time:.2f} sec")
    print(f"Avg Time per Doc: {avg_ms:.2f} ms")
    print(f"Peak RAM Usage: {peak_ram:.2f} GB")

    return embeddings


def load_embeddings() -> np.ndarray:
    """
    Load embeddings from disk if already computed.
    """

    if not EMBEDDINGS_PATH.exists():
        corpus = load_corpus()
        embeddings = get_embeddings(corpus)

        EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.save(EMBEDDINGS_PATH, embeddings)

        print(f"\nSaved embeddings → {EMBEDDINGS_PATH}")

        return embeddings

    print("Loading embeddings from disk...")
    embeddings = np.load(EMBEDDINGS_PATH)

    return embeddings


if __name__ == "__main__":

    corpus = load_corpus()

    embeddings = load_embeddings()

    print("\nEmbedding Matrix")
    print("----------------")
    print(f"Shape: {embeddings.shape}")
    print(f"Dtype: {embeddings.dtype}")