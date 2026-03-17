import time
import numpy as np

from app.data_layer.data_loader import load_corpus
from app.embedding_layer.embedding_model import get_embeddings
from app.vector_store.faiss_index import build_index
from app.clustering_layer.fuzzy_cluster import fit_clusters


def rebuild_pipeline():

    start_time = time.time()

    print("\nRebuilding Semantic Search Pipeline")
    print("-----------------------------------")

    # -------------------------
    # Step 1 — Load dataset
    # -------------------------
    print("\n[1/4] Loading dataset...")

    corpus = load_corpus()

    print(f"Documents loaded: {len(corpus)}")

    # -------------------------
    # Step 2 — Generate embeddings
    # -------------------------
    print("\n[2/4] Generating embeddings...")

    embeddings = get_embeddings(corpus)

    print(f"Embedding shape: {embeddings.shape}")

    # Save embeddings so other modules reuse them
    np.save("data/embeddings.npy", embeddings)

    print("Saved embeddings → data/embeddings.npy")

    # -------------------------
    # Step 3 — Build FAISS index
    # -------------------------
    print("\n[3/4] Building FAISS index...")

    index = build_index()

    print("FAISS index built.")

    # -------------------------
    # Step 4 — Run fuzzy clustering
    # -------------------------
    print("\n[4/4] Running fuzzy clustering...")

    fit_clusters(embeddings)

    print("Clustering completed.")

    # -------------------------
    # Pipeline complete
    # -------------------------
    total_time = time.time() - start_time

    print("\nPipeline Rebuild Complete")
    print("--------------------------")
    print(f"Total Time: {total_time:.2f} seconds")


if __name__ == "__main__":
    rebuild_pipeline()