import os
import json
import tarfile
import urllib.request
from pathlib import Path

"""
UCI 20 Newsgroups Data Loader

Uses the official UCI dataset instead of sklearn wrapper.
Removes headers, quotes, and signatures to improve embedding quality.
Stores corpus as JSONL for efficient loading of ~18k documents.
"""

DATA_URL = "http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz"

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
ARCHIVE_PATH = RAW_DIR / "20news-bydate.tar.gz"

TRAIN_DIR = RAW_DIR / "20news-bydate-train"
TEST_DIR = RAW_DIR / "20news-bydate-test"

CORPUS_PATH = DATA_DIR / "corpus.json"


# ------------------------------
# Download Dataset
# ------------------------------

def download_dataset():

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if ARCHIVE_PATH.exists():
        print("Dataset already downloaded.")
        return

    print("Downloading UCI 20 Newsgroups dataset...")

    urllib.request.urlretrieve(DATA_URL, ARCHIVE_PATH)

    print("Download complete.")


# ------------------------------
# Extract Dataset
# ------------------------------

def extract_dataset():

    if TRAIN_DIR.exists() and TEST_DIR.exists():
        print("Dataset already extracted.")
        return

    print("Extracting dataset...")

    with tarfile.open(ARCHIVE_PATH, "r:gz") as tar:
        tar.extractall(RAW_DIR)

    print("Extraction complete.")


# ------------------------------
# Clean Text
# ------------------------------

def clean_text(text):

    lines = text.splitlines()

    cleaned = []
    header_passed = False

    for line in lines:

        # skip headers until first blank line
        if not header_passed:
            if line.strip() == "":
                header_passed = True
            continue

        # remove quoted replies
        if line.strip().startswith(">"):
            continue

        # remove signatures
        if line.strip().startswith("--"):
            break

        line = line.strip()

        if len(line) > 50:
            cleaned.append(line)

    return " ".join(cleaned)


# ------------------------------
# Parse Dataset
# ------------------------------

def parse_documents():

    docs = []
    doc_id = 0

    folders = [TRAIN_DIR, TEST_DIR]

    for folder in folders:

        for root, _, files in os.walk(folder):

            for file in files:

                path = Path(root) / file

                try:

                    with open(path, "r", errors="ignore") as f:
                        text = f.read()

                    cleaned = clean_text(text)

                    # remove extremely short docs
                    if len(cleaned.split()) < 20:
                        continue

                    docs.append({
                        "id": doc_id,
                        "text": cleaned
                    })

                    doc_id += 1

                except Exception:
                    continue

    print(f"Parsed {len(docs)} documents.")

    return docs


# ------------------------------
# Save Corpus (JSONL)
# ------------------------------

def save_corpus(docs):

    DATA_DIR.mkdir(exist_ok=True)

    with open(CORPUS_PATH, "w", encoding="utf-8") as f:

        for doc in docs:
            f.write(json.dumps(doc) + "\n")

    print(f"Saved corpus → {CORPUS_PATH}")


# ------------------------------
# Load Corpus
# ------------------------------

def load_corpus():

    if CORPUS_PATH.exists():

        print("Loading existing corpus...")

        with open(CORPUS_PATH, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    download_dataset()

    extract_dataset()

    docs = parse_documents()

    save_corpus(docs)

    return docs


# ------------------------------
# Stats
# ------------------------------

def get_stats():

    corpus = load_corpus()

    lengths = [len(doc["text"]) for doc in corpus]

    avg_len = sum(lengths) / len(lengths)

    stats = {
        "total_docs": len(corpus),
        "avg_length": avg_len
    }

    return stats


# ------------------------------
# Test Runner
# ------------------------------

if __name__ == "__main__":

    corpus = load_corpus()

    stats = get_stats()

    print("\nCorpus Statistics")
    print("------------------")
    print(f"Total Documents: {stats['total_docs']}")
    print(f"Average Length: {stats['avg_length']:.2f}")

    print("\nExample Document:")
    print(corpus[0]["text"][:300])