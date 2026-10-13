from sentence_transformers import SentenceTransformer

from retrieve_tfidf import load_chunks

import hashlib
import json

import numpy as np

from config import PROCESSED_DIR

CACHE_FILE = PROCESSED_DIR / "embeddings" / "dense_index.npz"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def build_index(chunks):
    if not chunks:
        raise ValueError("Cannot build an index with no chunks")

    model = SentenceTransformer(MODEL_NAME)
    embeddings = load_or_generate_embeddings(chunks, model)

    return {
        "chunks": chunks,
        "model": model,
        "embeddings": embeddings,
    }


def corpus_fingerprint(chunks):
    payload = [
        [chunk["chunk_id"], chunk["text"]]
        for chunk in chunks
    ]
    content = json.dumps(payload, ensure_ascii=False)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def load_or_generate_embeddings(chunks, model):
    fingerprint = corpus_fingerprint(chunks)
    dimensions = model.get_sentence_embedding_dimension()

    if CACHE_FILE.exists():
        with np.load(CACHE_FILE, allow_pickle=False) as cached:
            embeddings = cached["embeddings"]

            if (
                cached["model_name"].item() == MODEL_NAME
                and cached["fingerprint"].item() == fingerprint
                and embeddings.shape == (len(chunks), dimensions)
            ):
                print("[cache] Loaded existing embeddings")
                return embeddings

    print("[cache] Generating embeddings")

    embeddings = model.encode(
        [chunk["text"] for chunk in chunks],
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        CACHE_FILE,
        embeddings=embeddings,
        model_name=MODEL_NAME,
        fingerprint=fingerprint,
    )

    print("[cache] Embeddings saved")
    return embeddings


def main():
    chunks = load_chunks()
    index = build_index(chunks)

    print(f"Loaded chunks: {len(chunks)}")
    print(f"Embeddings shape: {index['embeddings'].shape}")
    print(f"Model: {MODEL_NAME}")


if __name__ == "__main__":
    main()
