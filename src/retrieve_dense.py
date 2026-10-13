from sentence_transformers import SentenceTransformer

from retrieve_tfidf import load_chunks

import hashlib
import json

import numpy as np

import argparse

from config import PROCESSED_DIR

CACHE_FILE = PROCESSED_DIR / "embeddings" / "dense_index.npz"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

_INDEX = None


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
    dimensions = model.get_embedding_dimension()

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


def retrieve(question, k=3):
    if type(k) is not int or k < 1:
        raise ValueError("k must be a positive integer")

    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")

    global _INDEX
    if _INDEX is None:
        _INDEX = build_index(load_chunks())

    query_embedding = _INDEX["model"].encode(
        [question.strip()],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # normalized vectors: dot product equals cosine similarity.
    scores = _INDEX["embeddings"] @ query_embedding[0]
    top = np.argsort(-scores)[:k]

    return [
        {
            "rank": rank,
            "chunk_id": _INDEX["chunks"][i]["chunk_id"],
            "doc_id": _INDEX["chunks"][i]["doc_id"],
            "title": _INDEX["chunks"][i]["title"],
            "score": float(scores[i]),
            "text": _INDEX["chunks"][i]["text"],
            "source_url": _INDEX["chunks"][i]["source_url"],
        }
        for rank, i in enumerate(top, start=1)
    ]



def main():
    parser = argparse.ArgumentParser(
        description="Dense semantic retrieval over OWASP chunks"
    )
    parser.add_argument("question")
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()

    try:
        results = retrieve(args.question, k=args.k)
    except ValueError as exc:
        parser.error(str(exc))

    for result in results:
        excerpt = result["text"][:200].replace("\n", " ")

        print(
            f"#{result['rank']}  "
            f"score={result['score']:.3f}  "
            f"{result['title']}"
        )
        print(f"    {result['chunk_id']}")
        print(f"    {excerpt}...\n")


if __name__ == "__main__":
    main()
