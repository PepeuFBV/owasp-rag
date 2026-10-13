from sentence_transformers import SentenceTransformer

from retrieve_tfidf import load_chunks

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def build_index(chunks):
    """Generate normalized embeddings for all document chunks."""
    if not chunks:
        raise ValueError("Cannot build an index with no chunks")

    model = SentenceTransformer(MODEL_NAME)

    embeddings = model.encode(
        [chunk["text"] for chunk in chunks],
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    return {
        "chunks": chunks,
        "model": model,
        "embeddings": embeddings,
    }


def main():
    chunks = load_chunks()
    index = build_index(chunks)

    print(f"Loaded chunks: {len(chunks)}")
    print(f"Embeddings shape: {index['embeddings'].shape}")
    print(f"Model: {MODEL_NAME}")


if __name__ == "__main__":
    main()
