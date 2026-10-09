import json

from config import PROCESSED_DIR

CHUNK_SIZE = 200   # words per chunk
OVERLAP = 40       # words shared between neighbouring chunks

INPUT_FILE = PROCESSED_DIR / "documents.json"
OUTPUT_FILE = PROCESSED_DIR / "chunks.json"


def chunk_words(words, size, overlap):
    """Yield (start_index, words_in_chunk) with a fixed window and fixed overlap."""
    if overlap >= size:
        raise ValueError("OVERLAP must be smaller than CHUNK_SIZE")
    step = size - overlap
    for start in range(0, len(words), step):
        yield start, words[start:start + size]
        if start + size >= len(words):
            break


def main():
    data = json.loads(INPUT_FILE.read_text(encoding="utf-8"))

    # The file has metadata at the top; the documents are stored in a list inside it
    documents = data["documents"] if "documents" in data else next(
        v for v in data.values() if isinstance(v, list)
    )

    chunks = []
    for doc in documents:
        words = doc["text"].split()
        for i, (start, window) in enumerate(chunk_words(words, CHUNK_SIZE, OVERLAP)):
            chunks.append({
                "chunk_id": f"{doc['doc_id']}_c{i:03d}",
                "doc_id": doc["doc_id"],
                "title": doc["title"],
                "source_url": doc["source_url"],
                "chunk_index": i,
                "start_word": start,
                "end_word": start + len(window),
                "word_count": len(window),
                "text": " ".join(window),
            })

    OUTPUT_FILE.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"{len(documents)} documents -> {len(chunks)} chunks")
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
