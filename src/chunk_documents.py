import json
import statistics
from collections import Counter

from config import CHUNKS_DIR, EXTRACTED_DIR

CHUNK_SIZE = 50    # words per chunk
OVERLAP = 10       # words shared between neighbouring chunks

INPUT_FILE = EXTRACTED_DIR / "documents.json"
OUTPUT_FILE = CHUNKS_DIR / "chunks.json"
STATS_FILE = CHUNKS_DIR / "stats.json"


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

    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    word_counts = [chunk["word_count"] for chunk in chunks]
    chunks_per_doc = list(Counter(chunk["doc_id"] for chunk in chunks).values())
    stats = {
        "chunk_size": CHUNK_SIZE,
        "overlap": OVERLAP,
        "documents": len(documents),
        "chunks": len(chunks),
        "min_chunks_per_document": min(chunks_per_doc),
        "max_chunks_per_document": max(chunks_per_doc),
        "median_chunks_per_document": statistics.median(chunks_per_doc),
        "min_chunk_word_count": min(word_counts),
        "max_chunk_word_count": max(word_counts),
        "median_chunk_word_count": statistics.median(word_counts),
    }
    STATS_FILE.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"{len(documents)} documents -> {len(chunks)} chunks")
    print(f"Saved to {OUTPUT_FILE}")
    print(f"Saved stats to {STATS_FILE}")


if __name__ == "__main__":
    main()
