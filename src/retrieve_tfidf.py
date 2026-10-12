import argparse
import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import CHUNKS_DIR

CHUNKS_PATH = CHUNKS_DIR / "chunks.json"
_INDEX = None


def load_chunks(path=CHUNKS_PATH):
    chunks = json.loads(path.read_text(encoding="utf-8"))
    for i, c in enumerate(chunks):
        missing = {"chunk_id", "doc_id", "text"} - c.keys()
        if missing:
            raise ValueError(f"Chunk {i} is missing: {missing}")
    return chunks


def build_index(chunks):
    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), norm="l2")
    matrix = vectorizer.fit_transform([c["text"] for c in chunks])
    return {"chunks": chunks, "vectorizer": vectorizer, "matrix": matrix}


def retrieve(question, k=3):
    global _INDEX
    if _INDEX is None:
        _INDEX = build_index(load_chunks())

    q_vec = _INDEX["vectorizer"].transform([question])
    scores = cosine_similarity(q_vec, _INDEX["matrix"]).ravel()
    top = scores.argsort()[::-1][:k]

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
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--k", type=int, default=3)
    args = parser.parse_args()

    for r in retrieve(args.question, k=args.k):
        excerpt = r["text"][:200].replace("\n", " ")
        print(f"#{r['rank']}  score={r['score']:.3f}  {r['title']}")
        print(f"    {r['chunk_id']}")
        print(f"    {excerpt}...\n")


if __name__ == "__main__":
    main()
