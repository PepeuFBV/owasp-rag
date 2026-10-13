# RAG OWASP starter

Very small starter for the first step of the RAG assignment:

1. download a focused set of OWASP Cheat Sheet documents;
2. clean the Markdown;
3. measure document lengths;
4. select 25 documents closest to the candidate-set median;
5. save the corpus as JSON.
6. Chunk the documents into smaller pieces for retrieval.
7. Save the chunks as JSON.
8. Build a TF-IDF index over the chunks and retrieve relevant passages (P2).
9. Build a dense embedding index and retrieve semantically relevant passages (P3).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Tests

Run the automated tests from the repository root with the project virtual environment active:

```bash
python -m unittest discover -s tests -v
```

The tests cover chunk loading, TF-IDF and dense retrieval ranking, cache reuse and invalidation, chunk IDs, and invalid query inputs. Dense retrieval tests use a mock model, so running the suite does not download the embedding model.

## Run

```bash
python src/download_docs.py
python src/build_dataset.py
python src/chunk_documents.py
```

Outputs are written under `data/processed/`:

```text
data/processed/
├── extracted/
│   ├── documents.json   # selected, cleaned documents (build_dataset.py)
│   └── stats.json       # document size statistics
└── chunks/
    ├── chunks.json      # chunks for retrieval (chunk_documents.py)
    └── stats.json       # chunk count and size statistics
```

The main output is `data/processed/chunks/chunks.json`, an array of chunks in the format:

```json
{
  "chunk_id": "...",
  "doc_id": "...",
  "title": "...",
  "source_url": "...",
  "chunk_index": <int>,
  "start_word": <int>,
  "end_word": <int>,
  "word_count": <int>,
  "text": "..."
}
```

## P2: TF-IDF retrieval

The P2 retriever builds a TF-IDF index over the saved chunks and ranks them by cosine similarity to a question. Install the dependencies and generate the dataset and chunks first, then run a query from the repository root:

```bash
python src/retrieve_tfidf.py "How can I prevent SQL injection?" --k 3
```

`--k` controls how many results are returned (default: `3`). Each result shows its rank, cosine similarity score, title, chunk ID, and a short text excerpt. For example, the output has this shape (scores and matching passages depend on the generated corpus):

```text
#1  score=0.XXX  SQL Injection Prevention
    owasp_sql_injection_prevention_cheat_sheet_c000
    ...retrieved passage excerpt...

#2  score=0.XXX  Query Parameterization
    owasp_query_parameterization_cheat_sheet_c000
    ...retrieved passage excerpt...
```

## P3: Dense retrieval

The P3 retriever encodes each saved chunk with `nomic-ai/nomic-embed-text-v1.5` and ranks chunks by cosine similarity to the question. It uses `search_document:` and `search_query:` prefixes and caps the model input at 1,024 tokens. Install the requirements and generate the dataset and chunks as shown above before querying from the repository root:

```bash
python src/retrieve_dense.py "How can I prevent SQL injection?" --k 3
```

`--k` controls the number of results (default: `3`). Each result shows its rank, cosine similarity score, title, chunk ID, and a short text excerpt. The first query downloads the model from Hugging Face if it is not already available locally.

The first dense query also encodes all chunks and saves the embeddings to `data/processed/embeddings/dense_index.npz`. Later runs reuse that cache when its model name, chunk content fingerprint, and embedding dimensions match the current index. If the corpus or model no longer matches, embeddings are generated again and the cache is replaced.

P1 uses 200-word chunks with a 40-word overlap. The current 25-document corpus produces 441 chunks. With the Nomic tokenizer, their median is 346 tokens and the longest is 900, so none exceed the 1,024-token limit. Recalculate the truncation analysis after changing the corpus or model with:

```bash
python src/analyze_embedding_truncation.py
```

Run the test suite from the repository root to check dense and TF-IDF retrieval, cache behavior, and input validation:

```bash
python -m unittest discover -s tests -v
```
