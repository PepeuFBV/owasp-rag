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

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

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
