# RAG OWASP starter

Very small starter for the first step of the RAG assignment:

1. download a focused set of OWASP Cheat Sheet documents;
2. clean the Markdown;
3. measure document lengths;
4. select 25 documents closest to the candidate-set median;
5. save the corpus as JSON.
6. Chunk the documents into smaller pieces for retrieval.
7. Save the chunks as JSON.

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
