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

The main output is:

```text
data/processed/documents.json
```

Each document contains:

```json
{
  "doc_id": "...",
  "uuid": "...",
  "title": "...",
  "source_file": "...",
  "source_url": "...",
  "word_count": 0,
  "text": "..."
}
```

`data/processed/stats.json` contains basic size statistics.

The next step should be chunking `documents.json`; do not add retrieval yet.
