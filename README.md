# RAG OWASP

Python pipeline for the RAG assignment, reusing the original corpus and retrieval scripts:

1. download a focused set of OWASP Cheat Sheet documents;
2. clean the Markdown;
3. measure document lengths;
4. select 25 documents closest to the candidate-set median;
5. save the corpus as JSON.
6. Chunk the documents into smaller pieces for retrieval.
7. Save the chunks as JSON.
8. Build a TF-IDF index over the chunks and retrieve relevant passages (P2).
9. Build a dense embedding index and retrieve semantically relevant passages (P3).
10. Generate answers locally with a free Hugging Face LLM and chunk citations (P4).
11. Run five paired questions with and without retrieval, producing ten answers (P5).
12. Grade retrieval and answers separately by hand (P6).
13. Attribute two wrong answers and export a report of at most two pages (P7).

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

The tests cover chunk loading, TF-IDF and dense retrieval ranking, cache reuse and invalidation, chunk IDs, invalid query inputs, prompt evidence, citations, paired runs, log integrity, preservation of human reviews, and failure attribution. Tests use mock models and do not download weights.

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

## P4-P7: generation, experiments and manual evaluation

The default LLM is `Qwen/Qwen2.5-1.5B-Instruct`, a public, free, pretrained model downloaded from Hugging Face. No paid API or access token is needed. It runs locally on CUDA when available, or on CPU with float32. A free Colab GPU is recommended. The model download takes several GB; generation on CPU is slower. No fine-tuning is required: RAG provides the OWASP evidence at inference time, and this assignment does not request training a new LLM.

Generate a single answer:

```bash
python src/generate_answers.py "How can I prevent SQL injection?" --retriever dense --k 3
python src/generate_answers.py "How can I prevent SQL injection?" --retriever none
```

Run the full experiment after creating the corpus and chunks:

```bash
python src/run_experiment.py
```

Both retrievers rank the same five questions. The selected retriever (`dense` by default) supplies the RAG context. The LLM runs twice per question, with identical system instructions and greedy decoding settings. Reference answers in `data/questions.json` are for human review only and are never supplied to the LLM. Each run logs the full evidence and exact rendered prompt, model revision, generation settings, token counts, output-limit status, and citation ID checks. A valid ID alone does not prove semantic support.

The assignment asks for three paired questions (six answers) and ten manually graded answers. The default uses five paired questions, including the first three as the required comparison. Confirm this interpretation with the instructor; the question list and runner are parameterized. The final assignment report requires ten answers.

Optional experiment settings:

```bash
python src/run_experiment.py --retriever tfidf --k 4 --output-dir output/tfidf_experiment
python src/run_experiment.py --model Qwen/Qwen2.5-0.5B-Instruct --output-dir output/small_model
```

Use a fresh output directory when changing the experiment. Existing experiments are never overwritten, and interrupted runs retain their partial answer log and retrieved evidence. Prompts that exceed the model's input budget fail explicitly; chunks are never silently truncated. Increase `--max-input-tokens` within the model context window or reduce `--k` when needed. A different model is a separate experiment; keep it fixed within each paired comparison.

### Manual grading

Outputs under `output/experiment/`:

| File | Purpose |
| --- | --- |
| `experiment.json` | Status, model revision, settings, dependency versions and content fingerprints. |
| `questions.json` | Snapshot of the evaluated questions and reference answers. |
| `retrievals.json` | Full top-k chunks and scores from both retrievers, saved before generation. |
| `answers.jsonl` | Ten answers, exact prompts, supplied chunks and generation metadata. |
| `retrieval_grades.csv` | Human judgment of whether each top-k contains all required evidence. |
| `answer_grades.csv` | Human grades for correctness, support and citation correctness. |
| `review.json` | Two selected wrong answer IDs and the responsible-search risk/mitigation. |
| `report_draft.pdf` | Clearly labeled preview with pending manual fields. |
| `report.pdf` | Final two-page report, only after complete manual evaluation. |

1. Grade `retrieval_grades.csv` first: `evidence_found=yes/no`, evidence chunk IDs separated by `;`, and a justification in `notes`. A relevant document title is not enough; the retrieved text must contain every fact needed.
2. Grade all ten answers in `answer_grades.csv`: `correct`, `supported`, and `citation_correct` use `yes/no`. For baseline answers, support and citations are `na` because there is no retrieved context. Record evidence IDs and a human justification. Review factual correctness and every part of the question, independently of grounding.
3. Set exactly two distinct `failure_run_ids` in `review.json`, each with a manually graded failure: `correct=no`, or a RAG answer with `supported=no` or `citation_correct=no`. Their answer notes must explain the actual error. RAG failures are retrieval when evidence is absent, generation when evidence is present. Baseline errors are generation/knowledge failures, never retrieval failures. If fewer than two answers have failures, run a new set of questions rather than inventing them.
4. Review the suggested manipulation risk and mitigation in `review.json`. The prompt treats source excerpts as untrusted data. This reduces risk but is not a guarantee against prompt injection.
5. Validate and export:

```bash
python src/evaluate_answers.py --require-complete
python src/build_report.py
```

CSV column names and logged IDs must remain unchanged. Quote notes containing commas. Human grading files are never reset by rerunning template creation. The validator checks IDs, completeness and log fingerprints, not whether a person's semantic judgment is correct.

To preview the report before grading:

```bash
python src/build_report.py --draft
```

The exporter refuses a final report with incomplete grades or without two real wrong answers. It checks the two-page limit and asks for shorter justifications when necessary. A draft is not a completed submission.

### Notebook and Colab

Open `notebooks/rag_project.ipynb`. It prepares the corpus when needed, shows both retrieval rankings and full evidence, generates or validates existing paired results, displays all answers and prompts, guides manual evaluation, and exports the report. On Colab, choose a free GPU and upload the provided `owasp-rag-colab.zip`. Use the modified project files; cloning the unchanged upstream repository does not include P4-P7.

The notebook uses `output/experiment_colab/` for new Colab runs. To inspect the supplied local experiment without downloading models, change `OUTPUT_DIR` to `PROJECT_DIR / "output" / "experiment"`. If you change any inputs, choose a new directory instead of reusing old reviews.

After grading, rerun the evaluation/report cells and download the `.ipynb` through Colab's File menu to retain outputs. Submit the executed notebook and `report.pdf`; the final PDF includes the grading table and answers to assignment items 6 and 7.

For local execution with a registered Python kernel:

```bash
python -m ipykernel install --prefix .venv --name python3
jupyter nbconvert --to notebook --execute notebooks/rag_project.ipynb --output rag_project.executed.ipynb --ExecutePreprocessor.timeout=3600
```

`notebooks/rag_project.executed.ipynb`, when supplied, contains a real local run. It still labels any unfinished human evaluation as pending. Detailed file responsibilities are in `IMPLEMENTATION_NOTES.md`.
