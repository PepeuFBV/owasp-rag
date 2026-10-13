# Implementation audit and file responsibilities

## Existing work preserved

The repository already implemented P1-P3: public OWASP download, median-based selection of 25 documents, 200-word chunks with a 40-word overlap, TF-IDF retrieval, and dense Nomic embeddings with a persistent fingerprinted cache. It also included embedding truncation analysis and 13 automated tests. The clone initially had no generated corpus, embeddings, notebook, LLM generation, experiment log, grading workflow, or report exporter.

The existing corpus/retrieval implementations remain unchanged. `src/config.py` now also owns the question-file and experiment-directory paths. `requirements.txt` explicitly includes PyTorch and the notebook/PDF dependencies. `.gitignore` excludes downloaded model caches and notebook checkpoints.

## New code and inputs

| File | Responsibility |
| --- | --- |
| `src/generate_answers.py` | Loads one free pretrained Qwen LLM, selects CUDA/CPU, renders the same system policy in both conditions, injects exact top-k text as untrusted JSON, requests chunk citations, rejects oversized prompts, generates deterministically, and reports token/citation diagnostics. A standalone CLI answers one question. |
| `src/run_experiment.py` | Loads and validates questions, compares both existing retrievers, snapshots evidence before generation, runs five paired conditions, logs the exact experiment, records failure/completion, protects prior logs, and verifies complete saved results before reuse. |
| `src/evaluate_answers.py` | Creates human grading CSVs and the risk/failure review file without overwriting them. Validates row IDs, log integrity, yes/no/na choices, evidence IDs, notes, and two genuinely wrong answers. Attributes failures from human retrieval evidence judgments. It never auto-grades factual correctness. |
| `src/build_report.py` | Creates a labeled draft or a final PDF with ten answer grades, retrieval evidence judgments, two failure analyses, and one search risk/mitigation. Final export requires complete human evaluation and enforces exactly two pages. |
| `data/questions.json` | Five English OWASP questions, stable IDs, expected documents and reviewer reference answers. References are excluded from prompts. Topics: SQL injection, password storage, file uploads, SSRF and logging. |
| `notebooks/rag_project.ipynb` | Colab/local workflow for corpus preparation, configuration, retrieval/answer comparison, inspection of full prompts, manual grading, validation and PDF export. Uses a separate directory for fresh Colab runs. |
| `tests/test_rag_pipeline.py` | Focused tests with a fake LLM for source preservation, references excluded from prompts, citation checks, paired settings, immutable logs, failed runs, review preservation, artifact fingerprints, evidence integrity, incomplete grading and failure attribution. |
| `IMPLEMENTATION_NOTES.md` | This audit and explanation of the new files. |

## Generated artifacts

The existing scripts create downloaded Markdown, the 25-document corpus, 441 chunks and the Nomic embedding cache. The exact counts can change if upstream documents change.

The runner creates `output/experiment/experiment.json`, `questions.json`, `retrievals.json`, and `answers.jsonl`. These retain model identity/revision, settings, dependency versions, fingerprints, both rankings, all prompts, supplied sources, token counts and ten real model answers.

`answer_grades.csv`, `retrieval_grades.csv`, and `review.json` are human evaluation inputs. Baseline support/citation fields use `na`; the other grades and the selection of two errors remain pending until reviewed. The suggested risk is manipulation through instructions in retrieved documents, mitigated with allow-listed public sources, explicit data boundaries and human citation review. This is a risk reduction, not a security guarantee.

`notebooks/rag_project.executed.ipynb` preserves execution outputs when local execution completes. `output/experiment/report_draft.pdf` is a two-page preview, clearly labeled unfinished. `report.pdf` is created only after all manual review requirements pass. A completed generation run alone is not a completed assignment submission.

`owasp-rag-colab.zip` packages source files, notebook, questions and small experiment artifacts for upload to Colab; it omits the virtual environment, Git metadata and model weights. On Colab, the model is downloaded at runtime and the notebook uses a new experiment directory.

## Experimental choices and limits

Five questions in two conditions yield the required ten answers while including the three requested comparisons. Confirm this count interpretation with the instructor. Both retrievers use the same chunks/questions, but the selected retriever supplies the generation context; evaluating the other retriever does not produce extra LLM answers.

The default generator is `Qwen/Qwen2.5-1.5B-Instruct`, public and free. Fine-tuning is unnecessary for the requested RAG task and would add cost and a different experiment. All training weights are pretrained; no paid inference endpoint is used. The GPU/CPU choice can affect floating-point results, so actual device, dtype and model revision are logged. Model answers, abstentions, output truncation and citation mistakes remain visible rather than being rewritten or repaired.

Humans must judge correctness, support and citation semantics, then select and explain two actual failures. A selected answer must have incorrect facts, unsupported claims, or incorrect citations, as defined by the roadmap. Baseline answers qualify only through factual correctness. If the experiment has fewer than two failures, use a new question set and directory. Do not fabricate grades or failures. The code validates review consistency, not the truth of the reviewer's decisions.

## Verification performed

All 26 automated tests passed. The real Nomic retriever encoded the 441 chunks and produced ten rankings (five questions for each retriever). The real Qwen 1.5B model generated ten answers on CPU using the same settings in both conditions. All prompts fit the input budget, and no answer hit the output-token limit.

The executed notebook ran all nine code cells without errors and includes the ten complete model answers, both retrieval rankings, exact prompts and local reference documents. The draft PDF was rendered and visually checked: two A4 pages, embedded body/heading fonts, no clipping or overlap. Final report export was also exercised with explicitly synthetic human-review fixtures in a temporary test directory; those fixtures are not assignment grades.

The supplied real experiment has 51 pending human-review fields. All five RAG answers omitted chunk citations despite the prompt instruction; this is visible in the logs and notebook, not repaired or hidden. Reviewers must grade the answer facts, evidence support and citation semantics independently. The final assignment report has not been generated because human evaluation remains pending.
