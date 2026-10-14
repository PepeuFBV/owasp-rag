"""Compare retrievers and log paired LLM runs without overwriting reviews."""

import argparse
import hashlib
import importlib.metadata
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from config import EXPERIMENT_DIR, QUESTIONS_FILE
from generate_answers import LocalLLM, MAX_INPUT_TOKENS, MAX_NEW_TOKENS, MODEL_NAME, SYSTEM_PROMPT
from retrieve_tfidf import load_chunks


def fingerprint(value):
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_questions(path=QUESTIONS_FILE):
    questions = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(questions, list) or not questions:
        raise ValueError("Questions must be a non-empty JSON array")
    ids = []
    for question in questions:
        if not isinstance(question, dict):
            raise ValueError("Every question must be an object")
        for field in ["question_id", "question", "reference_answer"]:
            if not isinstance(question.get(field), str) or not question[field].strip():
                raise ValueError(f"Each question needs a non-empty {field}")
        if not re.fullmatch(r"[A-Za-z0-9_-]+", question["question_id"]):
            raise ValueError("question_id must contain only letters, digits, _ or -")
        ids.append(question["question_id"])
    if len(ids) != len(set(ids)):
        raise ValueError("Question IDs must be unique")
    return questions


def retrieve_chunks(retriever, question, k):
    if retriever == "tfidf":
        from retrieve_tfidf import retrieve
    elif retriever == "dense":
        from retrieve_dense import retrieve
    else:
        raise ValueError("retriever must be tfidf or dense")
    return retrieve(question, k=k)


def compare_retrievers(questions, k=3):
    return [
        {
            "question_id": question["question_id"],
            "question": question["question"],
            "retriever": retriever,
            "k": k,
            "chunks": retrieve_chunks(retriever, question["question"], k),
        }
        for question in questions
        for retriever in ["tfidf", "dense"]
    ]


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def load_runs(path):
    runs = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]
    ids = [run["run_id"] for run in runs]
    if not runs or len(ids) != len(set(ids)):
        raise ValueError("Run log must be non-empty and contain unique run IDs")
    return runs


def run_experiment(questions, generator, retriever="dense", k=3,
                   output_dir=EXPERIMENT_DIR, rankings=None):
    if retriever not in {"tfidf", "dense"}:
        raise ValueError("retriever must be tfidf or dense")
    if type(k) is not int or k < 1:
        raise ValueError("k must be a positive integer")
    if not questions:
        raise ValueError("At least one question is required")
    question_ids = [question["question_id"] for question in questions]
    if len(question_ids) != len(set(question_ids)):
        raise ValueError("Question IDs must be unique")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "experiment.json"
    if (output_dir / "answers.jsonl").exists():
        raise FileExistsError("Answers already exist. Use a new output directory to preserve grading.")
    # Exclusive creation protects against two experiment processes using the same directory.
    with manifest_path.open("x", encoding="utf-8") as stream:
        stream.write("{}")
    manifest = {
        "schema_version": 1,
        "status": "running",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": generator.model_name,
        "model_revision": generator.model_revision,
        "generation_settings": generator.settings,
        "retriever": retriever,
        "k": k,
        "question_count": len(questions),
        "expected_answers": 2 * len(questions),
        "questions_fingerprint": fingerprint(questions),
        "corpus_fingerprint": fingerprint(load_chunks()),
        "prompt_policy_fingerprint": fingerprint(SYSTEM_PROMPT),
        "versions": {name: importlib.metadata.version(name) for name in
                     ["torch", "transformers", "sentence-transformers", "scikit-learn"]},
    }
    save_json(manifest_path, manifest)
    runs = []
    try:
        rankings = compare_retrievers(questions, k) if rankings is None else rankings
        ranking_map = {(row["question_id"], row["retriever"]): row for row in rankings}
        if len(ranking_map) != 2 * len(questions) or any(
            (qid, name) not in ranking_map for qid in question_ids for name in ["tfidf", "dense"]
        ):
            raise ValueError("Rankings must contain both retrievers for every question")
        corpus_by_id = {chunk["chunk_id"]: chunk for chunk in load_chunks()}
        for row in rankings:
            chunks = row["chunks"]
            if (row["k"] != k or row["question"] != next(
                q["question"] for q in questions if q["question_id"] == row["question_id"]
            ) or len(chunks) != min(k, len(corpus_by_id))):
                raise ValueError("Ranking settings do not match the experiment")
            if len({chunk["chunk_id"] for chunk in chunks}) != len(chunks):
                raise ValueError("Ranking contains duplicate chunk IDs")
            for chunk in chunks:
                saved = corpus_by_id.get(chunk["chunk_id"])
                if saved is None or any(chunk[field] != saved[field] for field in
                                        ["text", "doc_id", "title", "source_url"]):
                    raise ValueError("Ranking evidence does not match the saved corpus")
        # Save evidence before the first LLM call; even failed generations remain auditable.
        save_json(output_dir / "retrievals.json", rankings)
        save_json(output_dir / "questions.json", questions)
        with (output_dir / "answers.jsonl").open("x", encoding="utf-8") as stream:
            for question in questions:
                for condition in ["without_retrieval", "with_retrieval"]:
                    chunks = ([] if condition == "without_retrieval" else
                              ranking_map[(question["question_id"], retriever)]["chunks"])
                    result = generator.generate(question["question"], chunks)
                    run = {
                        "run_id": f"{question['question_id']}_{condition}",
                        "question_id": question["question_id"],
                        "question": question["question"],
                        "condition": condition,
                        "model": generator.model_name,
                        "model_revision": generator.model_revision,
                        "generation_settings": generator.settings,
                        "retriever": retriever if chunks else "none",
                        "k": k if chunks else 0,
                        "retrieved_chunks": chunks,
                        **result,
                    }
                    stream.write(json.dumps(run, ensure_ascii=False) + "\n")
                    stream.flush()
                    runs.append(run)
                    print(f"{run['run_id']}: {run['output_tokens']} output tokens", flush=True)
        manifest["status"] = "completed"
        manifest["answers_fingerprint"] = fingerprint(runs)
        manifest["retrievals_fingerprint"] = fingerprint(rankings)
    except Exception:
        manifest["status"] = "failed"
        raise
    finally:
        save_json(manifest_path, manifest)
    from evaluate_answers import write_evaluation_templates

    write_evaluation_templates(runs, rankings, output_dir)
    return runs


def load_experiment(output_dir, questions, model, retriever, k,
                    max_new_tokens=MAX_NEW_TOKENS, max_input_tokens=MAX_INPUT_TOKENS):
    """Reuse only a complete, unchanged experiment with matching settings."""
    output_dir = Path(output_dir)
    manifest = json.loads((output_dir / "experiment.json").read_text(encoding="utf-8"))
    expected = {
        "model": model, "retriever": retriever, "k": k,
        "questions_fingerprint": fingerprint(questions),
        "corpus_fingerprint": fingerprint(load_chunks()),
        "prompt_policy_fingerprint": fingerprint(SYSTEM_PROMPT),
    }
    if manifest.get("status") != "completed" or any(
        manifest.get(key) != value for key, value in expected.items()
    ):
        raise ValueError("Saved experiment is incomplete or uses different inputs; use a new output directory")
    settings = manifest["generation_settings"]
    if settings["max_new_tokens"] != max_new_tokens or settings["max_input_tokens"] != max_input_tokens:
        raise ValueError("Saved experiment uses different generation budgets")
    runs = load_runs(output_dir / "answers.jsonl")
    rankings = json.loads((output_dir / "retrievals.json").read_text(encoding="utf-8"))
    if (len(runs) != 2 * len(questions)
            or fingerprint(runs) != manifest["answers_fingerprint"]
            or fingerprint(rankings) != manifest["retrievals_fingerprint"]):
        raise ValueError("Saved experiment artifacts do not match the completed manifest")
    return runs, rankings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=QUESTIONS_FILE)
    parser.add_argument("--retriever", choices=["tfidf", "dense"], default="dense")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    parser.add_argument("--max-input-tokens", type=int, default=MAX_INPUT_TOKENS)
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENT_DIR)
    args = parser.parse_args()
    if (args.output_dir / "experiment.json").exists():
        parser.error("Experiment already exists; choose a new --output-dir")
    try:
        questions = load_questions(args.questions)
        rankings = compare_retrievers(questions, args.k)
        generator = LocalLLM(args.model, args.max_new_tokens, args.max_input_tokens)
        run_experiment(questions, generator, args.retriever, args.k, args.output_dir, rankings)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
