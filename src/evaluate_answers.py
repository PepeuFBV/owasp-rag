"""Prepare and validate manual grades; never infer correctness from citations."""

import argparse
import csv
import json
from pathlib import Path

from config import EXPERIMENT_DIR
from run_experiment import fingerprint, load_runs

ANSWER_FIELDS = [
    "run_id", "question_id", "condition", "correct", "supported",
    "citation_correct", "evidence_chunk_ids", "notes",
]
RETRIEVAL_FIELDS = [
    "question_id", "retriever", "k", "evidence_found", "evidence_chunk_ids", "notes",
]
DEFAULT_RISK = {
    "failure_run_ids": [],
    "risk": "manipulation",
    "risk_description": (
        "An indexed document can contain instructions that try to change the LLM's "
        "behavior instead of providing evidence (indirect prompt injection)."
    ),
    "mitigation": (
        "Use the public OWASP source allow-list, preserve source URLs, and delimit "
        "excerpts as untrusted JSON data. The system prompt tells the LLM to ignore "
        "instructions in excerpts. Review evidence and citations manually; these "
        "controls reduce risk but do not guarantee that prompt injection is blocked."
    ),
}


def write_csv_if_missing(path, fields, rows):
    if path.exists():
        return
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_evaluation_templates(runs, rankings, output_dir=EXPERIMENT_DIR):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    answer_rows = []
    for run in runs:
        row = {field: "" for field in ANSWER_FIELDS}
        row.update({key: run[key] for key in ["run_id", "question_id", "condition"]})
        if run["condition"] == "without_retrieval":
            row.update(supported="na", citation_correct="na")
        answer_rows.append(row)
    retrieval_rows = []
    for ranking in rankings:
        row = {field: "" for field in RETRIEVAL_FIELDS}
        row.update({key: ranking[key] for key in ["question_id", "retriever", "k"]})
        retrieval_rows.append(row)
    write_csv_if_missing(output_dir / "answer_grades.csv", ANSWER_FIELDS, answer_rows)
    write_csv_if_missing(output_dir / "retrieval_grades.csv", RETRIEVAL_FIELDS, retrieval_rows)
    review_path = output_dir / "review.json"
    if not review_path.exists():
        with review_path.open("x", encoding="utf-8") as stream:
            json.dump(DEFAULT_RISK, stream, ensure_ascii=False, indent=2)


def read_csv(path, fields):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != fields:
            raise ValueError(f"{path.name}: CSV columns must be {fields}")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"{path.name}: malformed CSV row; quote notes containing commas")
    return rows


def chunk_ids(value):
    return [item.strip() for item in value.split(";") if item.strip()]


def indexed_rows(rows, keys, expected):
    indexed = {tuple(row[key] for key in keys): row for row in rows}
    if len(indexed) != len(rows) or set(indexed) != set(expected):
        raise ValueError("Grade rows must match logged runs/rankings exactly, without duplicates")
    return indexed


def validate_choice(row, field, choices, pending, label):
    value = row[field].strip().lower()
    row[field] = value
    if not value:
        pending.append(f"{label}: {field}")
    elif value not in choices:
        raise ValueError(f"{label}: {field} must be one of {sorted(choices)}")


def is_wrong_answer(run, grade):
    if grade.get("correct") == "no":
        return True
    return run["condition"] == "with_retrieval" and any(
        grade.get(field) == "no" for field in ["supported", "citation_correct"]
    )


def read_evaluation(output_dir=EXPERIMENT_DIR, require_complete=False):
    output_dir = Path(output_dir)
    manifest = json.loads((output_dir / "experiment.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise ValueError("Finish the experiment before grading it")
    runs = load_runs(output_dir / "answers.jsonl")
    rankings = json.loads((output_dir / "retrievals.json").read_text(encoding="utf-8"))
    questions = json.loads((output_dir / "questions.json").read_text(encoding="utf-8"))
    if (len(runs) != manifest["expected_answers"]
            or fingerprint(runs) != manifest["answers_fingerprint"]
            or fingerprint(rankings) != manifest["retrievals_fingerprint"]
            or fingerprint(questions) != manifest["questions_fingerprint"]):
        raise ValueError("Experiment logs changed after generation; grades cannot be trusted")
    answers = indexed_rows(
        read_csv(output_dir / "answer_grades.csv", ANSWER_FIELDS), ["run_id"],
        {(run["run_id"],) for run in runs},
    )
    retrieval = indexed_rows(
        read_csv(output_dir / "retrieval_grades.csv", RETRIEVAL_FIELDS),
        ["question_id", "retriever"],
        {(row["question_id"], row["retriever"]) for row in rankings},
    )
    pending = []
    for ranking in rankings:
        key = (ranking["question_id"], ranking["retriever"])
        row = retrieval[key]
        label = "/".join(key)
        if row["k"] != str(ranking["k"]):
            raise ValueError(f"{label}: k differs from the logged ranking")
        validate_choice(row, "evidence_found", {"yes", "no"}, pending, label)
        selected = chunk_ids(row["evidence_chunk_ids"])
        allowed = {chunk["chunk_id"] for chunk in ranking["chunks"]}
        if len(selected) != len(set(selected)) or not set(selected) <= allowed:
            raise ValueError(f"{label}: evidence IDs must be unique members of the top-k")
        if row["evidence_found"] == "yes" and not selected:
            pending.append(f"{label}: evidence_chunk_ids")
        if not row["notes"].strip():
            pending.append(f"{label}: notes")
    for run in runs:
        row = answers[(run["run_id"],)]
        label = run["run_id"]
        if any(row[key] != run[key] for key in ["question_id", "condition"]):
            raise ValueError(f"{label}: grade metadata differs from the run")
        baseline = run["condition"] == "without_retrieval"
        validate_choice(row, "correct", {"yes", "no"}, pending, label)
        for field in ["supported", "citation_correct"]:
            validate_choice(row, field, {"na"} if baseline else {"yes", "no"}, pending, label)
        selected = chunk_ids(row["evidence_chunk_ids"])
        allowed = {chunk["chunk_id"] for chunk in run["retrieved_chunks"]}
        if len(selected) != len(set(selected)) or not set(selected) <= allowed:
            raise ValueError(f"{label}: evidence IDs must refer to the actual supplied context")
        if not baseline and row["supported"] == "yes" and not selected:
            pending.append(f"{label}: evidence_chunk_ids")
        citation_check = run["citation_check"]
        if row["citation_correct"] == "yes" and (
            not citation_check["has_citation"] or citation_check["unknown_citation_ids"]
        ):
            raise ValueError(f"{label}: cannot grade missing or unknown citations as correct")
        if not row["notes"].strip():
            pending.append(f"{label}: notes")
    review = json.loads((output_dir / "review.json").read_text(encoding="utf-8"))
    failure_ids = review.get("failure_run_ids")
    if not isinstance(failure_ids, list) or any(not isinstance(item, str) for item in failure_ids):
        raise ValueError("failure_run_ids must be a list of run ID strings")
    if len(failure_ids) != len(set(failure_ids)):
        raise ValueError("Choose two distinct wrong answers")
    run_by_id = {run["run_id"]: run for run in runs}
    for run_id in failure_ids:
        if run_id not in run_by_id or not is_wrong_answer(run_by_id[run_id], answers[(run_id,)]):
            raise ValueError("Failure analysis must select answers with a manually graded failure")
    if len(failure_ids) != 2:
        pending.append("review.json: select exactly two failure_run_ids")
    if review.get("risk") not in {"freshness", "privacy", "bias", "manipulation"}:
        raise ValueError("Choose freshness, privacy, bias or manipulation as the search risk")
    for field in ["risk_description", "mitigation"]:
        if not isinstance(review.get(field), str) or not review[field].strip():
            pending.append(f"review.json: {field}")
    if require_complete and pending:
        raise ValueError("Manual evaluation is incomplete:\n" + "\n".join(pending))
    return {
        "manifest": manifest, "runs": runs, "rankings": rankings,
        "answer_grades": answers, "retrieval_grades": retrieval,
        "review": review, "pending": pending,
    }


def classify_failure(run, answer_grade, retrieval_grade=None):
    """Attribute a wrong answer using human evidence judgments, not its score."""
    if not is_wrong_answer(run, answer_grade):
        raise ValueError("Select a wrong answer for failure analysis")
    if run["condition"] == "without_retrieval":
        return "generation", "No retrieval was performed; this is a baseline knowledge/generation failure."
    if retrieval_grade is None or retrieval_grade["evidence_found"] not in {"yes", "no"}:
        raise ValueError("Grade retrieval evidence before classifying a RAG failure")
    if retrieval_grade["evidence_found"] == "no":
        return "retrieval", "The supplied top-k did not contain all evidence needed to answer."
    return "generation", "The top-k contained the evidence, but the answer was wrong, unsupported, or cited the wrong chunk."


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENT_DIR)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    try:
        result = read_evaluation(args.output_dir, args.require_complete)
    except ValueError as exc:
        parser.error(str(exc))
    print(f"Logged answers: {len(result['runs'])}")
    print(f"Pending manual fields: {len(result['pending'])}")
    for item in result["pending"]:
        print(f"- {item}")


if __name__ == "__main__":
    main()
