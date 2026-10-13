import copy
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from generate_answers import LocalLLM, build_messages, inspect_citations
from evaluate_answers import (
    ANSWER_FIELDS, RETRIEVAL_FIELDS, classify_failure, read_evaluation,
    write_evaluation_templates,
)
from run_experiment import load_experiment, load_questions, run_experiment


class FakeGenerator:
    model_name = "test/local-model"
    model_revision = "test-revision"
    settings = {
        "max_new_tokens": 256, "max_input_tokens": 4096,
        "do_sample": False, "seed": 42, "device": "cpu", "dtype": "torch.float32",
    }

    def __init__(self):
        self.calls = []

    def generate(self, question, chunks):
        self.calls.append((question, copy.deepcopy(chunks)))
        answer = "Use parameters." + (" [sql_c000]" if chunks else "")
        return {
            "answer": answer, "messages": build_messages(question, chunks),
            "prompt": "exact test prompt", "input_tokens": 50, "output_tokens": 10,
            "citation_check": inspect_citations(answer, chunks),
        }


class RagPipelineTests(unittest.TestCase):
    def setUp(self):
        self.chunk = {
            "chunk_id": "sql_c000", "doc_id": "sql", "title": "SQL prevention",
            "source_url": "https://example.test/sql", "text": "Use parameterized queries.",
        }
        self.questions = [{"question_id": "q1", "question": "How to prevent SQL injection?",
                           "reference_answer": "Use parameters."}]
        self.rankings = [
            {"question_id": "q1", "question": self.questions[0]["question"],
             "retriever": name, "k": 1, "chunks": [{**self.chunk, "rank": 1, "score": 0.8}]}
            for name in ["tfidf", "dense"]
        ]

    def run_saved_experiment(self, directory, generator=None):
        generator = generator or FakeGenerator()
        with patch("run_experiment.load_chunks", return_value=[self.chunk]):
            return run_experiment(self.questions, generator, k=1,
                                  output_dir=directory, rankings=self.rankings)

    def test_prompt_contains_exact_evidence_but_not_reference_answer(self):
        messages = build_messages(self.questions[0]["question"], [self.chunk])
        payload = json.loads(messages[1]["content"])
        self.assertEqual(payload["sources"][0]["text"], self.chunk["text"])
        self.assertNotIn("reference_answer", payload)
        self.assertIn("untrusted", messages[0]["content"])
        self.assertEqual(json.loads(build_messages("Question?", [])[1]["content"])["sources"], [])

    def test_citation_id_check_does_not_assign_semantic_grades(self):
        check = inspect_citations("Claim [sql_c000], wrong ID [invented], repeat [sql_c000]", [self.chunk])
        self.assertEqual(check["cited_chunk_ids"], ["sql_c000", "invented"])
        self.assertEqual(check["unknown_citation_ids"], ["invented"])
        self.assertNotIn("correct", check)

    def test_duplicate_context_ids_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            build_messages("Question?", [self.chunk, self.chunk])

    def test_question_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "questions.json"
            path.write_text(json.dumps(self.questions * 2), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unique"):
                load_questions(path)

    def test_paired_runs_keep_model_settings_and_evidence_consistent(self):
        with tempfile.TemporaryDirectory() as directory:
            generator = FakeGenerator()
            runs = self.run_saved_experiment(directory, generator)
            self.assertEqual(len(runs), 2)
            self.assertEqual(generator.calls[0][1], [])
            self.assertEqual(generator.calls[1][1], self.rankings[1]["chunks"])
            self.assertEqual(runs[0]["generation_settings"], runs[1]["generation_settings"])
            manifest = json.loads((Path(directory) / "experiment.json").read_text())
            self.assertEqual(manifest["status"], "completed")
            self.assertTrue((Path(directory) / "retrievals.json").exists())

    def test_does_not_overwrite_experiment_or_manual_reviews(self):
        with tempfile.TemporaryDirectory() as directory:
            runs = self.run_saved_experiment(directory)
            path = Path(directory) / "answer_grades.csv"
            path.write_text("manual review preserved", encoding="utf-8")
            write_evaluation_templates(runs, self.rankings, directory)
            self.assertEqual(path.read_text(), "manual review preserved")
            with self.assertRaises(FileExistsError):
                self.run_saved_experiment(directory)

    def test_failed_generation_keeps_evidence_and_partial_log(self):
        generator = FakeGenerator()
        original_generate = generator.generate

        def fail_second(question, chunks):
            if chunks:
                raise RuntimeError("test failure")
            return original_generate(question, chunks)

        generator.generate = fail_second
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "test failure"):
                self.run_saved_experiment(directory, generator)
            manifest = json.loads((Path(directory) / "experiment.json").read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(len((Path(directory) / "answers.jsonl").read_text().splitlines()), 1)
            self.assertTrue((Path(directory) / "retrievals.json").exists())

    def test_reuse_rejects_settings_changes_or_log_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            self.run_saved_experiment(directory)
            with patch("run_experiment.load_chunks", return_value=[self.chunk]):
                runs, _ = load_experiment(directory, self.questions, "test/local-model", "dense", 1)
                self.assertEqual(len(runs), 2)
                with self.assertRaisesRegex(ValueError, "budgets"):
                    load_experiment(directory, self.questions, "test/local-model", "dense", 1,
                                    max_new_tokens=128)
                with patch("run_experiment.SYSTEM_PROMPT", "Changed experimental instructions"):
                    with self.assertRaisesRegex(ValueError, "different inputs"):
                        load_experiment(directory, self.questions, "test/local-model", "dense", 1)
                path = Path(directory) / "answers.jsonl"
                text = path.read_text().replace("Use parameters.", "Changed answer.")
                path.write_text(text, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "manifest"):
                    load_experiment(directory, self.questions, "test/local-model", "dense", 1)

    def test_rankings_must_use_saved_chunk_text(self):
        self.rankings[1]["chunks"][0]["text"] = "Changed or truncated evidence"
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "saved corpus"):
                self.run_saved_experiment(directory)

    def test_ungraded_results_cannot_be_submitted_as_final(self):
        with tempfile.TemporaryDirectory() as directory:
            self.run_saved_experiment(directory)
            result = read_evaluation(directory)
            self.assertTrue(result["pending"])
            baseline = result["answer_grades"][("q1_without_retrieval",)]
            self.assertEqual(baseline["supported"], "na")
            with self.assertRaisesRegex(ValueError, "incomplete"):
                read_evaluation(directory, require_complete=True)

    def test_failure_attribution_uses_evidence_judgment(self):
        grade = {"correct": "no"}
        run = {"condition": "with_retrieval"}
        self.assertEqual(classify_failure(run, grade, {"evidence_found": "no"})[0], "retrieval")
        self.assertEqual(classify_failure(run, grade, {"evidence_found": "yes"})[0], "generation")
        citation_failure = {"correct": "yes", "supported": "yes", "citation_correct": "no"}
        self.assertEqual(classify_failure(run, citation_failure, {"evidence_found": "yes"})[0], "generation")
        self.assertEqual(classify_failure({"condition": "without_retrieval"}, grade)[0], "generation")
        with self.assertRaises(ValueError):
            classify_failure(run, grade, {"evidence_found": ""})

    def test_oversized_prompt_fails_before_model_generation(self):
        import torch

        generator = LocalLLM.__new__(LocalLLM)
        generator.tokenizer = Mock()
        generator.tokenizer.apply_chat_template.return_value = "oversized prompt"
        generator.tokenizer.return_value = {"input_ids": torch.ones((1, 11), dtype=torch.long)}
        generator.model = Mock()
        generator.settings = {"max_input_tokens": 10}
        with self.assertRaisesRegex(ValueError, "no retrieved evidence was silently truncated"):
            generator.generate("Question?", [self.chunk])
        generator.model.generate.assert_not_called()

    def test_complete_manual_review_exports_two_page_report(self):
        from build_report import build_report

        self.questions = [{**self.questions[0], "question_id": f"q{i}"} for i in range(1, 6)]
        self.rankings = [{**ranking, "question_id": question["question_id"]}
                         for question in self.questions for ranking in self.rankings]
        with tempfile.TemporaryDirectory() as directory:
            runs = self.run_saved_experiment(directory)
            output_dir = Path(directory)
            rows = []
            for run in runs:
                baseline = run["condition"] == "without_retrieval"
                rows.append({
                    "run_id": run["run_id"], "question_id": run["question_id"],
                    "condition": run["condition"],
                    "correct": "no" if run["run_id"] in {"q1_with_retrieval", "q2_with_retrieval"} else "yes",
                    "supported": "na" if baseline else "yes",
                    "citation_correct": "na" if baseline else "yes",
                    "evidence_chunk_ids": "" if baseline else "sql_c000",
                    "notes": "Synthetic reviewer fixture for export validation, not a real experiment grade.",
                })
            with (output_dir / "answer_grades.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=ANSWER_FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            with (output_dir / "retrieval_grades.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=RETRIEVAL_FIELDS)
                writer.writeheader()
                writer.writerows({
                    "question_id": ranking["question_id"], "retriever": ranking["retriever"],
                    "k": 1, "evidence_found": "yes", "evidence_chunk_ids": "sql_c000",
                    "notes": "Synthetic retrieval review fixture.",
                } for ranking in self.rankings)
            review_path = output_dir / "review.json"
            review = json.loads(review_path.read_text())
            review["failure_run_ids"] = ["q1_with_retrieval", "q2_with_retrieval"]
            review_path.write_text(json.dumps(review), encoding="utf-8")
            evaluation = read_evaluation(output_dir, require_complete=True)
            self.assertEqual(evaluation["pending"], [])
            path = build_report(output_dir)
            self.assertEqual(path.name, "report.pdf")
            self.assertTrue(path.read_bytes().startswith(b"%PDF"))


if __name__ == "__main__":
    unittest.main()
