import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import retrieve_tfidf


def sample_chunks():
    return [
        {
            "chunk_id": "sql_c000",
            "doc_id": "sql",
            "title": "SQL Injection Prevention",
            "source_url": "https://example.test/sql",
            "text": "Prevent SQL injection with parameterized queries and prepared statements.",
        },
        {
            "chunk_id": "csrf_c000",
            "doc_id": "csrf",
            "title": "CSRF Prevention",
            "source_url": "https://example.test/csrf",
            "text": "Protect sessions with CSRF tokens and same-site cookies.",
        },
        {
            "chunk_id": "password_c000",
            "doc_id": "password",
            "title": "Password Storage",
            "source_url": "https://example.test/password",
            "text": "Store passwords using a salted adaptive password hash.",
        },
    ]


class LoadChunksTests(unittest.TestCase):
    def test_loads_chunks_from_json(self):
        chunks = sample_chunks()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chunks.json"
            path.write_text(json.dumps(chunks), encoding="utf-8")

            self.assertEqual(retrieve_tfidf.load_chunks(path), chunks)

    def test_rejects_chunk_missing_required_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chunks.json"
            path.write_text(json.dumps([{"chunk_id": "broken"}]), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Chunk 0 is missing"):
                retrieve_tfidf.load_chunks(path)


class TfidfRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.chunks = sample_chunks()
        self.index = retrieve_tfidf.build_index(self.chunks)

    def test_builds_tfidf_matrix_for_every_chunk(self):
        self.assertEqual(self.index["matrix"].shape[0], len(self.chunks))
        self.assertEqual(self.index["chunks"], self.chunks)
        self.assertGreater(self.index["matrix"].shape[1], 0)

    def test_ranks_matching_chunk_first_and_returns_valid_ids(self):
        with patch.object(retrieve_tfidf, "_INDEX", self.index):
            results = retrieve_tfidf.retrieve("SQL injection parameterized queries", k=3)

        self.assertEqual(results[0]["chunk_id"], "sql_c000")
        self.assertEqual(
            [result["score"] for result in results],
            sorted((result["score"] for result in results), reverse=True),
        )
        self.assertEqual(
            {result["chunk_id"] for result in results},
            {chunk["chunk_id"] for chunk in self.chunks},
        )
        self.assertEqual([result["rank"] for result in results], [1, 2, 3])

    def test_rejects_invalid_k_values(self):
        with patch.object(retrieve_tfidf, "_INDEX", self.index):
            for k in (0, -1, 1.5, True, "2"):
                with self.subTest(k=k), self.assertRaisesRegex(
                    ValueError, "k must be a positive integer"
                ):
                    retrieve_tfidf.retrieve("SQL injection", k=k)

    def test_rejects_empty_or_non_string_questions(self):
        with patch.object(retrieve_tfidf, "_INDEX", self.index):
            for question in ("", "  \n", None, 42):
                with self.subTest(question=question), self.assertRaisesRegex(
                    ValueError, "question must be a non-empty string"
                ):
                    retrieve_tfidf.retrieve(question)

    def test_rejects_questions_without_indexed_terms(self):
        with patch.object(retrieve_tfidf, "_INDEX", self.index):
            with self.assertRaisesRegex(ValueError, "no terms in the indexed corpus"):
                retrieve_tfidf.retrieve("qzxwv unknownterm")


if __name__ == "__main__":
    unittest.main()
