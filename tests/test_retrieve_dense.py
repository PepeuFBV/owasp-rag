
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import retrieve_dense


def sample_chunks():
    return [
        {
            "chunk_id": f"chunk_{i}",
            "doc_id": f"doc_{i}",
            "title": title,
            "source_url": f"https://example.test/{i}",
            "text": text,
        }
        for i, (title, text) in enumerate([
            ("SQL Injection", "Parameterized queries prevent injection"),
            ("CSRF", "CSRF tokens protect sessions"),
            ("Password Storage", "Password hashing protects credentials"),
        ])
    ]


class DenseRetrievalTests(unittest.TestCase):
    def setUp(self):
        self.chunks = sample_chunks()

        self.model = Mock()
        self.model.encode.return_value = np.array(
            [[1.0, 0.0]], dtype=np.float32
        )

        self.index = {
            "chunks": self.chunks,
            "model": self.model,
            "embeddings": np.array(
                [[1.0, 0.0], [0.0, 1.0], [0.8, 0.6]],
                dtype=np.float32,
            ),
        }

    def test_ranking_and_metadata(self):
        with patch.object(retrieve_dense, "_INDEX", self.index):
            results = retrieve_dense.retrieve("SQL injection", k=2)

        self.assertEqual(len(results), 2)
        self.assertEqual(
            [result["chunk_id"] for result in results],
            ["chunk_0", "chunk_2"],
        )
        self.assertEqual([r["rank"] for r in results], [1, 2])
        self.assertGreaterEqual(results[0]["score"], results[1]["score"])

        for result in results:
            self.assertTrue({
                "rank", "chunk_id", "doc_id",
                "title", "score", "text", "source_url",
            }.issubset(result))

    def test_invalid_inputs(self):
        with patch.object(retrieve_dense, "_INDEX", self.index):
            for k in (0, -1, 1.5, True, "3"):
                with self.subTest(k=k):
                    with self.assertRaises(ValueError):
                        retrieve_dense.retrieve("SQL injection", k=k)

            for question in ("", "   ", None, 123):
                with self.subTest(question=question):
                    with self.assertRaises(ValueError):
                        retrieve_dense.retrieve(question)

    def test_cache_reuse_and_invalidation(self):
        self.model.get_sentence_embedding_dimension.return_value = 2
        self.model.encode.return_value = np.ones(
            (len(self.chunks), 2), dtype=np.float32
        )

        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "dense_index.npz"

            with patch.object(retrieve_dense, "CACHE_FILE", cache):
                first = retrieve_dense.load_or_generate_embeddings(
                    self.chunks, self.model
                )
                second = retrieve_dense.load_or_generate_embeddings(
                    self.chunks, self.model
                )

                self.assertEqual(self.model.encode.call_count, 1)
                np.testing.assert_array_equal(first, second)

                changed = [dict(chunk) for chunk in self.chunks]
                changed[0]["text"] = "Updated technical content"

                retrieve_dense.load_or_generate_embeddings(
                    changed, self.model
                )

                self.assertEqual(self.model.encode.call_count, 2)


if __name__ == "__main__":
    unittest.main()
