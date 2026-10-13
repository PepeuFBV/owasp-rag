import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyze_embedding_truncation import summarize_token_lengths


class EmbeddingTruncationTests(unittest.TestCase):
    def test_summarizes_chunks_over_limit_and_truncated_tokens(self):
        stats = summarize_token_lengths([100, 256, 300, 400], 256)

        self.assertEqual(stats["chunks"], 4)
        self.assertEqual(stats["chunks_over_limit"], 2)
        self.assertEqual(stats["percent_over_limit"], 50)
        self.assertEqual(stats["median_tokens"], 278)
        self.assertEqual(stats["max_tokens"], 400)
        self.assertEqual(stats["truncated_tokens_total"], 188)
        self.assertEqual(stats["max_truncated_tokens_per_chunk"], 144)

    def test_rejects_invalid_token_limit(self):
        with self.assertRaises(ValueError):
            summarize_token_lengths([1], 0)

    def test_rejects_empty_lengths(self):
        with self.assertRaises(ValueError):
            summarize_token_lengths([], 256)


if __name__ == "__main__":
    unittest.main()
