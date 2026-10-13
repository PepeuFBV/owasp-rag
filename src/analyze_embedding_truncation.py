"""Measure how many saved chunks exceed the dense model's token limit."""

import statistics

from retrieve_dense import DOCUMENT_PREFIX, MODEL_NAME, load_model
from retrieve_tfidf import load_chunks


def summarize_token_lengths(lengths, token_limit):
    if token_limit < 1:
        raise ValueError("token_limit must be positive")
    if not lengths:
        raise ValueError("lengths must not be empty")

    truncated = [max(0, length - token_limit) for length in lengths]
    over_limit = sum(length > token_limit for length in lengths)
    return {
        "chunks": len(lengths),
        "token_limit": token_limit,
        "chunks_over_limit": over_limit,
        "percent_over_limit": 100 * over_limit / len(lengths),
        "median_tokens": statistics.median(lengths),
        "max_tokens": max(lengths),
        "truncated_tokens_total": sum(truncated),
        "max_truncated_tokens_per_chunk": max(truncated),
    }


def assess_chunks(chunks, model):
    lengths = [
        len(
            model.tokenizer.encode(
                f"{DOCUMENT_PREFIX}{chunk['text']}",
                add_special_tokens=True,
                truncation=False,
            )
        )
        for chunk in chunks
    ]
    return summarize_token_lengths(lengths, model.max_seq_length)


def main():
    chunks = load_chunks()
    model = load_model()
    stats = assess_chunks(chunks, model)

    print(f"Model: {MODEL_NAME}")
    print(f"Chunks: {stats['chunks']}")
    print(f"Model token limit: {stats['token_limit']}")
    print(
        f"Over limit: {stats['chunks_over_limit']} "
        f"({stats['percent_over_limit']:.2f}%)"
    )
    print(
        f"Token counts: median {stats['median_tokens']}, "
        f"maximum {stats['max_tokens']}"
    )
    print(f"Tokens beyond limits in total: {stats['truncated_tokens_total']}")
    print(
        "Largest per-chunk truncation: "
        f"{stats['max_truncated_tokens_per_chunk']} tokens"
    )


if __name__ == "__main__":
    main()
