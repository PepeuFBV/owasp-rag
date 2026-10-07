import json
import re
import statistics
import uuid
from pathlib import Path

from config import PROCESSED_DIR, RAW_DIR, TARGET_DOCUMENTS


def clean_markdown(text: str) -> str:
    # normalize line endings and trailing spaces without changing Markdown, HTML, code, or technical operators.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip(" \t") for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def title_from_filename(path: Path) -> str:
    title = path.stem
    title = title.replace("_Cheat_Sheet", "")
    title = title.replace("_", " ")
    return title


def doc_id_from_filename(path: Path) -> str:
    value = path.stem.lower()
    value = re.sub(r"[^a-z0-9]+", "_", value).strip("_")
    return f"owasp_{value}"


def uuid_from_doc_id(doc_id: str) -> str:
    """Return a stable UUID for a document across dataset rebuilds."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"owasp-rag:{doc_id}"))


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    documents = []

    for path in sorted(RAW_DIR.glob("*.md")):
        raw_text = path.read_text(encoding="utf-8")
        text = clean_markdown(raw_text)
        word_count = len(text.split())
        doc_id = doc_id_from_filename(path)

        documents.append(
            {
                "doc_id": doc_id,
                "uuid": uuid_from_doc_id(doc_id),
                "title": title_from_filename(path),
                "source_file": path.name,
                "source_url": (
                    "https://github.com/OWASP/CheatSheetSeries/blob/master/"
                    f"cheatsheets/{path.name}"
                ),
                "word_count": word_count,
                "text": text,
            }
        )

    if len(documents) < TARGET_DOCUMENTS:
        raise RuntimeError(
            f"Only {len(documents)} documents are available; "
            f"{TARGET_DOCUMENTS} are required."
        )

    counts = [doc["word_count"] for doc in documents]
    median_length = statistics.median(counts)

    selected = sorted(
        documents,
        key=lambda doc: abs(doc["word_count"] - median_length),
    )[:TARGET_DOCUMENTS]

    selected.sort(key=lambda doc: doc["title"])

    dataset = {
        "name": "OWASP Cheat Sheet Series - Application Security Subset",
        "language": "en",
        "document_count": len(selected),
        "selection_method": (
            f"{TARGET_DOCUMENTS} documents with cleaned word counts closest "
            "to the candidate-pool median"
        ),
        "candidate_median_word_count": median_length,
        "documents": selected,
    }

    output_path = PROCESSED_DIR / "documents.json"
    output_path.write_text(
        json.dumps(dataset, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    stats = {
        "candidate_documents": len(documents),
        "selected_documents": len(selected),
        "candidate_median_word_count": median_length,
        "selected_min_word_count": min(doc["word_count"] for doc in selected),
        "selected_max_word_count": max(doc["word_count"] for doc in selected),
        "selected_median_word_count": statistics.median(
            doc["word_count"] for doc in selected
        ),
    }

    stats_path = PROCESSED_DIR / "stats.json"
    stats_path.write_text(
        json.dumps(stats, indent=2),
        encoding="utf-8",
    )

    print("\nSelected documents:\n")
    for doc in sorted(selected, key=lambda item: item["word_count"]):
        print(f'{doc["word_count"]:5d}  {doc["title"]}')

    print("\nDataset stats:")
    for key, value in stats.items():
        print(f"{key}: {value}")

    print(f"\nSaved dataset to {output_path}")


if __name__ == "__main__":
    main()
