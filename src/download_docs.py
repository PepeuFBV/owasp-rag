import requests

from config import CANDIDATE_FILES, GITHUB_API_URL, RAW_DIR


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    response = requests.get(GITHUB_API_URL, timeout=30)
    response.raise_for_status()

    repo_files = {
        item["name"]: item["download_url"]
        for item in response.json()
        if item["type"] == "file" and item["download_url"]
    }

    downloaded = 0

    for filename in CANDIDATE_FILES:
        url = repo_files.get(filename)

        if not url:
            print(f"[skip] not found: {filename}")
            continue

        content = requests.get(url, timeout=30)
        content.raise_for_status()

        output_path = RAW_DIR / filename
        output_path.write_text(content.text, encoding="utf-8")

        downloaded += 1
        print(f"[ok] {filename}")

    print(f"\nDownloaded {downloaded} candidate documents to {RAW_DIR}")


if __name__ == "__main__":
    main()
