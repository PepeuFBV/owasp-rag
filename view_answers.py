import json
from pathlib import Path
import html
import webbrowser

def main():
    jsonl_path = Path("output/experiment/answers.jsonl")
    if not jsonl_path.exists():
        print(f"Error: Could not find {jsonl_path}")
        return

    # Read and group runs by question_id
    questions_map = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            run = json.loads(line)
            qid = run["question_id"]
            if qid not in questions_map:
                questions_map[qid] = {
                    "question": run["question"],
                    "runs": {}
                }
            questions_map[qid]["runs"][run["condition"]] = run

    # Build clean HTML report
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>OWASP RAG - Answers Explorer</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 0; padding: 20px; background: #f4f6f9; color: #333; }
        .container { max-width: 1300px; margin: auto; }
        h1 { color: #1a202c; border-bottom: 2px solid #cbd5e0; padding-bottom: 10px; margin-bottom: 25px; }
        .card { background: white; border-radius: 8px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
        .question-title { font-size: 1.15em; font-weight: bold; color: #1a202c; margin-bottom: 15px; background: #edf2f7; padding: 10px 15px; border-radius: 6px; }
        .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        @media (max-width: 900px) { .grid { grid-template-columns: 1fr; } }
        .column { background: #fafbfc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 15px; }
        .column h3 { margin-top: 0; font-size: 0.95em; color: #4a5568; border-bottom: 1px solid #e2e8f0; padding-bottom: 8px; text-transform: uppercase; letter-spacing: 0.05em; }
        .meta { font-size: 0.82em; color: #718096; margin-bottom: 10px; background: #edf2f7; padding: 4px 8px; border-radius: 4px; display: inline-block; }
        .output { white-space: pre-wrap; background: #fff; padding: 12px; border: 1px solid #e2e8f0; border-radius: 4px; font-size: 0.95em; line-height: 1.5; color: #2d3748; }
        details { margin-top: 12px; font-size: 0.9em; }
        summary { cursor: pointer; color: #3182ce; font-weight: 500; user-select: none; }
        .chunk { background: #fff; padding: 10px; margin-top: 6px; border-left: 3px solid #3182ce; font-size: 0.85em; font-family: monospace; white-space: pre-wrap; border: 1px solid #edf2f7; border-radius: 0 4px 4px 0; }
    </style>
</head>
<body>
<div class="container">
    <h1>🛡️ OWASP RAG Experiment: Answers Report</h1>
"""

    for qid, data in questions_map.items():
        q_text = html.escape(data["question"])
        no_ret = data["runs"].get("without_retrieval", {})
        with_ret = data["runs"].get("with_retrieval", {})

        no_ret_out = no_ret.get("output") or no_ret.get("answer") or str(no_ret)
        with_ret_out = with_ret.get("output") or with_ret.get("answer") or str(with_ret)

        html_content += f"""
    <div class="card">
        <div class="question-title">[{qid}] {q_text}</div>
        <div class="grid">
            <div class="column">
                <h3>❌ Without Retrieval</h3>
                <div><span class="meta">Model: {no_ret.get('model', 'N/A')} &nbsp;|&nbsp; Tokens: {no_ret.get('output_tokens', 'N/A')}</span></div>
                <div class="output">{html.escape(no_ret_out)}</div>
            </div>
            <div class="column">
                <h3>✅ With Retrieval</h3>
                <div><span class="meta">Model: {with_ret.get('model', 'N/A')} &nbsp;|&nbsp; Retriever: {with_ret.get('retriever', 'N/A')} (k={with_ret.get('k', 0)}) &nbsp;|&nbsp; Tokens: {with_ret.get('output_tokens', 'N/A')}</span></div>
                <div class="output">{html.escape(with_ret_out)}</div>
"""
        
        chunks = with_ret.get("retrieved_chunks", [])
        if chunks:
            html_content += f"""
                <details>
                    <summary>View Retrieved Chunks ({len(chunks)})</summary>
"""
            for i, chunk in enumerate(chunks, 1):
                html_content += f'<div class="chunk"><strong>Chunk {i}:</strong><br>{html.escape(str(chunk))}</div>'
            html_content += "</details>"
        
        html_content += """
            </div>
        </div>
    </div>
"""

    html_content += """
</div>
</body>
</html>
"""

    output_html_path = Path("output/experiment/answers_report.html")
    output_html_path.write_text(html_content, encoding="utf-8")
    
    print(f"✅ Report generated successfully at: {output_html_path}")
    
    # Automatically open in default web browser
    webbrowser.open(output_html_path.absolute().as_uri())

if __name__ == "__main__":
    main()