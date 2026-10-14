"""Build a two-page report from logged experiments and validated human grades."""

import argparse
from pathlib import Path
from xml.sax.saxutils import escape

from config import EXPERIMENT_DIR
from evaluate_answers import classify_failure, read_evaluation


def build_report(output_dir=EXPERIMENT_DIR, draft=False):
    import reportlab
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    output_dir = Path(output_dir)
    evaluation = read_evaluation(output_dir, require_complete=not draft)
    if len(evaluation["runs"]) != 10:
        raise ValueError("The assignment report requires exactly 10 answers; run five paired questions")
    review = evaluation["review"]
    if any(len(review.get(field, "")) > 1100 for field in ["risk_description", "mitigation"]):
        raise ValueError("Shorten the risk and mitigation to at most 1,100 characters each")
    # Embed bundled fonts so readers do not depend on machine-specific substitutions.
    font_dir = Path(reportlab.__file__).parent / "fonts"
    pdfmetrics.registerFont(TTFont("ReportSans", str(font_dir / "Vera.ttf")))
    pdfmetrics.registerFont(TTFont("ReportSansBold", str(font_dir / "VeraBd.ttf")))
    styles = getSampleStyleSheet()
    for name in ["Title", "Heading2", "Heading3"]:
        styles[name].fontName = "ReportSansBold"
    styles["Title"].fontSize = 17
    styles["Title"].leading = 21
    styles["Heading2"].fontSize = 12
    styles["Heading2"].leading = 16
    styles["Heading3"].fontSize = 10
    styles["Heading3"].leading = 13
    styles.add(ParagraphStyle("BodySmall", fontName="ReportSans", fontSize=9, leading=12, spaceAfter=6))
    styles.add(ParagraphStyle("CellSmall", fontName="ReportSans", fontSize=7.5, leading=10))
    styles.add(ParagraphStyle("CenterSmall", parent=styles["CellSmall"], alignment=TA_CENTER))

    def paragraph(text, style="BodySmall"):
        return Paragraph(escape(str(text)).replace("\n", "<br/>"), styles[style])

    def table(rows, widths):
        result = Table([[paragraph(value, "CellSmall") for value in row] for row in rows],
                       colWidths=widths, repeatRows=1, hAlign="LEFT")
        result.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e7edf4")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#bac4cf")),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        return result

    manifest = evaluation["manifest"]
    settings = manifest["generation_settings"]
    title = "OWASP RAG - " + ("DRAFT: manual grading pending" if draft else "evaluation report")
    story = [Paragraph(title, styles["Title"])]
    story.append(paragraph(
        f"Model: {manifest['model']}. Retriever: {manifest['retriever']}; k={manifest['k']}. "
        f"Greedy decoding; seed={settings['seed']}; output budget={settings['max_new_tokens']} tokens. "
        "The same model, system instructions and decoding settings are used in both conditions. "
        "Five paired questions yield ten answers, including the three comparisons requested by item 4."
    ))
    if draft:
        story.append(paragraph(
            f"This is not a completed submission. {len(evaluation['pending'])} manual fields remain. "
            "PENDING is not a grade. N/A means no retrieval context or chunk citation was available."
        ))
    story.append(Paragraph("Answer grading (assignment item 5)", styles["Heading2"]))
    questions = {run["question_id"]: run["question"] for run in evaluation["runs"]}
    for qid, question in questions.items():
        story.append(paragraph(f"{qid}: {question}"))
    rows = [["Question", "Condition", "Correct", "Supported", "Citation", "Manual justification (excerpt)"]]
    for run in evaluation["runs"]:
        grade = evaluation["answer_grades"][(run["run_id"],)]
        rows.append([
            run["question_id"], "RAG" if run["condition"] == "with_retrieval" else "Baseline",
            grade["correct"] or "PENDING", grade["supported"] or "PENDING",
            grade["citation_correct"] or "PENDING", grade["notes"][:100] or "PENDING",
        ])
    story.append(table(rows, [16*mm, 20*mm, 18*mm, 20*mm, 18*mm, 82*mm]))
    story.append(Spacer(1, 5*mm))
    story.append(paragraph(
        "yes/no are independent manual judgments. Baseline support and citation are na (not applicable). "
        "Citation ID membership alone does not prove that the cited text supports a claim. "
        "Full answers, prompts, evidence, reference answers and review notes are preserved in the notebook and logs."
    ))
    story.append(PageBreak())
    story.append(Paragraph("Retrieval evidence and failure analysis (item 6)", styles["Heading2"]))
    retrieval_rows = [["Question", "TF-IDF: evidence in top-k?", "Dense: evidence in top-k?"]]
    for qid in questions:
        retrieval_rows.append([qid, *[
            evaluation["retrieval_grades"][(qid, name)]["evidence_found"] or "PENDING"
            for name in ["tfidf", "dense"]
        ]])
    story.append(table(retrieval_rows, [24*mm, 75*mm, 75*mm]))
    story.append(Spacer(1, 4*mm))
    run_by_id = {run["run_id"]: run for run in evaluation["runs"]}
    if not review["failure_run_ids"]:
        story.append(paragraph(
            "PENDING: select two genuinely wrong answers in review.json after manual grading. "
            "If fewer than two answers are wrong, choose additional questions and run a new experiment; "
            "do not invent errors. A RAG failure is retrieval when the required evidence is absent, "
            "and generation when the evidence is present but the answer is wrong."
        ))
    for run_id in review["failure_run_ids"]:
        run = run_by_id[run_id]
        grade = evaluation["answer_grades"][(run_id,)]
        retrieval_grade = evaluation["retrieval_grades"].get((run["question_id"], run["retriever"]))
        if retrieval_grade and retrieval_grade["evidence_found"] not in {"yes", "no"}:
            label, reason = "PENDING", "Grade retrieval evidence before attributing this failure."
        else:
            label, reason = classify_failure(run, grade, retrieval_grade)
        story.append(Paragraph(f"{escape(run_id)}: {escape(label)}", styles["Heading3"]))
        story.append(paragraph(reason))
        if len(grade["notes"]) > 950:
            raise ValueError("Shorten failure justifications to at most 950 characters")
        story.append(paragraph("Manual justification: " + (grade["notes"] or "PENDING")))
        if retrieval_grade:
            ids = retrieval_grade["evidence_chunk_ids"] or "none marked"
            story.append(paragraph("Evidence IDs: " + ids))
    story.append(Paragraph("Responsible-search risk and mitigation (item 7)", styles["Heading2"]))
    story.append(paragraph(f"Risk ({review['risk']}): {review.get('risk_description', '') or 'PENDING'}"))
    story.append(paragraph("Mitigation: " + (review.get("mitigation", "") or "PENDING")))
    story.append(paragraph(
        "Traceability: answers.jsonl contains each exact prompt and context; retrievals.json records both rankings; "
        "answer_grades.csv and retrieval_grades.csv contain the manual judgments. "
        "experiment.json stores model revision, dependency versions and content fingerprints."
    ))
    destination = output_dir / ("report_draft.pdf" if draft else "report.pdf")
    temporary = destination.with_suffix(".tmp.pdf")
    pages = []

    def footer(canvas, doc):
        pages.append(doc.page)
        canvas.setFont("ReportSans", 8)
        canvas.drawRightString(A4[0] - 18*mm, 10*mm, f"{doc.page} / 2")

    document = SimpleDocTemplate(str(temporary), pagesize=A4, rightMargin=18*mm,
                                 leftMargin=18*mm, topMargin=14*mm, bottomMargin=16*mm,
                                 title=title, author="OWASP RAG project")
    try:
        document.build(story, onFirstPage=footer, onLaterPages=footer)
        if len(pages) != 2:
            raise ValueError("Report exceeds two pages. Shorten review text before exporting.")
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENT_DIR)
    parser.add_argument("--draft", action="store_true", help="Export a clearly labeled unfinished preview")
    args = parser.parse_args()
    try:
        path = build_report(args.output_dir, args.draft)
    except ValueError as exc:
        parser.error(str(exc))
    print(f"Saved {'draft' if args.draft else 'final'} report to {path}")


if __name__ == "__main__":
    main()
