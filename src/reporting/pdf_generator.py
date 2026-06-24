from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer
)

from reportlab.lib.styles import getSampleStyleSheet

def generate_pdf_report(
    filename,
    report_text
):
    doc = SimpleDocTemplate(filename)

    styles = getSampleStyleSheet()

    story = []

    story.append(
        Paragraph(
            "AI Maintenance Report",
            styles["Title"]
        )
    )

    story.append(Spacer(1,12))

    story.append(
        Paragraph(
            report_text.replace("\n","<br/>"),
            styles["BodyText"]
        )
    )

    doc.build(story)

    print(
        f"PDF saved: {filename}"
    )