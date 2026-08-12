import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

def generate_pdf_report(output_path=r"D:\FUTURE_ML_O3\report\FUTURE_ML_O3_Project_Report.pdf"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"[ReportGenerator] Starting report compilation to: {output_path}...")
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=54,
        leftMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    
    styles = getSampleStyleSheet()
    
    # Custom Palette
    primary_color = colors.HexColor("#1A365D")   # Dark Blue
    secondary_color = colors.HexColor("#2B6CB0") # Medium Blue
    accent_color = colors.HexColor("#2D3748")    # Charcoal
    text_color = colors.HexColor("#2D3748")
    
    # Custom Typography Styles
    title_style = ParagraphStyle(
        "CoverTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=26,
        leading=32,
        textColor=primary_color,
        spaceAfter=15
    )
    
    subtitle_style = ParagraphStyle(
        "CoverSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=13,
        leading=18,
        textColor=secondary_color,
        spaceAfter=40,
        alignment=1 # Center
    )
    
    h1_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=primary_color,
        spaceBefore=15,
        spaceAfter=10,
        keepWithNext=True
    )
    
    h2_style = ParagraphStyle(
        "SubsectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=16,
        textColor=secondary_color,
        spaceBefore=10,
        spaceAfter=6,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=text_color,
        spaceAfter=10
    )
    
    meta_style = ParagraphStyle(
        "MetadataText",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=accent_color,
        alignment=1 # Center
    )
    
    bullet_style = ParagraphStyle(
        "ReportBullet",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=text_color,
        leftIndent=20,
        firstLineIndent=-10,
        spaceAfter=5
    )
    
    story = []
    
    # ==========================================
    # 1. COVER PAGE / HEADER BLOCK
    # ==========================================
    story.append(Spacer(1, 1.5 * inch))
    story.append(Paragraph("AI-Powered Resume Screening & Candidate Ranking System", title_style))
    story.append(Paragraph("Natural Language Processing (NLP) & Supervised Skill Alignment Pipeline", subtitle_style))
    story.append(Spacer(1, 1.5 * inch))
    
    # Metadata Box
    meta_data = [
        [Paragraph("<b>Internship Task:</b> Future Interns Machine Learning Internship — Task 3", meta_style)],
        [Paragraph("<b>Project Identifier:</b> FUTURE_ML_O3", meta_style)],
        [Paragraph("<b>Date of Submission:</b> August 2026", meta_style)],
        [Paragraph("<b>Author:</b> Machine Learning Intern", meta_style)]
    ]
    t = Table(meta_data, colWidths=[5 * inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#F7FAFC")),
        ('BOX', (0,0), (-1,-1), 1, colors.HexColor("#E2E8F0")),
        ('PADDING', (0,0), (-1,-1), 12),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE')
    ]))
    story.append(t)
    
    story.append(PageBreak())
    
    # ==========================================
    # 2. INTRODUCTION & BUSINESS PROBLEM
    # ==========================================
    story.append(Paragraph("1. Executive Summary & Business Context", h1_style))
    story.append(Paragraph(
        "In modern recruitment, Human Resource departments are overwhelmed by the sheer volume of resumes submitted for open positions. "
        "Manual resume screening is time-consuming, prone to cognitive fatigue, and introduces subjective biases. "
        "To optimize recruitment workflows, organizations require decision-support tools that quickly filter, analyze, and rank applicants "
        "according to job-specific requirements.",
        body_style
    ))
    story.append(Paragraph(
        "This project, developed under the <b>Future Interns Machine Learning Internship (Task 3)</b>, implements an end-to-end NLP and "
        "Machine Learning pipeline. The system processes raw candidate profiles, performs keyword and semantic text similarity analysis, "
        "matches technical skill sets, analyzes missing qualifications, and ranks applicants to help recruiters prioritize high-potential candidates.",
        body_style
    ))
    
    story.append(Spacer(1, 10))
    
    # ==========================================
    # 3. TECHNICAL METHODOLOGY
    # ==========================================
    story.append(Paragraph("2. Technical Architecture & Methodology", h1_style))
    story.append(Paragraph(
        "The application architecture consists of five core stages:",
        body_style
    ))
    story.append(Paragraph("&bull; <b>Data Extraction:</b> Ingestion of the Kaggle Resume Dataset (2,400+ resumes scraped from livecareer.com) and anonymization of profiles with CAND_XXX IDs.", bullet_style))
    story.append(Paragraph("&bull; <b>NLP Preprocessing:</b> Lowercasing, removal of HTML tags, URLs, email addresses, phone numbers, and custom tokenization rules designed to preserve critical programming tags (e.g., C++, .NET, C#) that standard tokenizers corrupt.", bullet_style))
    story.append(Paragraph("&bull; <b>Supervised Skill Extraction:</b> Regex-based dictionary lookup using a configurable technical skill dictionary of 40+ concepts grouped into distinct categories (Databases, Programming, BI, Cloud, etc.).", bullet_style))
    story.append(Paragraph("&bull; <b>TF-IDF Vectorization & Similarity:</b> A TfidfVectorizer fits on the preprocessed corpus and transforms resumes and job descriptions. Textual Cosine Similarity is calculated as a baseline matching score.", bullet_style))
    story.append(Paragraph("&bull; <b>Transparent Hybrid Scoring:</b> Candidates are ranked using a weighted fit score: 60% TF-IDF Cosine Similarity and 40% Exact Skill Match percentage.", bullet_style))
    
    story.append(Spacer(1, 10))
    
    # ==========================================
    # 4. ALIGNMENT FORMULAS
    # ==========================================
    story.append(Paragraph("3. Scoring & Skill Matching Formulas", h1_style))
    story.append(Paragraph(
        "Rather than relying on closed-source algorithms, this project uses transparent and explainable equations:",
        body_style
    ))
    story.append(Paragraph(
        "<b>Skill Match Percentage:</b><br/>"
        "$$\\text{Skill Match \\%} = \\left( \\frac{\\text{Number of Matched Required Skills}}{\\text{Total Number of Required Skills}} \\right) \\times 100$$",
        body_style
    ))
    story.append(Paragraph(
        "<b>Final Candidate Score:</b><br/>"
        "$$\\text{Final Score} = (0.60 \\times \\text{Cosine Similarity \\%}) + (0.40 \\times \\text{Skill Match \\%})$$",
        body_style
    ))
    
    story.append(PageBreak())
    
    # ==========================================
    # 5. ETHICAL CONSIDERATIONS & LIMITATIONS
    # ==========================================
    story.append(Paragraph("4. Ethical Considerations & Limitations", h1_style))
    story.append(Paragraph(
        "<b>Recruiter Decision-Support:</b> The screening system is designed explicitly as a decision-support assistant. "
        "It is <b>not</b> an automated decision-making system. The final hiring and evaluation steps must remain under human control "
        "to prevent bias and ensure compliance with employment legislation.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Biases & Fairness:</b> Keyword and TF-IDF similarity algorithms rank resumes based on wording. Candidates who describe their "
        "skills using different terminology or who have non-traditional career paths might receive lower scores, even if highly qualified. "
        "Hiring managers must ensure human oversight to verify rankings and avoid discriminating against non-standard resume designs.",
        body_style
    ))
    story.append(Paragraph(
        "<b>System Limitations:</b> The current model performs lexical (exact keyword) matching and bag-of-words similarity. It does not "
        "fully capture semantic context (e.g., distinguishing 'learning machine learning' from 'managing a machine learning team'). "
        "Experience depth and education credentials are also not parsed in this prototype.",
        body_style
    ))
    
    story.append(Spacer(1, 10))
    
    # ==========================================
    # 6. RESULTS & INSIGHTS
    # ==========================================
    story.append(Paragraph("5. Expected Project Outcomes", h1_style))
    story.append(Paragraph(
        "Upon full pipeline execution, the system generates: "
        "ranked candidate sheets (`ranked_candidates.csv`), visual skill gap distribution analytics (`skill_gap.png`, `top_skills.png`), "
        "and launches a web dashboard utilizing Streamlit. The dashboard allows recruiters to adjust algorithm weights in real time, "
        "paste job descriptions dynamically, upload PDF resumes, and read contextual candidate fit justifications.",
        body_style
    ))
    story.append(Paragraph(
        "This project successfully combines foundational natural language processing, data analysis, and visual business reporting, "
        "making it a strong submission for the Machine Learning Internship portfolio.",
        body_style
    ))
    
    # Compile document
    doc.build(story)
    print(f"[ReportGenerator] Report successfully compiled at: {output_path}")

if __name__ == "__main__":
    generate_pdf_report()
