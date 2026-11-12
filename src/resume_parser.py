import PyPDF2
import docx
import os
import re

# ============================================================
# EXTRACTION
# ============================================================

def extract_text_from_pdf(file_path):
    text = ""
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text.strip()

def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text.strip()

def parse_resume(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext == ".docx":
        return extract_text_from_docx(file_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")


# ============================================================
# SECTION SPLITTING
# ============================================================

SECTION_HEADERS = {
    "skills":     r"(?i)^\s*(skills|technical skills|core competencies|technologies|tech stack|tools)\s*:?\s*$",
    "experience": r"(?i)^\s*(experience|work experience|professional experience|employment|work history)\s*:?\s*$",
    "projects":   r"(?i)^\s*(projects|personal projects|key projects|selected projects)\s*:?\s*$",
    "education":  r"(?i)^\s*(education|academic background|qualifications)\s*:?\s*$",
    "summary":    r"(?i)^\s*(summary|objective|profile|about me)\s*:?\s*$",
    "certifications": r"(?i)^\s*(certifications|certificates|licenses)\s*:?\s*$",
}

def split_into_sections(raw_text):
    lines = raw_text.split("\n")
    sections = {k: [] for k in SECTION_HEADERS}
    current = "summary"

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        matched = None
        for sec, pattern in SECTION_HEADERS.items():
            if re.match(pattern, stripped):
                matched = sec
                break

        if matched:
            current = matched
        else:
            sections[current].append(stripped)

    return {k: "\n".join(v).strip() for k, v in sections.items()}


def to_bullets(section_text):
    if not section_text:
        return []
    raw_bullets = re.split(
        r"(?:^|\n)\s*[•\-\*·▪◦]\s*|(?<=[.!?])\s+(?=[A-Z])",
        section_text
    )
    bullets = []
    for b in raw_bullets:
        b = b.strip().strip("•-*·▪◦").strip()
        if len(b) >= 20:
            bullets.append(b)
    return bullets


def parse_resume_structured(file_path):
    raw_text = parse_resume(file_path)
    sections = split_into_sections(raw_text)
    return {
        "skills":         sections["skills"],
        "experience":     to_bullets(sections["experience"]),
        "projects":       to_bullets(sections["projects"]),
        "education":      sections["education"],
        "summary":        sections["summary"],
        "certifications": sections["certifications"],
        "raw_text":       raw_text,
    }


# ============================================================
# TEST BLOCK
# ============================================================

if __name__ == "__main__":
    # The folder where you said the resume is
    folder_path = r"C:\Samad\SundarPitchTheAi\Projects\semantic-resume-scorer\Sample space\Simulated SS"
    
    print("=" * 60)
    print("CHECKING FOLDER CONTENTS...")
    print("=" * 60)
    
    try:
        files = os.listdir(folder_path)
        print(f"Files actually in that folder:\n{files}\n")
        
        # Find the PDF automatically
        pdf_file = next((f for f in files if f.lower().endswith('.pdf')), None)
        
        if not pdf_file:
            print("❌ No PDF found in that folder! Please check the folder path.")
            exit(1)
            
        print(f"✅ Found PDF: {pdf_file}")
        
        # Build the exact path
        exact_path = os.path.join(folder_path, pdf_file)
        print(f"Full path: {exact_path}\n")
        
    except FileNotFoundError:
        print(f"❌ FOLDER NOT FOUND: {folder_path}")
        print("Check if the folder name 'Sample space' or 'Simulated SS' has a typo.")
        exit(1)

    # Parse using the exact path we just found
    parsed = parse_resume_structured(exact_path)

    print("\n--- SKILLS ---")
    print(parsed["skills"][:500] if parsed["skills"] else "(EMPTY)")

    print("\n--- EXPERIENCE BULLETS ---")
    if parsed["experience"]:
        for i, b in enumerate(parsed["experience"][:5], 1):
            print(f"{i}. {b[:200]}")
    else:
        print("(EMPTY)")

    print("\n--- PROJECT BULLETS ---")
    if parsed["projects"]:
        for i, b in enumerate(parsed["projects"][:5], 1):
            print(f"{i}. {b[:200]}")
    else:
        print("(EMPTY)")

    print("\n--- EDUCATION ---")
    print(parsed["education"][:300] if parsed["education"] else "(EMPTY)")

    print("\n--- SUMMARY ---")
    print(parsed["summary"][:300] if parsed["summary"] else "(EMPTY)")

    print("\n--- COUNTS ---")
    print(f"Skills chars:       {len(parsed['skills'])}")
    print(f"Experience bullets: {len(parsed['experience'])}")
    print(f"Project bullets:    {len(parsed['projects'])}")
    print(f"Raw text length:    {len(parsed['raw_text'])}")