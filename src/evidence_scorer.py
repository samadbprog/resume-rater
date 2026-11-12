import re
from resume_parser import parse_resume_structured
import os

# 1. The vague phrases to penalize
VAGUE_PATTERNS = [
    r"\bworked on\b", r"\bexposure to\b", r"\bparticipated in\b",
    r"\binvolved in\b", r"\bfamiliar with\b", r"\bhelped with\b"
]

# 2. Metric detector (AUC, %, numbers, etc.)
METRIC_RE = re.compile(
    r"(AUC|precision|recall|f1|RMSE|MAE|MAPE|accuracy|"
    r"\d+(\.\d+)?\s?%|\$\d+|\b\d{2,}\b)",
    re.I
)

def skill_evidence_score(parsed_resume, skill):
    """
    Checks where a skill appears and assigns a tier:
    0.0 = Missing
    0.3 = Only in Skills section
    0.7 = In an Experience/Project bullet (no metric)
    1.0 = In an Experience/Project bullet WITH a metric
    """
    skill_l = skill.lower()
    
    # Combine experience and project bullets
    all_bullets = parsed_resume.get("experience", []) + parsed_resume.get("projects", [])
    
    best_score = 0.0
    
    # Check bullets first
    for bullet in all_bullets:
        if skill_l in bullet.lower():
            # Has a metric?
            if METRIC_RE.search(bullet):
                return 1.0  # Highest tier found, return immediately
            # Is it vague?
            elif any(re.search(p, bullet, re.I) for p in VAGUE_PATTERNS):
                best_score = max(best_score, 0.3) # Vague claim = treat like a list
            else:
                best_score = max(best_score, 0.7) # Good claim
                
    # If not in bullets, check the skills section
    if best_score == 0.0:
        skills_text = parsed_resume.get("skills", "") + " " + parsed_resume.get("education", "")
        if skill_l in skills_text.lower():
            best_score = 0.3
            
    return best_score


# ============================================================
# TEST BLOCK
# ============================================================
if __name__ == "__main__":
    folder_path = r"C:\Samad\SundarPitchTheAi\Projects\semantic-resume-scorer\Sample space\Simulated SS"
    pdf_file = "R1.pdf"
    exact_path = os.path.join(folder_path, pdf_file)
    
    print(f"Parsing {pdf_file}...")
    parsed = parse_resume_structured(exact_path)
    
    # Let's test the specific skills from Karthik's resume
    test_skills = ["Python", "TensorFlow", "Machine Learning", "SQL"]
    
    print("\n" + "="*50)
    print("EVIDENCE SCORING TEST")
    print("="*50)
    
    for skill in test_skills:
        score = skill_evidence_score(parsed, skill)
        print(f"Skill: {skill:<15} | Evidence Tier: {score}")