import os
import re
from resume_parser import parse_resume_structured
from evidence_scorer import skill_evidence_score
from semantic_matcher import get_requirement_level_similarity

# Vague phrases that kill depth
VAGUE_PATTERNS = [
    r"\bworked on\b", r"\bexposure to\b", r"\bparticipated in\b",
    r"\binvolved in\b", r"\bfamiliar with\b", r"\bhelped with\b"
]

# Metric detection (numbers, %, AUC, etc.)
METRIC_RE = re.compile(
    r"(AUC|precision|recall|f1|RMSE|MAE|MAPE|accuracy|"
    r"\d+(\.\d+)?\s?%|\$\d+|\b\d{2,}\b)",
    re.I
)

def calculate_depth_score(resume_bullets):
    """
    Calculates depth based on metrics and penalizes vague claims.
    """
    if not resume_bullets:
        return 0.0
        
    total_metrics = 0
    vague_count = 0
    
    for bullet in resume_bullets:
        if METRIC_RE.search(bullet):
            total_metrics += 1
        if any(re.search(p, bullet, re.I) for p in VAGUE_PATTERNS):
            vague_count += 1
            
    # Base depth score: 1 metric = 0.2, 5 metrics = 1.0
    base_depth = min(total_metrics / 5.0, 1.0)
    
    # Vagueness penalty: subtract 0.1 for every vague bullet (max -0.3)
    penalty = min(vague_count * 0.1, 0.3)
    
    return max(0.0, base_depth - penalty)


def get_rule_score(parsed_resume, jd_text, required_skills):
    """
    Combines Keyword, Semantic, and Depth into a single Rule Score.
    """
    # 1. Keyword (Evidence-Tiered)
    kw_scores = [skill_evidence_score(parsed_resume, s) for s in required_skills]
    kw_score = sum(kw_scores) / len(kw_scores) if kw_scores else 0.0

    # 2. Semantic (Requirement-Level)
    bullets = parsed_resume.get("experience", []) + parsed_resume.get("projects", [])
    sem_score = get_requirement_level_similarity(jd_text, bullets)

    # 3. Depth (Metrics + Vagueness)
    depth_score = calculate_depth_score(bullets)

    # 4. Balanced Weights (0.35 sem, 0.30 kw, 0.35 depth)
    # Note: These are the rule weights, not the ML blend.
    final_rule_score = (0.35 * sem_score) + (0.30 * kw_score) + (0.35 * depth_score)

    return {
        "sem_sim": round(sem_score, 2),
        "kw_overlap": round(kw_score, 2),
        "depth": round(depth_score, 2),
        "final_rule_score": round(final_rule_score, 2)
    }

# ============================================================
# TEST BLOCK
# ============================================================
if __name__ == "__main__":
    base_dir = r"C:\Samad\SundarPitchTheAi\Projects\semantic-resume-scorer\Sample space\Simulated SS"
    pdf_path = os.path.join(base_dir, "R1.pdf")
    jd_path = os.path.join(base_dir, "Job Description Sample.txt")

    with open(jd_path, "r", encoding="utf-8") as f:
        jd_text = f.read()

    parsed = parse_resume_structured(pdf_path)
    
    # Let's define the must-have skills from the JD manually
    required_skills = ["Python", "SQL", "Machine Learning", "Git"]

    print("\n" + "="*50)
    print("RULE ENGINE TEST (R1.pdf)")
    print("="*50)
    
    results = get_rule_score(parsed, jd_text, required_skills)
    
    print("\n--- BREAKDOWN ---")
    print(f"Semantic Similarity: {results['sem_sim']}")
    print(f"Keyword Evidence:    {results['kw_overlap']}")
    print(f"Depth Score:         {results['depth']}")
    print("-" * 20)
    print(f"FINAL RULE SCORE:    {results['final_rule_score']} / 1.0")
