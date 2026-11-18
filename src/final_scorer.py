import os
from resume_parser import parse_resume_structured
from rule_engine import get_rule_score

def get_ml_score(parsed_resume, jd_text):
    """
    SIMULATED ML MODEL.
    In the future, this will call your trained HuggingFace model.
    For now, it acts as a basic semantic whole-document cosine similarity.
    """
    # Simulating an ML model that is a bit more forgiving than the strict rules
    # (This is just a placeholder to make the math work for now)
    bullets = parsed_resume.get("experience", []) + parsed_resume.get("projects", [])
    if len(bullets) > 3:
        return 0.65 # Fake high score for having a lot of bullets
    return 0.40

def calculate_final_score(parsed_resume, jd_text, required_skills):
    # 1. Get the Rule Score
    rule_results = get_rule_score(parsed_resume, jd_text, required_skills)
    rule_score = rule_results["final_rule_score"]

    # 2. Get the ML Score
    ml_score = get_ml_score(parsed_resume, jd_text)

    # 3. BLEND: 0.30 ML + 0.70 Rule
    blended_score = (0.30 * ml_score) + (0.70 * rule_score)

    # 4. Convert to out-of-10 scale
    final_score_out_of_10 = round(blended_score * 10, 1)

    return {
        "rule_score": round(rule_score, 2),
        "ml_score": round(ml_score, 2),
        "blended_score": round(blended_score, 2),
        "final_score_out_of_10": final_score_out_of_10,
        "breakdown": rule_results
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
    required_skills = ["Python", "SQL", "Machine Learning", "Git"]

    print("\n" + "="*50)
    print("FINAL SCORER TEST (R1.pdf)")
    print("="*50)
    
    results = calculate_final_score(parsed, jd_text, required_skills)
    
    print("\n--- SCORE BREAKDOWN ---")
    print(f"Semantic Similarity: {results['breakdown']['sem_sim']}")
    print(f"Keyword Evidence:    {results['breakdown']['kw_overlap']}")
    print(f"Depth Score:         {results['breakdown']['depth']}")
    print("-" * 20)
    print(f"Rule Score (70%):    {results['rule_score']}")
    print(f"ML Score (30%):      {results['ml_score']}")
    print("-" * 20)
    print(f"BLENDED SCORE:       {results['blended_score']} / 1.0")
    print(f"FINAL SCORE:         {results['final_score_out_of_10']} / 10")
    print("="*50)