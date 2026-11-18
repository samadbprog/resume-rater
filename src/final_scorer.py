import os
from sentence_transformers import util
from resume_parser import parse_resume_structured
from rule_engine import get_rule_score
# Import the model we already loaded in semantic_matcher
from semantic_matcher import model 

def get_ml_score(parsed_resume, jd_text):
    """
    REAL ML SCORE: Whole-document semantic similarity.
    Instead of a fake placeholder, we embed the entire resume and JD,
    and calculate the cosine similarity.
    """
    resume_text = parsed_resume["raw_text"]
    
    # Embed the full text
    res_emb = model.encode(resume_text, convert_to_tensor=True)
    jd_emb = model.encode(jd_text, convert_to_tensor=True)
    
    # Cosine similarity (usually lands between 0.2 and 0.6 for whole docs)
    cosine_sim = float(util.cos_sim(res_emb, jd_emb)[0][0])
    
    # Rescale it to 0-1 (assuming 0.15 is floor, 0.65 is ceiling)
    rescaled = max(0.0, min(1.0, (cosine_sim - 0.15) / 0.50))
    
    return rescaled

def calculate_final_score(parsed_resume, jd_text, required_skills):
    # 1. Get the Rule Score
    rule_results = get_rule_score(parsed_resume, jd_text, required_skills)
    rule_score = rule_results["final_rule_score"]

    # 2. Get the REAL ML Score
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