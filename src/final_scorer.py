import os
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from resume_parser import parse_resume_structured
from rule_engine import get_rule_score

# 1. Load the ML model you just trained
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(root_dir, "models", "ml_scorer.pkl")

if not os.path.exists(model_path):
    print("⚠️ ML Model not found. Falling back to generic embedding.")
    ml_model = None
else:
    ml_model = joblib.load(model_path)

# 2. Load the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_ml_score(parsed_resume, jd_text):
    """
    Uses the trained Ridge Regression model to score the resume-JD pair.
    """
    if ml_model is None:
        # Fallback if model is missing
        return 0.5
        
    resume_text = parsed_resume["raw_text"]
    
    # Embed the texts
    res_emb = model.encode(resume_text)
    jd_emb = model.encode(jd_text)
    
    # Recreate the exact features we trained on
    cos_sim = np.dot(res_emb, jd_emb) / (np.linalg.norm(res_emb) * np.linalg.norm(jd_emb))
    euclidean = np.linalg.norm(res_emb - jd_emb)
    
    # Predict the score (returns a value between 0 and 1)
    features = np.array([[cos_sim, euclidean]])
    predicted_score = ml_model.predict(features)[0]
    
    # Ensure it doesn't go below 0 or above 1
    return max(0.0, min(1.0, float(predicted_score)))

def calculate_final_score(parsed_resume, jd_text, required_skills):
    # 1. Get the Rule Score
    rule_results = get_rule_score(parsed_resume, jd_text, required_skills)
    rule_score = rule_results["final_rule_score"]

    # 2. Get the Real ML Score
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