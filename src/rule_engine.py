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
