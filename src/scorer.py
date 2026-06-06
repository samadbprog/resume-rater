import re
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer

# ── Model ────────────────────────────────────────────────────────────────────
model = SentenceTransformer('all-MiniLM-L6-v2')

# ── Depth Indicators ─────────────────────────────────────────────────────────
DEPTH_INDICATORS = {
    'auc-roc', 'precision', 'recall', 'f1', '%', 'improved',
    'fine-tuned', 'pipeline', 'deployed', 'kaggle'
}

# ── Normalization ─────────────────────────────────────────────────────────────
MIN_SIM = 0.30
MAX_SIM = 0.88

def normalize_score(similarity):
    """Normalize cosine similarity to a 0–10 scale."""
    normalized = (similarity - MIN_SIM) / (MAX_SIM - MIN_SIM)
    return round(max(0.0, min(10.0, normalized * 10)), 1)


# ── Embedding ─────────────────────────────────────────────────────────────────
def get_embedding(text):
    """Get sentence embedding for a given text."""
    return model.encode(text, convert_to_tensor=True)


# ── Keyword Extraction ────────────────────────────────────────────────────────
def extract_keywords(text):
    """Extract keywords from text using TF-IDF."""
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=200, stop_words='english')
    try:
        vectorizer.fit([text])
        return set(vectorizer.get_feature_names_out())
    except Exception:
        return set()


def extract_critical_keywords_from_jd(jd_text, top_n=20):
    """Extract the most critical keywords from the job description."""
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=200, stop_words='english')
    try:
        tfidf_matrix = vectorizer.fit_transform([jd_text])
        feature_names = vectorizer.get_feature_names_out()
        scores = tfidf_matrix.toarray()[0]
        top_indices = scores.argsort()[::-1][:top_n]
        return set(feature_names[i] for i in top_indices)
    except Exception:
        return set()


# ── Depth Bonus ───────────────────────────────────────────────────────────────
def depth_bonus(resume_text):
    """Award bonus points for measurable results in resume."""
    text_lower = resume_text.lower()
    hits = sum(1 for indicator in DEPTH_INDICATORS if indicator in text_lower)
    if hits >= 8:
        return 1.5
    elif hits >= 5:
        return 1.0
    elif hits >= 3:
        return 0.5
    return 0.0


# ── Irrelevance Penalty ───────────────────────────────────────────────────────
def detect_irrelevant_content(resume_text, jd_critical_keywords):
    """
    Check how little of the resume overlaps with JD critical keywords.
    If overlap is very low, apply a penalty.
    """
    resume_words = set(re.findall(r'\b[a-z]{3,}\b', resume_text.lower()))
    overlap = len(resume_words & jd_critical_keywords)
    overlap_ratio = overlap / max(len(jd_critical_keywords), 1)

    if overlap_ratio < 0.1:
        return 2.0
    elif overlap_ratio < 0.2:
        return 1.2
    elif overlap_ratio < 0.3:
        return 0.5
    return 0.0


# ── Main Scorer ───────────────────────────────────────────────────────────────
def score_resume(resume_text, jd_text):
    try:
        # 1. Dynamically extract critical keywords from THIS JD
        jd_critical = extract_critical_keywords_from_jd(jd_text, top_n=20)

        # 2. Semantic score
        resume_emb = get_embedding(resume_text)
        jd_emb     = get_embedding(jd_text)
        similarity = util.cos_sim(resume_emb, jd_emb).item()
        sem_score  = normalize_score(similarity)

        # 3. Keyword score
        resume_keywords = extract_keywords(resume_text)
        jd_keywords     = extract_keywords(jd_text)

        if jd_keywords:
            total_weight   = 0.0
            matched_weight = 0.0
            for word in jd_keywords:
                weight = 3.0 if word in jd_critical else 1.0
                total_weight   += weight
                if word in resume_keywords:
                    matched_weight += weight
            kw_score = round((matched_weight / total_weight) * 10, 1)
        else:
            kw_score = 0.0

        # 4. Base hybrid (60% semantic, 40% keyword)
        base_score = round((0.6 * sem_score) + (0.4 * kw_score), 1)

        # 5. Depth bonus
        bonus = depth_bonus(resume_text)
        score = round(base_score + bonus, 1)

        # 6. Irrelevance penalty
        penalty = detect_irrelevant_content(resume_text, jd_critical)
        score   = round(max(0.0, score - penalty), 1)

        # 7. Hard caps
        if sem_score < 3.0:
            score = min(score, 2.0)
        elif sem_score < 4.5:
            score = min(score, 4.0)
        elif sem_score < 6.0:
            score = min(score, 6.5)

        # 8. Absolute ceiling
        return float(min(score, 10.0))

    except Exception as e:
        print(f"[score_resume ERROR] {e}")
        return 0.0




# ── Feedback ──────────────────────────────────────────────────────────────────
def get_feedback(score):
    """Return feedback message based on score."""
    if score >= 9.0:
        return "🏆 Excellent match! Your resume is highly aligned with the job description."
    elif score >= 7.5:
        return "✅ Good match! A few tweaks could make it even stronger."
    elif score >= 5.0:
        return "⚠️ Moderate match. Consider adding more relevant keywords and experience."
    elif score >= 3.0:
        return "❌ Below average. Your resume needs significant alignment with the JD."
    else:
        return "🚫 Poor match. The resume does not align well with this job description."
