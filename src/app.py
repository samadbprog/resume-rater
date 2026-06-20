import streamlit as st
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import docx
import io

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH = "model.pkl"
EMBED_MODEL = "all-MiniLM-L6-v2"

DEPTH_INDICATORS = {
    'auc-roc', 'precision', 'recall', 'f1', '%',
    'improved', 'fine-tuned', 'pipeline', 'deployed', 'kaggle'
}

# ---- SCORING VERSION TOGGLE ----
VERSION = "B"  # Change to "B" to test Version B
# --------------------------------

# Hybrid blend weights
BLEND_MODEL  = 0.30
BLEND_RULE   = 0.70

# Rule-based sub-weights (must sum to 1.0)
if VERSION == "A":
    W_SEMANTIC = 0.40
    W_KEYWORD  = 0.20
    W_DEPTH    = 0.40
elif VERSION == "B":
    W_SEMANTIC = 0.35
    W_KEYWORD  = 0.30
    W_DEPTH    = 0.35

# ── Loaders ───────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_embedder():
    return SentenceTransformer(EMBED_MODEL)

# ── File Parser ───────────────────────────────────────────────────────────────
def parse_file(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
        return " ".join(page.extract_text() or "" for page in reader.pages)
    elif name.endswith(".docx"):
        doc = docx.Document(io.BytesIO(uploaded_file.read()))
        return " ".join(p.text for p in doc.paragraphs)
    else:
        raise ValueError(f"Unsupported file format: {uploaded_file.name}")

# ── Feature Extractors ────────────────────────────────────────────────────────
def get_semantic_similarity(embedder, resume_text: str, jd_text: str) -> float:
    emb = embedder.encode([resume_text, jd_text], batch_size=32)
    return float(cosine_similarity([emb[0]], [emb[1]])[0][0])

def get_keyword_overlap(resume_text: str, jd_text: str, top_n: int = 20) -> float:
    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=200, stop_words="english")
    tfidf.fit([jd_text])
    scores  = dict(zip(tfidf.get_feature_names_out(),
                       tfidf.transform([jd_text]).toarray()[0]))
    top_kws = sorted(scores, key=scores.get, reverse=True)[:top_n]
    resume_lower = resume_text.lower()
    hits = sum(1 for kw in top_kws if kw in resume_lower)
    return hits / top_n

def get_depth_score(resume_text: str) -> float:
    resume_lower = resume_text.lower()
    hits = sum(1 for ind in DEPTH_INDICATORS if ind in resume_lower)
    if hits >= 8:
        return 1.00
    elif hits >= 5:
        return 0.67
    elif hits >= 3:
        return 0.33
    return 0.00

# ── Scoring ───────────────────────────────────────────────────────────────────
def compute_score(model_bundle, embedder, resume_text: str, jd_text: str) -> dict:
    # Features
    sem   = get_semantic_similarity(embedder, resume_text, jd_text)
    kw    = get_keyword_overlap(resume_text, jd_text)
    depth = get_depth_score(resume_text)

    # ML model score
    features     = np.array([[sem, kw, depth]])
    model_score  = float(np.clip(model_bundle["model"].predict(features)[0], 0, 1))

    # Rule-based score (weighted formula)
    rule_score   = (W_SEMANTIC * sem) + (W_KEYWORD * kw) + (W_DEPTH * depth)

    # Hybrid blend
    final_score  = (BLEND_MODEL * model_score) + (BLEND_RULE * rule_score)
    display      = round(float(np.clip(final_score, 0, 1)) * 10, 1)

    return {
        "display_score"  : display,
        "semantic"       : round(sem,   4),
        "keyword_overlap": round(kw,    4),
        "depth_score"    : round(depth, 4),
        "model_score"    : round(model_score, 4),
        "rule_score"     : round(rule_score,  4),
        "final_raw"      : round(final_score, 4),
    }

# ── Score Label ───────────────────────────────────────────────────────────────
def get_score_label(score: float):
    if score < 4.0:
        return "Poor Match ❌", "red"
    elif score < 6.5:
        return "Moderate Match 🔶", "orange"
    elif score < 8.0:
        return "Good Match 👍", "blue"
    else:
        return "Excellent Match 🌟", "green"

# ── Feedback ──────────────────────────────────────────────────────────────────
def generate_feedback(scores: dict) -> list[str]:
    fb = []

    # Semantic
    if scores["semantic"] < 0.4:
        fb.append("⚠️ Low semantic similarity — your resume language doesn't closely match the job description.")
    elif scores["semantic"] > 0.7:
        fb.append("✅ High semantic similarity — strong language alignment with the job description.")
    else:
        fb.append("🔶 Moderate semantic similarity — consider mirroring more JD language in your resume.")

    # Keyword
    if scores["keyword_overlap"] < 0.3:
        fb.append("⚠️ Low keyword coverage — many critical JD keywords are missing from your resume.")
    elif scores["keyword_overlap"] >= 0.6:
        fb.append("✅ Strong keyword coverage — great match on critical job keywords.")
    else:
        fb.append("🔶 Moderate keyword coverage — add more role-specific keywords from the JD.")

    # Depth
    if scores["depth_score"] == 0.0:
        fb.append("⚠️ Low depth — add quantified achievements, metrics, or technical specifics.")
    elif scores["depth_score"] >= 0.67:
        fb.append("✅ Strong depth — your resume demonstrates specific, measurable impact.")
    else:
        fb.append("🔶 Moderate depth — include more metrics or technical depth indicators.")

    return fb

# ── UI ────────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="AI Resume Scorer", page_icon="📄", layout="centered")
    st.title("📄 AI Resume Scorer")
    st.caption(f"Hybrid ML + Rule-Based scoring engine — Version {VERSION} "
               f"(Sem {W_SEMANTIC} | Key {W_KEYWORD} | Depth {W_DEPTH})")

    # Load resources
    try:
        model_bundle = load_model()
        embedder     = load_embedder()
    except Exception as e:
        st.error(f"Failed to load model or embedder: {e}")
        return

    # Training metrics (if stored in bundle)
    with st.expander("📊 Model Training Metrics"):
        metrics = model_bundle.get("metrics", {})
        if metrics:
            col1, col2, col3 = st.columns(3)
            col1.metric("Test MAE",  f'{metrics.get("test_mae",  "N/A"):.4f}')
            col2.metric("Test R²",   f'{metrics.get("test_r2",   "N/A"):.4f}')
            col3.metric("CV R² Mean",f'{metrics.get("cv_r2_mean","N/A"):.4f} ± '
                                     f'{metrics.get("cv_r2_std", "N/A"):.4f}')
        else:
            st.info("No training metrics found in model bundle.")

    st.markdown("---")

    # Inputs
    uploaded = st.file_uploader("📁 Upload Resume (PDF or DOCX)", type=["pdf", "docx"])
    jd_text  = st.text_area("📋 Paste Job Description", height=200)

    if st.button("🚀 Score Resume", use_container_width=True):
        if not uploaded:
            st.warning("Please upload a resume file.")
            return
        if not jd_text.strip():
            st.warning("Please paste a job description.")
            return

        with st.spinner("Analysing resume..."):
            try:
                resume_text = parse_file(uploaded)
            except ValueError as e:
                st.error(str(e))
                return

            scores = compute_score(model_bundle, embedder, resume_text, jd_text)

        # ── Score Display ──────────────────────────────────────────────────
        st.markdown("---")
        score = scores["display_score"]
        label, colour = get_score_label(score)

        st.markdown(
            f"<h1 style='text-align:center; color:{colour};'>{score} / 10</h1>"
            f"<h3 style='text-align:center;'>{label}</h3>",
            unsafe_allow_html=True
        )

        # ── Score Breakdown ────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("📐 Score Breakdown")

        col1, col2, col3 = st.columns(3)
        col1.metric("🧠 Semantic",  scores["semantic"])
        col2.metric("🔑 Keywords",  scores["keyword_overlap"])
        col3.metric("📏 Depth",     scores["depth_score"])

        st.markdown("---")
        col4, col5, col6 = st.columns(3)
        col4.metric("🤖 Model Score", scores["model_score"])
        col5.metric("📐 Rule Score",  scores["rule_score"])
        col6.metric("🏁 Final Raw",   scores["final_raw"])

        # ── Formula Display ────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("🧮 Scoring Formula")
        st.markdown(
            f"""
            **Rule Score** = ({W_SEMANTIC} × Semantic) + ({W_KEYWORD} × Keyword) + ({W_DEPTH} × Depth)
            = ({W_SEMANTIC} × {scores['semantic']}) + ({W_KEYWORD} × {scores['keyword_overlap']}) + ({W_DEPTH} × {scores['depth_score']})
            = **{scores['rule_score']}**

            **Final Score** = (0.30 × Model) + (0.70 × Rule)
            = (0.30 × {scores['model_score']}) + (0.70 × {scores['rule_score']})
            = **{scores['final_raw']}** → **{score} / 10**
            """
        )

        # ── Feedback ───────────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("💬 Feedback")
        for msg in generate_feedback(scores):
            st.markdown(msg)

if __name__ == "__main__":
    main()
