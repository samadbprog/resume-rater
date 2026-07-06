import streamlit as st
import pickle
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import docx
import io
from huggingface_hub import hf_hub_download

# ── Config ────────────────────────────────────────────────────────────────────
# Hugging Face model info
REPO_ID = "Goldenf4ng/resume-scorer-model"
FILENAME = "model.pkl"

EMBED_MODEL = "all-MiniLM-L6-v2"

DEPTH_INDICATORS = {
    'auc-roc', 'precision', 'recall', 'f1', '%',
    'improved', 'fine-tuned', 'pipeline', 'deployed', 'kaggle'
}

# Hybrid blend weights (applied independently to all 3 versions)
BLEND_MODEL = 0.30
BLEND_RULE  = 0.70

# ── Version Definitions ───────────────────────────────────────────────────────
VERSIONS = {
    "A": {
        "label"     : "ATS Match Score",
        "emoji"     : "🤖",
        "semantic"  : 0.10,
        "keyword"   : 0.70,
        "depth"     : 0.20,
        "avg_weight": 0.20,   # contribution to final weighted average
    },
    "B": {
        "label"     : "Overall Fit Score",
        "emoji"     : "🎯",
        "semantic"  : 0.35,
        "keyword"   : 0.30,
        "depth"     : 0.35,
        "avg_weight": 0.40,
    },
    "C": {
        "label"     : "Candidate Quality Score",
        "emoji"     : "🌟",
        "semantic"  : 0.40,
        "keyword"   : 0.20,
        "depth"     : 0.40,
        "avg_weight": 0.40,
    },
}

# ── Loaders ───────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Download and load the model from Hugging Face Hub"""
    try:
        # Try hf_hub_download first with token=False for public repo
        model_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=FILENAME,
            repo_type="model",
            token=False  # Explicitly no token for public repo
        )
        
        with open(model_path, "rb") as f:
            model_bundle = pickle.load(f)
        
        st.success("✅ Model loaded successfully from Hugging Face!")
        return model_bundle
        
    except Exception as e:
        st.warning(f"hf_hub_download failed: {e}. Trying direct download...")
        
        try:
            # Fallback: direct download
            url = f"https://huggingface.co/{REPO_ID}/resolve/main/{FILENAME}"
            response = requests.get(url)
            response.raise_for_status()
            
            model_bundle = pickle.loads(response.content)
            st.success("✅ Model loaded successfully via direct download!")
            return model_bundle
            
        except Exception as e2:
            st.error(f"Failed to load model: {e2}")
            return None

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

# ── Per-Version Scorer ────────────────────────────────────────────────────────
def compute_version_score(model_bundle, sem: float, kw: float,
                          depth: float, ver: dict) -> dict:
    # Rule-based score (always computed)
    rule_score = (ver["semantic"] * sem) + (ver["keyword"] * kw) + (ver["depth"] * depth)
    
    # ML model score (only if model exists)
    if model_bundle is not None:
        features = np.array([[sem, kw, depth]])
        model_score = float(np.clip(model_bundle["model"].predict(features)[0], 0, 1))
        # Hybrid blend
        final_raw = (BLEND_MODEL * model_score) + (BLEND_RULE * rule_score)
    else:
        # Rule-based only
        model_score = 0.0
        final_raw = rule_score
    
    final_raw = float(np.clip(final_raw, 0, 1))
    
    return {
        "model_score": round(model_score, 4),
        "rule_score": round(rule_score, 4),
        "final_raw": round(final_raw, 4),
        "out_of_100": round(final_raw * 100, 1),
    }

# ── Master Scorer ─────────────────────────────────────────────────────────────
def compute_all_scores(model_bundle, embedder,
                       resume_text: str, jd_text: str) -> dict:
    # Shared features (computed once)
    sem = get_semantic_similarity(embedder, resume_text, jd_text)
    kw = get_keyword_overlap(resume_text, jd_text)
    depth = get_depth_score(resume_text)

    results = {}
    for ver_key, ver_cfg in VERSIONS.items():
        results[ver_key] = compute_version_score(
            model_bundle, sem, kw, depth, ver_cfg
        )

    # Weighted average of the three out_of_100 scores → final /10
    weighted_avg = sum(
        results[k]["out_of_100"] * VERSIONS[k]["avg_weight"]
        for k in VERSIONS
    )
    final_score = round(float(np.clip(weighted_avg / 10, 0, 10)), 1)

    return {
        "versions": results,
        "sem": round(sem, 4),
        "kw": round(kw, 4),
        "depth": round(depth, 4),
        "final_score": final_score,  # out of 10
    }

# ── Match Label ───────────────────────────────────────────────────────────────
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
def generate_feedback(sem: float, kw: float, depth: float) -> list:
    fb = []

    if sem < 0.4:
        fb.append("⚠️ Low semantic similarity — your resume language doesn't closely match the job description.")
    elif sem > 0.7:
        fb.append("✅ High semantic similarity — strong language alignment with the job description.")
    else:
        fb.append("🔶 Moderate semantic similarity — consider mirroring more JD language in your resume.")

    if kw < 0.3:
        fb.append("⚠️ Low keyword coverage — many critical JD keywords are missing from your resume.")
    elif kw >= 0.6:
        fb.append("✅ Strong keyword coverage — great match on critical job keywords.")
    else:
        fb.append("🔶 Moderate keyword coverage — add more role-specific keywords from the JD.")

    if depth == 0.0:
        fb.append("⚠️ Low depth — add quantified achievements, metrics, or technical specifics.")
    elif depth >= 0.67:
        fb.append("✅ Strong depth — your resume demonstrates specific, measurable impact.")
    else:
        fb.append("🔶 Moderate depth — include more metrics or technical depth indicators.")

    return fb

# ── UI ────────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="AI Resume Scorer", page_icon="📄", layout="centered")
    st.title("📄 AI Resume Scorer")
    st.caption("Hybrid ML + Rule-Based · Three-Lens Scoring Engine")

    # Load resources
    model_bundle = load_model()
    embedder     = load_embedder()

    if model_bundle is None:
        st.warning("⚠️ Running without ML model - only rule-based scoring available")
    else:
        # Training metrics (only if model loaded)
        with st.expander("📊 Model Training Metrics"):
            metrics = model_bundle.get("metrics", {})
            if metrics:
                col1, col2, col3 = st.columns(3)
                col1.metric("Test MAE",   f'{metrics.get("test_mae",   "N/A"):.4f}')
                col2.metric("Test R²",    f'{metrics.get("test_r2",    "N/A"):.4f}')
                col3.metric("CV R² Mean", f'{metrics.get("cv_r2_mean", "N/A"):.4f} ± '
                                          f'{metrics.get("cv_r2_std",  "N/A"):.4f}')
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

            # Show warning if model is not available
            if model_bundle is None:
                st.info("ℹ️ Using rule-based scoring only (ML model unavailable)")

            data = compute_all_scores(model_bundle, embedder, resume_text, jd_text)

        # ── Final Score Display ────────────────────────────────────────────
        st.markdown("---")
        final  = data["final_score"]
        label, colour = get_score_label(final)

        st.markdown(
            f"<h1 style='text-align:center; color:{colour};'>{final} / 10</h1>"
            f"<h3 style='text-align:center;'>{label}</h3>",
            unsafe_allow_html=True
        )
        st.caption(
            "Final score = weighted average of the three lenses "
            "(ATS 20% · Overall Fit 40% · Candidate Quality 40%)"
        )

        # ── Three-Lens Score Cards ─────────────────────────────────────────
        st.markdown("---")
        st.subheader("🔭 Three-Lens Breakdown")

        col_a, col_b, col_c = st.columns(3)
        cols = [col_a, col_b, col_c]

        for idx, (ver_key, ver_cfg) in enumerate(VERSIONS.items()):
            ver_result = data["versions"][ver_key]
            s = ver_result["out_of_100"]

            # Colour band per score
            if s >= 75:
                s_colour = "green"
            elif s >= 55:
                s_colour = "orange"
            else:
                s_colour = "red"

            with cols[idx]:
                st.markdown(
                    f"<div style='text-align:center;'>"
                    f"<p style='font-size:14px; margin-bottom:2px;'>"
                    f"{ver_cfg['emoji']} <b>{ver_cfg['label']}</b></p>"
                    f"<p style='font-size:36px; font-weight:bold; color:{s_colour}; margin:0;'>"
                    f"{s}</p>"
                    f"<p style='font-size:12px; color:grey; margin-top:2px;'>out of 100</p>"
                    f"</div>",
                    unsafe_allow_html=True
                )

        # ── Per-Version Formula Detail ─────────────────────────────────────
        st.markdown("---")
        with st.expander("🧮 Detailed Score Breakdown per Lens"):
            for ver_key, ver_cfg in VERSIONS.items():
                vr = data["versions"][ver_key]
                st.markdown(f"### {ver_cfg['emoji']} {ver_cfg['label']} (Version {ver_key})")
                st.markdown(
                    f"**Weights →** Semantic `{ver_cfg['semantic']}` · "
                    f"Keyword `{ver_cfg['keyword']}` · Depth `{ver_cfg['depth']}`\n\n"
                    f"**Rule Score** = "
                    f"({ver_cfg['semantic']} × {data['sem']}) + "
                    f"({ver_cfg['keyword']} × {data['kw']}) + "
                    f"({ver_cfg['depth']} × {data['depth']}) "
                    f"= **{vr['rule_score']}**\n\n"
                    f"**ML Model Score** = **{vr['model_score']}**\n\n"
                    f"**Hybrid** = (0.30 × {vr['model_score']}) + (0.70 × {vr['rule_score']}) "
                    f"= **{vr['final_raw']}** → **{vr['out_of_100']} / 100**"
                )
                st.markdown("---")

            st.markdown(
                f"**Final Score** = "
                f"(0.20 × {data['versions']['A']['out_of_100']}) + "
                f"(0.40 × {data['versions']['B']['out_of_100']}) + "
                f"(0.40 × {data['versions']['C']['out_of_100']}) "
                f"= **{final * 10:.1f} / 100** → **{final} / 10**"
            )

        # ── Raw Feature Scores ─────────────────────────────────────────────
        st.markdown("---")
        st.subheader("📐 Raw Feature Scores")
        c1, c2, c3 = st.columns(3)
        c1.metric("🧠 Semantic Similarity", data["sem"])
        c2.metric("🔑 Keyword Overlap",     data["kw"])
        c3.metric("📏 Depth Score",         data["depth"])

        # ── Feedback ───────────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("💬 Feedback")
        for msg in generate_feedback(data["sem"], data["kw"], data["depth"]):
            st.markdown(msg)

if __name__ == "__main__":
    main()