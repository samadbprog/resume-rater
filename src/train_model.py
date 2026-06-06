# =============================================================
# train_model.py
# Trains a Gradient Boosting Regressor on resume–JD pairs
# and saves the trained model pipeline to model.pkl
# =============================================================

import pandas as pd
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings("ignore")

# ── 1. Load Dataset ───────────────────────────────────────────
print("📂 Loading dataset...")
df = pd.read_csv("../data/resumeJD2_pairs.csv")



# Validate columns
required_cols = {"resume_text", "job_description", "match_score"}
if not required_cols.issubset(df.columns):
    raise ValueError(f"Missing columns! Expected: {required_cols}, Got: {set(df.columns)}")

# Drop nulls
df = df.dropna(subset=["resume_text", "job_description", "match_score"])
print(f"✅ Loaded {len(df)} valid pairs.")

# ── 2. Load Sentence Transformer ─────────────────────────────
print("\n🤖 Loading SentenceTransformer model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ── 3. Compute Semantic Features ─────────────────────────────
print("🔢 Computing semantic embeddings...")

resume_embeddings = embedder.encode(
    df["resume_text"].tolist(), 
    show_progress_bar=True, 
    batch_size=32
)
jd_embeddings = embedder.encode(
    df["job_description"].tolist(), 
    show_progress_bar=True, 
    batch_size=32
)

# Cosine similarity between resume and JD embeddings
def cosine_similarity_rowwise(a, b):
    dot = np.sum(a * b, axis=1)
    norm = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    return dot / (norm + 1e-9)

semantic_sim = cosine_similarity_rowwise(resume_embeddings, jd_embeddings).reshape(-1, 1)
print(f"✅ Semantic similarity computed. Range: [{semantic_sim.min():.3f}, {semantic_sim.max():.3f}]")

# ── 4. Compute TF-IDF Keyword Overlap Features ───────────────
print("\n📊 Computing TF-IDF keyword overlap features...")

# Concatenated text for TF-IDF vocabulary building
all_texts = df["resume_text"].tolist() + df["job_description"].tolist()

tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=200)
tfidf.fit(all_texts)

# Get top keywords from JD for each pair
def keyword_overlap_score(resume, jd, vectorizer):
    """
    Computes the ratio of JD top keywords found in the resume.
    Returns a single float score per pair.
    """
    jd_vec = vectorizer.transform([jd]).toarray()[0]
    resume_vec = vectorizer.transform([resume]).toarray()[0]

    # Top 20 JD keywords by TF-IDF weight
    top_indices = np.argsort(jd_vec)[-20:]
    jd_top = jd_vec[top_indices]
    resume_top = resume_vec[top_indices]

    # Overlap: how many top JD keywords appear in resume
    overlap = np.sum((resume_top > 0) & (jd_top > 0))
    return overlap / 20.0  # normalize to 0–1

keyword_scores = np.array([
    keyword_overlap_score(row["resume_text"], row["job_description"], tfidf)
    for _, row in df.iterrows()
]).reshape(-1, 1)

print(f"✅ Keyword overlap computed. Range: [{keyword_scores.min():.3f}, {keyword_scores.max():.3f}]")

# ── 5. Compute Depth Bonus Feature ───────────────────────────
print("\n🔍 Computing depth indicator features...")

DEPTH_INDICATORS = {
    'auc-roc', 'precision', 'recall', 'f1', '%', 'improved',
    'fine-tuned', 'pipeline', 'deployed', 'kaggle',
    'accuracy', 'optimized', 'reduced', 'increased', 'achieved',
    'built', 'designed', 'led', 'managed', 'automated'
}

def depth_score(resume_text):
    """Count how many depth indicators appear in the resume."""
    text_lower = resume_text.lower()
    hits = sum(1 for indicator in DEPTH_INDICATORS if indicator in text_lower)
    if hits >= 8:
        return 1.0
    elif hits >= 5:
        return 0.67
    elif hits >= 3:
        return 0.33
    else:
        return 0.0

depth_scores = np.array([
    depth_score(row["resume_text"]) for _, row in df.iterrows()
]).reshape(-1, 1)

print(f"✅ Depth scores computed. Range: [{depth_scores.min():.3f}, {depth_scores.max():.3f}]")

# ── 6. Combine All Features ───────────────────────────────────
print("\n🔗 Combining feature matrix...")

# Feature matrix: [semantic_sim, keyword_overlap, depth_score]
X = np.hstack([semantic_sim, keyword_scores, depth_scores])
y = df["match_score"].values  # ground truth: 0.05–0.98

print(f"✅ Feature matrix shape: {X.shape}")
print(f"✅ Target range: [{y.min():.3f}, {y.max():.3f}]")

# ── 7. Train/Test Split ───────────────────────────────────────
print("\n✂️  Splitting into train/test sets (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"✅ Train: {len(X_train)} samples | Test: {len(X_test)} samples")

# ── 8. Train Gradient Boosting Regressor ─────────────────────
print("\n🚀 Training Gradient Boosting Regressor...")

model = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    min_samples_split=5,
    min_samples_leaf=3,
    subsample=0.8,
    random_state=42
)

model.fit(X_train, y_train)
print("✅ Training complete!")

# ── 9. Evaluate Model ─────────────────────────────────────────
print("\n📈 Evaluating model...")

y_pred = model.predict(X_test)
y_pred_clipped = np.clip(y_pred, 0.0, 1.0)

mae  = mean_absolute_error(y_test, y_pred_clipped)
r2   = r2_score(y_test, y_pred_clipped)

# Cross-validation on full dataset
cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2")

print(f"\n{'='*40}")
print(f"  MAE  (Test Set)     : {mae:.4f}")
print(f"  R²   (Test Set)     : {r2:.4f}")
print(f"  R²   (5-Fold CV)    : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"{'='*40}")

# Feature importance
feature_names = ["Semantic Similarity", "Keyword Overlap", "Depth Score"]
importances = model.feature_importances_
print("\n🔍 Feature Importances:")
for name, imp in zip(feature_names, importances):
    bar = "█" * int(imp * 40)
    print(f"  {name:<25} {imp:.4f}  {bar}")

# ── 10. Save Model + TF-IDF + Embedder Info ──────────────────
print("\n💾 Saving model pipeline to model.pkl ...")

model_bundle = {
    "model"       : model,
    "tfidf"       : tfidf,
    "feature_names": feature_names,
    "depth_indicators": DEPTH_INDICATORS,
    "embedder_name": "all-MiniLM-L6-v2",  # name only; load fresh in app
    "train_mae"   : mae,
    "train_r2"    : r2,
    "cv_r2_mean"  : cv_scores.mean(),
    "cv_r2_std"   : cv_scores.std(),
}

with open("model.pkl", "wb") as f:
    pickle.dump(model_bundle, f)

print("✅ model.pkl saved successfully!")
print("\n🎉 Training pipeline complete. Run app.py to start scoring resumes.")
