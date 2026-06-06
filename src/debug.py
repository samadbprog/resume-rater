# debug.py
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

with open("model.pkl", "rb") as f:
    bundle = pickle.load(f)

model            = bundle["model"]
tfidf            = bundle["tfidf"]
DEPTH_INDICATORS = bundle["depth_indicators"]
embedder         = SentenceTransformer("all-MiniLM-L6-v2")

resume_text = "Experienced data scientist with Python, ML pipelines, and deployed models. Improved AUC-ROC by 15%."
jd_text     = "Looking for a data scientist with Python, machine learning, and pipeline experience."

print("--- Step 1: Semantic Similarity ---")
resume_emb = embedder.encode([resume_text])
jd_emb     = embedder.encode([jd_text])
dot        = np.sum(resume_emb * jd_emb, axis=1)
norm       = np.linalg.norm(resume_emb, axis=1) * np.linalg.norm(jd_emb, axis=1)
sem_sim    = float((dot / (norm + 1e-9))[0])
print("sem_sim:", sem_sim)

print("--- Step 2: TF-IDF Keyword Overlap ---")
jd_vec     = tfidf.transform([jd_text]).toarray()[0]
resume_vec = tfidf.transform([resume_text]).toarray()[0]
top_indices   = np.argsort(jd_vec)[-20:]
jd_top        = jd_vec[top_indices]
resume_top    = resume_vec[top_indices]
overlap       = np.sum((resume_top > 0) & (jd_top > 0))
keyword_score = overlap / 20.0
print("keyword_score:", keyword_score)

print("--- Step 3: Depth Score ---")
text_lower = resume_text.lower()
hits = sum(1 for ind in DEPTH_INDICATORS if ind in text_lower)
print("hits:", hits)

print("--- Step 4: Predict ---")
features  = np.array([[float(sem_sim), float(keyword_score), float(0.33)]])
raw_score = float(model.predict(features)[0])
print("raw_score:", raw_score)

print("--- ALL STEPS PASSED ✅ ---")
