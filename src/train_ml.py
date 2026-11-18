import os
import numpy as np
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
import joblib

def train_model():
    print("1. Downloading dataset from HuggingFace (this may take a minute)...")
    ds = load_dataset("dev6565/resume-ats-score-v1-en", split="train")
    
    df = ds.to_pandas()
    print(f"Loaded {len(df)} resume-JD pairs.")
    print("Columns found:", df.columns.tolist())
    
    # The dataset combines the resume and JD into a single 'text' column separated by '[SEP]'
    print("Splitting 'text' column into Resume and JD...")
    def split_text(text):
        if '[SEP]' in text:
            res, jd = text.split('[SEP]', 1)
            return res, jd
        return text, "" # Fallback if no [SEP] is found
        
    df['resume_text'], df['jd_text'] = zip(*df['text'].apply(split_text))
    
    # 2. Load our embedding model
    print("\n2. Loading embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # 3. Embed the data 
    print("\n3. Embedding 3000 resumes and JDs (this takes a few minutes)...")
    sample_df = df.head(3000) 
    
    resume_embeddings = model.encode(sample_df['resume_text'].astype(str).tolist(), show_progress_bar=True)
    jd_embeddings = model.encode(sample_df['jd_text'].astype(str).tolist(), show_progress_bar=True)
    
    # 4. Feature Engineering
    print("\n4. Calculating similarity features...")
    X = []
    for i in range(len(sample_df)):
        res_emb = resume_embeddings[i]
        jd_emb = jd_embeddings[i]
        
        # Cosine similarity
        cos_sim = np.dot(res_emb, jd_emb) / (np.linalg.norm(res_emb) * np.linalg.norm(jd_emb))
        
        # Euclidean distance
        euclidean = np.linalg.norm(res_emb - jd_emb)
        
        X.append([cos_sim, euclidean])
        
    X = np.array(X)
    
    # Target: Normalize score. The dataset scores are 0-100.
    y = sample_df['ats_score'].values / 100.0
    
    # 5. Train the Ridge Regression model
    print("\n5. Training Ridge Regression model...")
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    reg_model = Ridge(alpha=1.0)
    reg_model.fit(X_train, y_train)
    
    # 6. Evaluate
    predictions = reg_model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    print(f"Model Mean Absolute Error: {mae:.4f} (lower is better)")
    
    # 7. Save the model to disk
    # Save inside the root project folder, not the src folder
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(root_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, "ml_scorer.pkl")
    joblib.dump(reg_model, model_path)
    print(f"\n✅ Model saved to {model_path}")
    print("You can now update final_scorer.py to use this model!")

if __name__ == "__main__":
    train_model()