# 📄 AI Resume Scorer

Hybrid ML + Rule-Based resume scoring engine. Upload your resume, paste a job description, and get a match score out of 10 with detailed feedback.

**[Live Demo](https://resume-rater-gq9fnp88qjx4zud5jzf32v.streamlit.app)**

---

## 🚀 Features

- 📁 Supports **PDF and DOCX** uploads
- 🧠 **Semantic similarity** via SentenceTransformer (`all-MiniLM-L6-v2`)
- 🔑 **Keyword extraction** using TF-IDF (unigrams + bigrams)
- 📏 **Depth scoring** based on technical indicators
- 🤖 **Gradient Boosting** ML model trained on 500 resume–JD pairs
- 🔭 **Three-lens breakdown**: ATS Match, Overall Fit, Candidate Quality
- 💬 **Actionable feedback** for each scoring dimension

---

## 🧮 Scoring Formula

| Component  | Weight |
|------------|--------|
| ML Model   | 35%    |
| Rule-Based | 65%    |

**Rule Score** = (Semantic × 0.20) + (Keyword × 0.40) + (Depth × 0.40)

**Final Score** = Weighted average of three lenses → out of 10

---

## 📦 Installation

```bash
git clone https://github.com/samadbprog/resume-rater.git
cd resume-rater
pip install -r requirements.txt
streamlit run app.py
```

---

## 🗂️ Project Structure

```
resume-rater/
├── app.py              # Main Streamlit app
├── train.py            # Model training script
├── model.pkl           # Trained model (hosted on Hugging Face)
├── requirements.txt    # Dependencies
└── README.md           # This file
```

---

## 🛠️ Tech Stack

- [Streamlit](https://streamlit.io/) — UI
- [SentenceTransformers](https://www.sbert.net/) — Embeddings
- [scikit-learn](https://scikit-learn.org/) — TF-IDF + Gradient Boosting
- [Hugging Face Hub](https://huggingface.co/) — Model hosting

---

## 📌 Dataset

Trained on [Resume–JD Matching Dataset](https://www.kaggle.com/datasets/jainishkumar/resume-job-description-matching-dataset) from Kaggle (500 pairs).

---

## 📬 Contact

**GitHub:** [samadbprog](https://github.com/samadbprog)
**Email:** beparsamad@gmail.com

---

⭐ If you found this useful, consider giving it a star!
