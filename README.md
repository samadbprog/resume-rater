# 📄 AI Resume Scorer

A hybrid ML + Rule-Based resume scoring engine built with Streamlit.
Upload your resume, paste a job description, and get an intelligent
match score out of 10 — with detailed feedback and score breakdown.

---

## 🚀 Features

- 📁 Supports **PDF and DOCX** resume uploads
- 🧠 **Semantic similarity** using SentenceTransformer (`all-MiniLM-L6-v2`)
- 🔑 **Keyword overlap** extraction using TF-IDF (unigrams + bigrams)
- 📏 **Depth scoring** based on technical specificity indicators
- 🤖 **Gradient Boosting** ML model (trained on 500 resume–JD pairs)
- 📐 **Hybrid blend** of ML model score and rule-based score
- 💬 **Actionable feedback** for each scoring dimension
- 📊 **Live formula display** with actual computed values

---

## 🧮 Scoring Formula

### Rule-Based Score
$$
\text{Rule Score} = (0.20 \times \text{Semantic}) + (0.40 \times \text{Keyword}) + (0.40 \times \text{Depth})
$$

### Final Hybrid Score
$$
\text{Final Score} = (0.30 \times \text{Model Score}) + (0.70 \times \text{Rule Score})
$$

### Display Score
$$
\text{Display Score} = \text{clip}(\text{Final Score},\ 0,\ 1) \times 10
$$

---

## 📐 Score Interpretation

| Score Range | Label              |
|-------------|--------------------|
| 7.5 – 10.0  | Excellent Match 🌟 |
| 5.5 – 7.4   | Good Match 👍      |
| 3.5 – 5.4   | Moderate Match 🔶  |
| 0.0 – 3.4   | Poor Match ❌      |

---

## 🗂️ Project Structure

