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
Rule Score = (0.20 × Semantic) + (0.40 × Keyword) + (0.40 × Depth)


### Final Hybrid Score
Final Score = (0.30 × Model Score) + (0.70 × Rule Score)


### Display Score
Display Score = clip(Final Score, 0, 1) × 10


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
semantic-resume-scorer/
│
├── app.py # Main Streamlit application
├── train.py # Model training script
├── model.pkl # Trained model bundle (model + metrics)
├── requirements.txt # Python dependencies
├── README.md # Project documentation
│
└── data/
└── resumeJD2_pairs.csv # Training dataset (500 resume–JD pairs)



---

## ⚙️ Installation

### 1. Clone the Repository

``bash
git clone https://github.com/samadbprog/semantic-resume-scorer.git
cd semantic-resume-scorer

### 2. Create a Virtual Environment

python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

### 3. Install Dependencies
pip install -r requirements.txt

Training Configuration
Parameter	Value
Model	GradientBoostingRegressor
n_estimators	300
learning_rate	0.05
max_depth	4
min_samples_split	5
min_samples_leaf	3
subsample	0.8
Train/Test Split	80/20
Cross-Validation	5-Fold
Training Results
Metric	Value
Test MAE	0.1865
Test R²	0.3207
CV R² Mean	0.3581 ± 0.0927
Feature Importances
Feature	Importance
Semantic Similarity	0.8487
Keyword Overlap	0.1041
Depth Score	0.0472
▶️ Running the App
bash
streamlit run app.py
Then open your browser at http://localhost:8501

🔍 How It Works
1. Resume Parsing
PDF files are parsed using PyPDF2

DOCX files are parsed using python-docx

2. Semantic Similarity
Both resume and JD are encoded using all-MiniLM-L6-v2

Cosine similarity is computed between the two embeddings

3. Keyword Overlap
Top 20 JD keywords are extracted via TF-IDF (ngram_range=(1,2), max_features=200)

Overlap ratio = matched keywords / 20

4. Depth Score
Checks for presence of technical depth indicators in the resume:

Hits	Depth Score
≥ 8	1.00
≥ 5	0.67
≥ 3	0.33
< 3	0.00
Current Depth Indicators:
auc-roc, precision, recall, f1, %, improved,
fine-tuned, pipeline, deployed, kaggle

5. Hybrid Scoring
The ML model prediction and rule-based score are blended:

Component	Weight
Model Score	30%
Rule Score	70%
💬 Feedback System
Dimension	Threshold	Feedback
Semantic	< 0.4	⚠️ Low semantic similarity
Semantic	> 0.7	✅ High semantic similarity
Semantic	0.4 – 0.7	🔶 Moderate semantic similarity
Keywords	< 0.3	⚠️ Low keyword coverage
Keywords	≥ 0.6	✅ Strong keyword coverage
Keywords	0.3 – 0.6	🔶 Moderate keyword coverage
Depth	0.0	⚠️ Low depth — add metrics/specifics
Depth	≥ 0.67	✅ Strong depth
Depth	0.33	🔶 Moderate depth
🛠️ Built With
Streamlit — Web UI

SentenceTransformers — Semantic embeddings

scikit-learn — TF-IDF + Gradient Boosting

PyPDF2 — PDF parsing

python-docx — DOCX parsing

📌 Dataset
Trained on the
Resume–JD Matching Dataset
from Kaggle — 500 validated resume–job description pairs.

🔮 Future Improvements
Expand DEPTH_INDICATORS across multiple domains

Add domain detection (ML, Finance, Marketing, etc.)

Improve poor/mediocre score separation

Add resume section parser (Skills, Experience, Education)

Support multi-page PDF resumes more robustly

Add batch scoring for multiple resumes

🤝 Contributing
Contributions are welcome! Here's how to get started:

Fork the repository

Create a new branch

bash
git checkout -b feature/your-feature-name
Make your changes and commit

bash
git commit -m "Add: your feature description"
Push to your branch

bash
git push origin feature/your-feature-name
Open a Pull Request

Please make sure your code is clean, commented, and tested before submitting.

🐛 Reporting Issues
Found a bug or have a suggestion? Please open an issue on GitHub:

Describe the problem clearly

Include steps to reproduce

Attach any error messages or screenshots if possible

📄 License
This project is licensed under the MIT License — feel free to use,
modify, and distribute this project with attribution.

See the LICENSE file for full details.

🙌 Acknowledgements
Hugging Face for the all-MiniLM-L6-v2 model

Kaggle for the training dataset

Streamlit for making ML apps effortless

The open-source community for the amazing libraries that power this project

📬 Contact
For questions, feedback, or collaboration:

GitHub: samadbprog

Email: beparsamad@gmail.com

⭐ Show Your Support
If you found this project useful, consider giving it a star ⭐ on GitHub —
it helps others discover the project and motivates further development!

Built with ❤️ using Python, Streamlit, and SentenceTransformers.
