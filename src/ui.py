import streamlit as st
import os
import sys
import joblib
import re

# Add src to path
sys.path.append(os.path.abspath("src"))

from resume_parser import parse_resume_structured
from final_scorer import calculate_final_score

st.set_page_config(page_title="AI Resume Scorer", page_icon="🚀", layout="wide")

st.title("🚀 Semantic Resume Scorer")

# --- 1. Load the pre-trained calibrator ---
@st.cache_resource
def load_calibrator():
    return joblib.load("models/calibrator.pkl")

try:
    calibrator = load_calibrator()
except Exception as e:
    st.error(f"Calibrator not found. Please run `python src/save_calibrator.py` first. Error: {e}")
    st.stop()

# --- 2. Auto Skill Extractor ---
def extract_skills_from_jd(jd_text):
    # A basic dictionary of common skills to look for
    SKILL_DB = [
        "python", "sql", "machine learning", "ml", "deep learning", "tensorflow", 
        "pytorch", "keras", "scikit-learn", "sklearn", "pandas", "numpy", 
        "aws", "azure", "gcp", "docker", "kubernetes", "git", "java", "c++", 
        "tableau", "power bi", "excel", "nlp", "spark", "hadoop", "statistics",
        "data science", "data analysis", "etl", "api", "django", "react"
    ]
    
    jd_lower = jd_text.lower()
    found_skills = []
    
    for skill in SKILL_DB:
        # Use regex word boundaries so "R" doesn't match inside "React"
        if re.search(r'\b' + re.escape(skill) + r'\b', jd_lower):
            found_skills.append(skill.title()) # Capitalize for display
            
    return list(set(found_skills)) # Remove duplicates

# --- 3. The UI ---
st.markdown("Paste your Job Description and upload a resume to see the score.")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Job Description")
    jd_text = st.text_area("Paste the JD here:", height=300, placeholder="We are looking for a Junior Data Scientist...")

with col2:
    st.subheader("2. Upload Resume")
    uploaded_file = st.file_uploader("Upload a Resume (PDF)", type=["pdf"])

# --- 4. Score ---
if st.button("Score Candidate", type="primary"):
    if not jd_text:
        st.warning("Please paste a Job Description.")
    elif not uploaded_file:
        st.warning("Please upload a Resume.")
    else:
        # Automatically extract skills
        required_skills = extract_skills_from_jd(jd_text)
        
        if not required_skills:
            st.warning("Could not detect any skills from the JD. Using defaults.")
            required_skills = ["Python", "SQL"]
            
        st.info(f"🤖 Auto-detected skills from JD: {', '.join(required_skills)}")
        
        # Save temp file
        temp_path = "temp_resume.pdf"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        with st.spinner("Scoring Resume..."):
            # 1. Parse
            parsed = parse_resume_structured(temp_path)
            
            # 2. Score
            results = calculate_final_score(parsed, jd_text, required_skills)
            
            # 3. Calibrate
            raw_score = results["final_score_out_of_10"]
            final_score = float(calibrator.predict([raw_score])[0])
            
            # 4. Display
            st.divider()
            st.header("Results")
            
            res_col1, res_col2 = st.columns([1, 2])
            
            with res_col1:
                st.metric(label="Final Calibrated Score", value=f"{final_score:.1f} / 10")
                if final_score >= 8.0:
                    st.success("✅ Interview Immediately")
                elif final_score >= 6.0:
                    st.warning("⚠️ Hold / Phone Screen")
                else:
                    st.error("❌ Reject")
                    
            with res_col2:
                st.write("**Breakdown of Scores**")
                st.progress(results["breakdown"]["sem_sim"], text=f"Semantic Match: {results['breakdown']['sem_sim']:.2f}")
                st.progress(results["breakdown"]["kw_overlap"], text=f"Keyword Evidence: {results['breakdown']['kw_overlap']:.2f}")
                st.progress(results["breakdown"]["depth"], text=f"Depth/Metrics: {results['breakdown']['depth']:.2f}")
                
        # Cleanup
        os.remove(temp_path)