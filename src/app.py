import streamlit as st
from resume_parser import parse_resume
from scorer import score_resume, get_feedback
import tempfile
import os

# Page config
st.set_page_config(
    page_title="Semantic Resume Scorer",
    page_icon="📄",
    layout="centered"
)

# Title
st.title("📄 Semantic Resume Scorer")
st.markdown("Upload your **resume** and paste a **job description** to see how well you match!")

st.divider()

# --- Job Description Input ---
st.subheader("📋 Job Description")
job_description = st.text_area(
    "Paste the job description here:",
    height=200,
    placeholder="e.g. We are looking for a Python developer with experience in machine learning..."
)

st.divider()

# --- Resume Upload ---
st.subheader("📎 Upload Your Resume")
uploaded_file = st.file_uploader(
    "Upload your resume (PDF or DOCX):",
    type=["pdf", "docx"]
)

st.divider()

# --- Score Button ---
if st.button("🚀 Score My Resume"):
    if not job_description.strip():
        st.warning("⚠️ Please paste a job description first.")
    elif uploaded_file is None:
        st.warning("⚠️ Please upload your resume first.")
    else:
        with st.spinner("Analyzing your resume... please wait ⏳"):
            # Save uploaded file to a temp location
            suffix = os.path.splitext(uploaded_file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            # Parse and score
            resume_text = parse_resume(tmp_path)
            score = score_resume(resume_text, job_description)
            feedback = get_feedback(score)

            # Cleanup temp file
            os.remove(tmp_path)

        # --- Display Results ---
        st.divider()
        st.subheader("📊 Results")

        col1, col2 = st.columns(2)
        with col1:
            st.metric(label="Match Score", value=f"{score} / 10")
        with col2:
            st.markdown(f"### {feedback}")

        # Score bar
        st.progress(int(score * 10))
