import os
import numpy as np
from sklearn.isotonic import IsotonicRegression
from resume_parser import parse_resume_structured
from final_scorer import calculate_final_score

# All 5 resumes are back in the Gold Set!
HUMAN_SCORES = {
    "R1.pdf": 3.0,
    "R2.pdf": 4.5,
    "R3.pdf": 6.5,
    "R4.pdf": 8.5,
    "R5.pdf": 5.5
}

def train_calibrator(base_dir, jd_text, required_skills):
    raw_scores = []
    human_scores = []

    print("Training calibration on Gold Set...")
    for filename, h_score in HUMAN_SCORES.items():
        pdf_path = os.path.join(base_dir, filename)
        if not os.path.exists(pdf_path):
            print(f"⚠️ Skipping {filename} - File not found!")
            continue
            
        parsed = parse_resume_structured(pdf_path)
        results = calculate_final_score(parsed, jd_text, required_skills)
        
        raw_scores.append(results["final_score_out_of_10"])
        human_scores.append(h_score)
        print(f"  Processed {filename} (Raw: {results['final_score_out_of_10']} -> Human: {h_score})")

    # Fit the curve
    X = np.array(raw_scores).reshape(-1, 1)
    y = np.array(human_scores)
    
    iso_reg = IsotonicRegression(out_of_bounds='clip')
    iso_reg.fit(X.flatten(), y)
    
    print("✅ Calibration Curve Trained!\n")
    return iso_reg

def score_new_resume(pdf_filename, base_dir, jd_text, required_skills, calibrator):
    pdf_path = os.path.join(base_dir, pdf_filename)
    
    if not os.path.exists(pdf_path):
        print(f"❌ Could not find {pdf_filename}")
        return

    parsed = parse_resume_structured(pdf_path)
    
    # 1. Get Raw Score
    results = calculate_final_score(parsed, jd_text, required_skills)
    raw_score = results["final_score_out_of_10"]
    
    # 2. Apply Calibration
    calibrated_score = float(calibrator.predict([raw_score])[0])
    
    # 3. Print the Final Human-Readable Output
    print("\n" + "="*40)
    print(f"RESUME: {pdf_filename}")
    print("="*40)
    print(f"Raw System Score:  {raw_score} / 10")
    print(f"CALIBRATED SCORE:  {calibrated_score:.1f} / 10")
    
    # 4. Print the Breakdown so you know WHY
    print("\n--- Breakdown ---")
    print(f"Semantic Match:    {results['breakdown']['sem_sim']}")
    print(f"Keyword Evidence:  {results['breakdown']['kw_overlap']}")
    print(f"Depth Score:       {results['breakdown']['depth']}")
    print("="*40 + "\n")


if __name__ == "__main__":
    base_dir = r"C:\Samad\SundarPitchTheAi\Projects\semantic-resume-scorer\Sample space\Simulated SS"
    jd_path = os.path.join(base_dir, "Job Description Sample.txt")

    with open(jd_path, "r", encoding="utf-8") as f:
        jd_text = f.read()

    required_skills = ["Python", "SQL", "Machine Learning", "Git"]

    # 1. Train the calibrator on our Gold Set
    calibrator = train_calibrator(base_dir, jd_text, required_skills)

    # 2. Test it on R4 (Divya) to see if it pushes her up to 8.5
    score_new_resume("R4.pdf", base_dir, jd_text, required_skills, calibrator)
    
    # Test it on R1 (Karthik) to see if it keeps him low
    score_new_resume("R1.pdf", base_dir, jd_text, required_skills, calibrator)