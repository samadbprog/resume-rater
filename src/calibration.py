import os
import numpy as np
from sklearn.isotonic import IsotonicRegression
from resume_parser import parse_resume_structured
from final_scorer import calculate_final_score

# Your Human Gold Set
HUMAN_SCORES = {
    "R1.pdf": 3.0,
    "R2.pdf": 4.5,
    "R3.pdf": 6.5,
    "R4.pdf": 8.5,
    "R5.pdf": 5.5
}

def run_calibration():
    base_dir = r"C:\Samad\SundarPitchTheAi\Projects\semantic-resume-scorer\Sample space\Simulated SS"
    jd_path = os.path.join(base_dir, "Job Description Sample.txt")

    with open(jd_path, "r", encoding="utf-8") as f:
        jd_text = f.read()

    # Must-have skills from the JD
    required_skills = ["Python", "SQL", "Machine Learning", "Git"]

    raw_system_scores = []
    human_scores = []

    print("="*60)
    print("STEP 1: SCORING ALL RESUMES (RAW SYSTEM OUTPUT)")
    print("="*60)

    for filename, human_score in HUMAN_SCORES.items():
        pdf_path = os.path.join(base_dir, filename)
        parsed = parse_resume_structured(pdf_path)
        
        # Call our pipeline from Step 3
        results = calculate_final_score(parsed, jd_text, required_skills)
        
        raw_score = results["final_score_out_of_10"]
        
        raw_system_scores.append(raw_score)
        human_scores.append(human_score)
        
        print(f"{filename:<10} | Raw Score: {raw_score:<5} | Human Score: {human_score}")

    # Convert to numpy arrays for the math
    X = np.array(raw_system_scores).reshape(-1, 1)
    y = np.array(human_scores)

    print("\n" + "="*60)
    print("STEP 2: FITTING CALIBRATION CURVE (ISOTONIC REGRESSION)")
    print("="*60)

    # Isotonic regression finds the best monotonic mapping from raw to human
    iso_reg = IsotonicRegression(out_of_bounds='clip')
    iso_reg.fit(X.flatten(), y)

    print("Calibration curve fitted successfully.")
    
    # Test the calibration on our 5 resumes to see the difference
    calibrated_scores = iso_reg.predict(X.flatten())

    print("\n--- CALIBRATED RESULTS ---")
    print(f"{'File':<10} | {'Human':<6} | {'Raw':<6} | {'Calibrated':<10}")
    print("-" * 45)
    for i, filename in enumerate(HUMAN_SCORES.keys()):
        print(f"{filename:<10} | {y[i]:<6} | {X[i][0]:<6} | {calibrated_scores[i]:.1f}")

    print("\n✅ Calibration complete. These are the scores the final UI should show.")

if __name__ == "__main__":
    run_calibration()