import os
import glob
import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from resume_parser import parse_resume_structured
from final_scorer import calculate_final_score

HUMAN_SCORES = {
    "01_Aarav_Mehta_Resume": 9.8, "02_Priya_Sharma_Resume": 9.2, "03_Rohan_Verma_Resume": 8.9,
    "04_Sneha_Reddy_Resume": 8.6, "05_Karthik_Nair_Resume": 9.5, "06_Aditya_Rao_Resume": 8.2,
    "07_Ishita_Kapoor_Resume": 8.7, "08_Varun_Gupta_Resume": 6.8, "09_Meera_Iyer_Resume": 9.6,
    "10_Rahul_Kumar_Resume": 5.4, "11_Nikhil_Menon_Resume": 8.4, "12_Ananya_Das_Resume": 9.1,
    "13_Arjun_Patel_Resume": 7.2, "14_Diya_Fernandes_Resume": 8.1, "15_Siddharth_Jain_Resume": 9.9,
    "16_Pooja_Singh_Resume": 4.8, "17_Harsh_Vardhan_Resume": 7.8, "18_Neha_Kulkarni_Resume": 8.8,
    "19_Mohit_Agarwal_Resume": 3.9, "20_Tanya_Bose_Resume": 8.5, "21_Abhishek_Reddy_Resume": 6.2,
    "22_Kavya_Ramesh_Resume": 9.0, "23_Manish_Sinha_Resume": 7.5, "24_Aishwarya_Nambiar_Resume": 7.0,
    "25_Dev_Malhotra_Resume": 6.5
}

base_dir = r"C:\Samad\SundarPitchTheAi\Projects\semantic-resume-scorer\Sample space\Simulated SS"
jd_path = os.path.join(base_dir, "Job Description Sample.txt")

with open(jd_path, "r", encoding="utf-8") as f:
    jd_text = f.read()

required_skills = ["Python", "SQL", "Machine Learning", "Git"]

raw_scores = []
human_scores = []

for filename_base, h_score in HUMAN_SCORES.items():
    matching_files = glob.glob(os.path.join(base_dir, f"{filename_base}.*"))
    if not matching_files: continue
    parsed = parse_resume_structured(matching_files[0])
    results = calculate_final_score(parsed, jd_text, required_skills)
    raw_scores.append(results["final_score_out_of_10"])
    human_scores.append(h_score)

X = np.array(raw_scores).reshape(-1, 1)
y = np.array(human_scores)

iso_reg = IsotonicRegression(out_of_bounds='clip')
iso_reg.fit(X.flatten(), y)

# Save to models folder
joblib.dump(iso_reg, "models/calibrator.pkl")
print("✅ Calibrator saved to models/calibrator.pkl")