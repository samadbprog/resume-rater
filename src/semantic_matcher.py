import os
import re
from sentence_transformers import SentenceTransformer, util
from resume_parser import parse_resume_structured

print("\n⚠️ V2 SCRIPT ACTIVE - LOADING MODEL...")
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_requirement_level_similarity(jd_text, resume_bullets):
    # 1. Split JD by bullet points or newlines
    raw_reqs = re.split(r'\n|•', jd_text)
    
    # 2. Clean and filter out junk
    requirements = []
    ignore_phrases = ["responsibilities:", "requirements:", "nice to have", "full-time", "hyderabad", "we are looking", "you will work", "junior data scientist"]
    
    for r in raw_reqs:
        r = r.strip()
        if len(r) > 20 and not any(phrase in r.lower() for phrase in ignore_phrases):
            requirements.append(r)
    
    print(f"\n✅ Extracted {len(requirements)} real requirements from JD (filtered out junk).\n")
    
    if not requirements or not resume_bullets:
        return 0.0

    # 3. Embed requirements and resume bullets
    req_embeddings = model.encode(requirements, convert_to_tensor=True)
    bullet_embeddings = model.encode(resume_bullets, convert_to_tensor=True)

    # 4. Compute cosine similarity
    cosine_scores = util.cos_sim(req_embeddings, bullet_embeddings)

    best_scores = []
    for i in range(len(requirements)):
        max_score = float(cosine_scores[i].max())
        best_scores.append(max_score)
        
        best_bullet_idx = int(cosine_scores[i].argmax())
        print(f"Req: {requirements[i][:60]}...")
        print(f"  -> Match ({max_score:.2f}): {resume_bullets[best_bullet_idx][:60]}...")

    # 5. Average the best scores
    avg_max_score = sum(best_scores) / len(best_scores)
    
    # 6. NEW RESCALING LOGIC (Fixing the 0.00 issue)
    # If the average match is around 0.17, we subtract 0.05 and divide by 0.40.
    rescaled_score = max(0.0, min(1.0, (avg_max_score - 0.05) / 0.40))
    
    return rescaled_score


# ============================================================
# TEST BLOCK
# ============================================================
if __name__ == "__main__":
    base_dir = r"C:\Samad\SundarPitchTheAi\Projects\semantic-resume-scorer\Sample space\Simulated SS"
    pdf_path = os.path.join(base_dir, "R1.pdf")
    jd_path = os.path.join(base_dir, "Job Description Sample.txt")

    with open(jd_path, "r", encoding="utf-8") as f:
        jd_text = f.read()

    parsed = parse_resume_structured(pdf_path)
    resume_bullets = parsed["experience"] + parsed["projects"]

    print("\n" + "="*50)
    print("REQUIREMENT-LEVEL SEMANTIC MATCHING (V2)")
    print("="*50)
    
    final_sem_score = get_requirement_level_similarity(jd_text, resume_bullets)
    
    print("\n" + "="*50)
    print(f"FINAL SEMANTIC SCORE: {final_sem_score:.2f} / 1.0")
    print("="*50)