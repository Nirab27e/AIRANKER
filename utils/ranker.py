
import os
import fitz  # PyMuPDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def rank_resumes(jd_path, resume_paths):
    jd_text = extract_text_from_pdf(jd_path)
    resumes_texts = [extract_text_from_pdf(p) for p in resume_paths]

    docs = [jd_text] + resumes_texts
    vectorizer = TfidfVectorizer().fit_transform(docs)
    scores = cosine_similarity(vectorizer[0:1], vectorizer[1:]).flatten()

    ranked = sorted(zip(resume_paths, scores), key=lambda x: x[1], reverse=True)
    return [{"filename": os.path.basename(f), "score": float(s)} for f, s in ranked]
