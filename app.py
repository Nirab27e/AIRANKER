import os
from flask import Flask, render_template, request, send_file
import spacy
import pandas as pd
from werkzeug.utils import secure_filename
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import csv


# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Flask app setup
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Function to extract and preprocess resume text
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text()
    return text

def preprocess(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return ' '.join(tokens)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/rank', methods=['POST'])
def rank_resumes():
    uploaded_files = request.files.getlist("resumes")
    jd_text = request.form['job_description']

    resume_texts = [jd_text]
    filenames = []
    

    upload_folder = "uploads"
    os.makedirs(upload_folder, exist_ok=True)

    for file in uploaded_files:
        if file.filename.endswith(".pdf"):
            filepath = os.path.join(upload_folder, file.filename)
            file.save(filepath)
            text = extract_text_from_pdf(filepath)
            resume_texts.append(text)
            filenames.append(file.filename)

    preprocessed = [preprocess(text) for text in resume_texts]
    tfidf = TfidfVectorizer()
    vectors = tfidf.fit_transform(preprocessed).toarray()

    scores = cosine_similarity([vectors[0]], vectors[1:])[0]
    ranked = sorted(zip(filenames, scores), key=lambda x: x[1], reverse=True)

    output_path = 'output/ranked_resumes.csv'
    os.makedirs('output', exist_ok=True)
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Filename', 'Score'])
        writer.writerows(ranked)
    
    import matplotlib
    matplotlib.use('Agg')  # Use the Agg backend for non-GUI environments
    import matplotlib.pyplot as plt

    filenames_clean = [f.split('.')[0] for f in filenames]  # Shorter names
    plt.figure(figsize=(10, 5))
    plt.barh(filenames_clean[::-1], scores[::-1], color='skyblue')
    plt.xlabel('Score')
    plt.title('Resume Similarity Scores')
    plt.tight_layout()

    output_path = os.path.join('static', 'scores_plot.png')
    plt.savefig(output_path)
    plt.close()



    return render_template("results.html", ranked=ranked, plot_url='static/scores_plot.png')



if __name__ == '__main__':
    app.run(debug=True)

