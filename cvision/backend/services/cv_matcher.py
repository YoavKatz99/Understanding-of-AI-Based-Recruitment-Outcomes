
# cv_matcher.py

import spacy
from pdfminer.high_level import extract_text
import numpy as np
import re

nlp = spacy.load("en_core_web_sm")
stopwords = nlp.Defaults.stop_words

# רשימת הטכנולוגיות שאנחנו מחפשים בקו"ח
TECH_SKILLS = ['scala', 'nosql', 'pandas', 'pytorch', 'keras', 'nlp', 'numpy', 'mongodb',
               'tensorflow', 'azure', 'aws', 'scikit learn', 'matplotlib', 'seaborn', 'nltk',
               'spacy', 'beautiful soup', 'pyspark', 'hadoop', 'computer vision', 'opencv',
               'django', 'graphql', 'deep learning', 'matlab']

def clean_text(text):
    # למטט ולסנן מילים מיותרות
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if token.lemma_.lower() not in stopwords]
    return ' '.join(tokens).lower()

def extract_year_from_text(text):
    match = re.search(r'(20[0-2][0-9]|19[9][0-9])', text)
    return int(match.group(1)) if match else None

def get_tech_vector(text):
    return np.array([1 if tech in text else 0 for tech in TECH_SKILLS])

def match_resume_to_job(cv_path, job_description_text):
    # קריאת קובץ PDF
    raw_text = extract_text(cv_path)
    cleaned_text = clean_text(raw_text)

    # וקטור טכנולוגיות
    tech_vector = get_tech_vector(cleaned_text)
    job_vector = get_tech_vector(clean_text(job_description_text))

    # חישוב אחוז ההתאמה
    if job_vector.sum() == 0:
        match_score = 0.0
    else:
        match_score = (tech_vector * job_vector).sum() / job_vector.sum()

    # תוספת "בונוס" לפי כמות טכנולוגיות בקו"ח
    bonus = tech_vector.sum() - np.mean([tech_vector.sum()])
    final_score = match_score * 10 + bonus

    return round(final_score, 2), cleaned_text, tech_vector.sum()

def match_resume_to_job_from_text(raw_text, job_description_text):
    cleaned_text = clean_text(raw_text)
    tech_vector = get_tech_vector(cleaned_text)
    job_vector = get_tech_vector(clean_text(job_description_text))

    tech_count = tech_vector.sum()
    match_score = np.dot(tech_vector, job_vector) / (np.linalg.norm(tech_vector) * np.linalg.norm(job_vector) + 1e-5)

    return match_score, cleaned_text, int(tech_count)
