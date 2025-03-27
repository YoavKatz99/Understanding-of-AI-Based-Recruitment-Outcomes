import os
import pandas as pd
import numpy as np
import re
import spacy
from pdfminer.high_level import extract_text
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

# Load SpaCy model
nlp = spacy.load("en_core_web_sm")
stopwords = nlp.Defaults.stop_words

# Skills list
list_of_skills = [
    'data science', 'computer vision', 'natural language processing', 'ai', 'ml',
    'machine learning', 'deep learning', 'logistic regression', 'classification',
    'scikit learn', 'numpy', 'pandas', 'matplotlib', 'seaborn', 'tensorflow',
    'keras', 'pytorch', 'cnn', 'rnn', 'nlp', 'opencv', 'django', 'mongodb', 'sql'
]

# Clean and process text
def process_text(text):
    doc = nlp(text.lower())
    return " ".join([token.lemma_ for token in doc if token.text not in stopwords])

# Extract features from text
def extract_features(text):
    clean = process_text(text)
    return {skill: int(skill in clean) for skill in list_of_skills}

# Load resume PDFs and extract features
def load_resumes_features(folder_path):
    data = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            candidate_id = filename.replace(".pdf", "")
            full_path = os.path.join(folder_path, filename)
            try:
                text = extract_text(full_path)
                features = extract_features(text)
                features["CandidateID"] = candidate_id
                data.append(features)
            except Exception as e:
                print(f"❌ Error reading {filename}: {e}")
    return pd.DataFrame(data)

# Paths
resume_folder = "backend/trainResumes"
train_csv_path = "backend/train.csv"


# Load data
features_df = load_resumes_features(resume_folder)
print(f"✅ Loaded {len(features_df)} resumes")

train_df = pd.read_csv(train_csv_path)
merged_df = train_df.merge(features_df, on="CandidateID", how="inner")
merged_df.to_csv("training_data.csv", index=False)
print("✅ Saved training_data.csv")

# Train model
X = merged_df.drop(columns=["CandidateID", "Match Percentage"])
y = merged_df["Match Percentage"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

print(f"✅ Model trained. MAE: {mae}")
joblib.dump(model, "random_forest_model.pkl")
print("✅ Model saved to random_forest_model.pkl")
