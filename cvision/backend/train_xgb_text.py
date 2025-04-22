# train_xgb_text.py
import os
import pandas as pd
import numpy as np
from pdfminer.high_level import extract_text
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

resume_folder = "trainResumes"
train_csv_path = "train.csv"
model_path = "xgb_text_model.pkl"
vectorizer_path = "tfidf_vectorizer.pkl"

def load_resume_texts(folder_path):
    texts = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            path = os.path.join(folder_path, filename)
            text = extract_text(path)
            texts[filename.replace(".pdf", "")] = text
    return texts

print("📄 Loading resume texts...")
resume_texts = load_resume_texts(resume_folder)

print("📊 Loading training labels...")
df = pd.read_csv(train_csv_path)

print("🔗 Merging texts with labels...")
df["text"] = df["CandidateID"].map(resume_texts)
df.dropna(subset=["text"], inplace=True)

vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df["text"])
y = (df["Match Percentage"] >= 50).astype(int)  # convert to binary classification

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("🚀 Training XGBoost Text Classifier...")
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Trained. Accuracy: {acc:.2f}")

print("💾 Saving model and vectorizer...")
joblib.dump(model, model_path)
joblib.dump(vectorizer, vectorizer_path)
