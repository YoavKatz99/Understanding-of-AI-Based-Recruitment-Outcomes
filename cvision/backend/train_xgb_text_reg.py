# train_xgb_text_reg.py
import os
import pandas as pd
import numpy as np
from pdfminer.high_level import extract_text
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

resume_folder = "trainResumes"
train_csv_path = "train.csv"
model_path = "xgb_text_model_reg.pkl"  # regression model
vectorizer_path = "tfidf_vectorizer.pkl"  # load existing

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

# === Load the vectorizer used for classifier ===
print("\n📦 Loading EXISTING vectorizer...")
vectorizer = joblib.load(vectorizer_path)

# === Feature extraction ===
X = vectorizer.transform(df["text"])
y = df["Match Percentage"]  # regression target

# === Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Train model ===
print("🚀 Training XGBoost Regressor...")
model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# === Evaluate ===
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"✅ MAE: {mae:.2f}")
print(f"✅ R^2: {r2:.2f}")

# === Save ONLY model ===
print("💾 Saving regressor model...")
joblib.dump(model, model_path)

# (No need to save vectorizer again!)
