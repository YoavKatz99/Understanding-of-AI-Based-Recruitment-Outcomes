# train_xgb_text.py
import os
import pandas as pd
import numpy as np
from pdfminer.high_level import extract_text
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import matplotlib.pyplot as plt

# === Config ===
resume_folder = "trainResumes"
train_csv_path = "train.csv"
model_path = "xgb_text_model.pkl"
vectorizer_path = "tfidf_vectorizer.pkl"

# === Load resume texts ===
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

# Filter out samples with missing or too short texts
df = df.dropna(subset=["text"])
df = df[df["text"].str.len() > 100]  # Only keep meaningful resumes

# === Feature extraction ===
vectorizer = TfidfVectorizer(
    max_features=3000,      # 🔥 more features -> better capture of skills
    stop_words="english",   # 🔥 removes meaningless common words
    ngram_range=(1,2)       # 🔥 use unigrams + bigrams ("machine learning")
)
X = vectorizer.fit_transform(df["text"])
y = df["Match Percentage"]

print(f"📝 Dataset size after filtering: {X.shape}")

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
print(f"✅ R^2 score: {r2:.2f}")

# Plot real vs predicted
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("True Match %")
plt.ylabel("Predicted Match %")
plt.title("True vs Predicted")
plt.grid(True)
plt.show()

# === Save model and vectorizer ===
print("💾 Saving model and vectorizer...")
joblib.dump(model, model_path)
joblib.dump(vectorizer, vectorizer_path)
