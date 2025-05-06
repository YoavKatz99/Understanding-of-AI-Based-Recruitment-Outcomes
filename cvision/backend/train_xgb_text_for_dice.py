# train_xgb_dice_regressor.py
import os
import pandas as pd
import numpy as np
import joblib
from pdfminer.high_level import extract_text
import spacy
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# === Skills used for binary features
TECH_SKILLS = [
    'data_science', 'computer_vision', 'natural_language_processing', 'ai', 'ml',
    'machine_learning', 'deep_learning', 'logistic_regression', 'classification',
    'scikit_learn', 'numpy', 'pandas', 'matplotlib', 'seaborn', 'tensorflow',
    'keras', 'pytorch', 'cnn', 'rnn', 'nlp', 'opencv', 'django', 'mongodb', 'sql'
]

# === Paths
resume_folder = "trainResumes"
train_csv_path = "train.csv"
model_path = "xgb_regressor_model.pkl"
feature_list_path = "feature_list.pkl"

# === NLP Setup
nlp = spacy.load("en_core_web_sm")

def process_text(text):
    doc = nlp(text.lower())
    return " ".join([token.lemma_ for token in doc])

def extract_features(text):
    text = process_text(text)
    return {skill: float(skill.replace("_", " ") in text) for skill in TECH_SKILLS}

def load_resume_texts(folder):
    data = {}
    for file in os.listdir(folder):
        if file.endswith(".pdf"):
            path = os.path.join(folder, file)
            text = extract_text(path)
            if not text.strip():
                raise ValueError(f"Empty text extracted from {file}")
            data[file.replace(".pdf", "")] = text
    return data

print("📄 Extracting texts...")
resumes = load_resume_texts(resume_folder)
df = pd.read_csv(train_csv_path)
df["text"] = df["CandidateID"].map(resumes)
df = df.dropna(subset=["text"])

print("🧠 Extracting features...")
X = pd.DataFrame([extract_features(t) for t in df["text"]])
y = df["Match Percentage"] / 100.0  # Scale target to [0, 1] for logistic regression

print("🚀 Training regressor...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
model.fit(X_train, y_train)

pred_scaled = model.predict(X_test)
pred_percentage = np.clip(pred_scaled * 100, 0, 100)
mae = mean_absolute_error(y_test * 100, pred_percentage)
print(f"✅ Model trained. MAE: {mae:.2f}")

print("💾 Saving model and feature list...")
joblib.dump(model, model_path)
joblib.dump(TECH_SKILLS, feature_list_path)

print("✅ Regressor training complete.")
