import os
import pandas as pd
import numpy as np
from pdfminer.high_level import extract_text
import spacy
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score
import joblib

# הגדרות
resume_folder = "trainResumes"
train_csv_path = "train.csv"
model_path = "xgboost_model.pkl"
feature_names_path = "feature_names.txt"

# רשימת מיומנויות
TECH_SKILLS = [
    'data_science', 'computer_vision', 'natural_language_processing', 'ai', 'ml',
    'machine_learning', 'deep_learning', 'logistic_regression', 'classification',
    'scikit_learn', 'numpy', 'pandas', 'matplotlib', 'seaborn', 'tensorflow',
    'keras', 'pytorch', 'cnn', 'rnn', 'nlp', 'opencv', 'django', 'mongodb', 'sql'
]

# עיבוד טקסט
nlp = spacy.load("en_core_web_sm")
def process_text(text):
    doc = nlp(text.lower())
    return " ".join([token.lemma_ for token in doc])

def extract_features(text):
    text = process_text(text)
    return {skill: int(skill.replace("_", " ") in text) for skill in TECH_SKILLS}

def load_resumes_features(folder_path):
    data = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            path = os.path.join(folder_path, filename)
            text = extract_text(path)
            features = extract_features(text)
            features["CandidateID"] = filename.replace(".pdf", "")
            data.append(features)
    return pd.DataFrame(data)

print("📥 Loading resumes...")
features_df = load_resumes_features(resume_folder)
print(f"✅ Loaded {len(features_df)} resumes")

print("📥 Loading training CSV...")
train_df = pd.read_csv(train_csv_path)

print("🔗 Merging features with labels...")
df = train_df.merge(features_df, on="CandidateID", how="inner")
X = df.drop(columns=["CandidateID", "Match Percentage"])
y = df["Match Percentage"]

print("🎯 Training XGBoost Regressor...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"✅ Model trained. MAE: {mae:.2f}")

print(f"💾 Model saved to {model_path}")
joblib.dump(model, model_path)

print(f"📝 Feature names saved to {feature_names_path}")
with open(feature_names_path, "w") as f:
    for col in X.columns:
        f.write(col + "\n")