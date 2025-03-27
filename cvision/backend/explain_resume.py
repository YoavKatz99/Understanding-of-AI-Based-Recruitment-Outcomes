import joblib
import shap
import lime
import lime.lime_tabular
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import spacy
from pdfminer.high_level import extract_text

# === הגדרות בסיסיות ===
model_path = "random_forest_model.pkl"
resume_path = "test_resume.pdf"

TECH_SKILLS = [
    'data_science', 'computer_vision', 'natural_language_processing', 'ai', 'ml',
    'machine_learning', 'deep_learning', 'logistic_regression', 'classification',
    'scikit_learn', 'numpy', 'pandas', 'matplotlib', 'seaborn', 'tensorflow',
    'keras', 'pytorch', 'cnn', 'rnn', 'nlp', 'opencv', 'django', 'mongodb', 'sql'
]

# === עיבוד טקסט וחילוץ מאפיינים ===
nlp = spacy.load("en_core_web_sm")

def process_text(text):
    doc = nlp(text.lower())
    return " ".join([token.lemma_ for token in doc])

def extract_features_from_text(text):
    text = process_text(text)
    features = {skill: int(skill.replace('_', ' ') in text) for skill in TECH_SKILLS}
    print("\n📊 Extracted Features:")
    print(features)
    return pd.DataFrame([features])

# === שליפת טקסט מהקובץ ===
print("\n📄 Extracting text from PDF...")
text = extract_text(resume_path)
print("\n📄 Extracted Resume Text:")
print(text)

# === חיזוי באמצעות המודל ===
print("\n📦 Loading model...")
model = joblib.load(model_path)

features_df = extract_features_from_text(text)

# טען את שמות הפיצ'רים המקוריים מהאימון
with open("feature_names.txt") as f:
    feature_order = [line.strip() for line in f]

# ודא שכל העמודות קיימות ושהסדר נכון
for feat in feature_order:
    if feat not in features_df.columns:
        features_df[feat] = 0

features_df = features_df[feature_order]  # סדר העמודות הנכון

prediction = model.predict(features_df)[0]
print(f"\n🔎 Match Score Prediction: {round(prediction, 2)}")

# === הסבר עם SHAP ===
print("\n📌 Top SHAP contributions:")
explainer = shap.Explainer(model, features_df)
shap_values = explainer(features_df)
shap_df = pd.DataFrame({
    "feature": features_df.columns,
    "shap_value": shap_values.values[0]
}).sort_values(by="shap_value", ascending=False)
print(shap_df.head())

shap.plots.bar(shap_values[0], show=False)
plt.tight_layout()
plt.savefig("shap_importance.png")
print("📊 SHAP plot saved to shap_importance.png")

# === הסבר עם LIME ===
print("\n📌 Top LIME contributions:")
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=np.array(features_df),
    feature_names=features_df.columns.tolist(),
    mode="regression"
)

lime_exp = lime_explainer.explain_instance(
    data_row=features_df.iloc[0].values,
    predict_fn=model.predict,
    num_features=5
)

lime_exp.save_to_file("lime_explanation.html")
print("📄 LIME explanation saved to lime_explanation.html")
