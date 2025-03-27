import os
import pandas as pd
import shap
import lime.lime_tabular
import joblib
from pdfminer.high_level import extract_text
import spacy
import matplotlib.pyplot as plt

# הגדרות בסיס
resume_path = "backend/test_resume.pdf"
model_path = "backend/random_forest_model.pkl"

# טען את המודל
model = joblib.load(model_path)

# הגדר רשימת טכנולוגיות
TECH_SKILLS = [
    'data science', 'computer vision', 'natural language processing', 'ai', 'ml',
    'machine learning', 'deep learning', 'logistic regression', 'classification',
    'scikit learn', 'numpy', 'pandas', 'matplotlib', 'seaborn', 'tensorflow',
    'keras', 'pytorch', 'cnn', 'rnn', 'nlp', 'opencv', 'django', 'mongodb', 'sql'
]

# טען מודל של spaCy
nlp = spacy.load("en_core_web_sm")
stopwords = nlp.Defaults.stop_words

# פונקציית ניקוי טקסט
def process_text(text):
    doc = nlp(text.lower())
    return " ".join([token.lemma_ for token in doc if token.text not in stopwords])

# המרת קובץ PDF לטקסט
text = extract_text(resume_path)
cleaned_text = process_text(text)

# המרת טקסט לתכונות
features = {skill: 1 if skill in cleaned_text else 0 for skill in TECH_SKILLS}
df_features = pd.DataFrame([features])

# תחזית
prediction = model.predict(df_features)[0]
print(f"🔎 Match Score Prediction: {round(prediction, 2)}")

# 🔍 הסבר עם SHAP
explainer = shap.Explainer(model, df_features)
shap_values = explainer(df_features)

print("\n📌 Top SHAP contributions:")
shap_vals = shap_values.values[0]
for skill, val in sorted(zip(df_features.columns, shap_vals), key=lambda x: abs(x[1]), reverse=True)[:5]:
    print(f"{skill}: {val:.3f}")

# גרף SHAP
shap.plots.bar(shap_values[0], show=False)
plt.title("SHAP Feature Importance")
plt.tight_layout()
plt.savefig("shap_importance.png")
print("📊 SHAP plot saved to shap_importance.png")

# 🔍 הסבר עם LIME
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=df_features.values,
    feature_names=df_features.columns.tolist(),
    mode="regression"
)

lime_exp = lime_explainer.explain_instance(
    df_features.iloc[0].values,
    model.predict,
    num_features=5
)

print("\n📌 Top LIME contributions:")
for feat, weight in lime_exp.as_list():
    print(f"{feat}: {weight:.3f}")

lime_exp.save_to_file("lime_explanation.html")
print("📄 LIME explanation saved to lime_explanation.html")
