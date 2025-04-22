import joblib
import shap
import lime
import lime.lime_tabular
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import spacy
from pdfminer.high_level import extract_text
import os

# === הגדרות ===
MODEL_PATH = "xgboost_model.pkl"
FEATURE_NAMES_PATH = "feature_names.txt"
TECH_SKILLS = [
    'data_science', 'computer_vision', 'natural_language_processing', 'ai', 'ml',
    'machine_learning', 'deep_learning', 'logistic_regression', 'classification',
    'scikit_learn', 'numpy', 'pandas', 'matplotlib', 'seaborn', 'tensorflow',
    'keras', 'pytorch', 'cnn', 'rnn', 'nlp', 'opencv', 'django', 'mongodb', 'sql'
]

# טען את מודל ה-Spacy
nlp = spacy.load("en_core_web_sm")

def process_text(text):
    doc = nlp(text.lower())
    return " ".join([token.lemma_ for token in doc])

def extract_features_from_text(text):
    text = process_text(text)
    features = {skill: int(skill.replace("_", " ") in text) for skill in TECH_SKILLS}
    print("\n📊 Extracted Features:")
    print(features)
    return pd.DataFrame([features])

def run_explanation(filepath, tool):
    print("\n📄 Extracting text from PDF...")
    text = extract_text(filepath)
    print("\n📄 Extracted Resume Text:")
    print(text)

    print("\n📦 Loading model...")
    model = joblib.load(MODEL_PATH)

    features_df = extract_features_from_text(text)

    # ודא שהסדר של הפיצ'רים תואם לאימון
    with open(FEATURE_NAMES_PATH) as f:
        feature_order = [line.strip() for line in f]

    for feat in feature_order:
        if feat not in features_df.columns:
            features_df[feat] = 0
    features_df = features_df[feature_order]

    prediction = model.predict(features_df)[0]
    print(f"\n🔎 Match Score Prediction: {prediction}")

    # ודא שתיקיית התוצרים קיימת
    OUTPUT_DIR = "outputs"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if tool == "shap":
        print("\n📌 Top SHAP contributions:")
        explainer = shap.Explainer(model)
        shap_values = explainer(features_df)
        shap_df = pd.DataFrame({
            "feature": features_df.columns,
            "shap_value": shap_values.values[0]
        }).sort_values(by="shap_value", ascending=False)
        print(shap_df.head())

        # יצירת גרף SHAP ושמירתו
        shap.plots.bar(shap_values[0], show=False)
        plt.tight_layout()
        shap_plot_path = os.path.join(OUTPUT_DIR, "shap_importance.png")
        plt.savefig(shap_plot_path)
        plt.close()
        print(f"📊 SHAP plot saved to {shap_plot_path}")

    """elif tool == "lime":
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

    lime_plot_path = os.path.join(OUTPUT_DIR, "lime_explanation.html")
    lime_exp.save_to_file(lime_plot_path)

    # Extract feature importances (top skills)
    important_words = lime_exp.as_list()
    highlighted_words = [
        {"word": w.replace(" ", "_"), "weight": float(weight)}
        for w, weight in important_words
    ]

    print(f"📄 LIME explanation saved to {lime_plot_path}")
    
    return {
        "prediction": round(float(prediction), 2),
        "highlighted_words": highlighted_words,
        "output_file": "lime_explanation.html"
    }
    """

    # מחזירים מילון עם תוצאת החיזוי
    return {"prediction": round(float(prediction), 2)}
    
if __name__ == "__main__":
    # להרצה לבד לצורך בדיקה
    result = run_explanation("test_resume.pdf", "shap")
    print(result)
