import os
import joblib
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import spacy
from pdfminer.high_level import extract_text


# === Settings ===
MODEL_PATH = "xgb_text_model.pkl"        # New text-based model
VECTORIZER_PATH = "tfidf_vectorizer.pkl" # TF-IDF vectorizer
OUTPUT_DIR = "outputs"

# Load Spacy for optional text processing (if needed later)
nlp = spacy.load("en_core_web_sm")

# === Functions ===

def run_explanation(filepath, tool):
    print("\n📄 Extracting text from PDF...")
    text = extract_text(filepath)
    print("\n📄 Extracted Resume Text:")
    print(text)

    print("\n📦 Loading model and vectorizer...")
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)

    # Preprocess text (basic lowercase cleaning)
    text_cleaned = text.lower()
    features = vectorizer.transform([text_cleaned])  # Shape (1, n_features)

    # Predict match score
    prediction = model.predict(features)[0]
    print(f"\n🔎 Match Score Prediction: {prediction:.2f}%")

    # Make sure output folder exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if tool == "shap":
        print("\n📌 Explaining with SHAP...")

        predict_fn = lambda X: model.predict(X)

        # Vectorize the text
        features = vectorizer.transform([text_cleaned])

        # Create SHAP explainer on model+vectorizer input
        explainer = shap.Explainer(model)

        # Get SHAP values on sparse TF-IDF input
        shap_values = explainer(features)

        feature_names = vectorizer.get_feature_names_out()

        # Build a SHAP Explanation object manually
        explanation = shap.Explanation(
            values=shap_values.values[0],  # SHAP values for 1 resume
            base_values=shap_values.base_values[0],  # expected value
            data=features.toarray()[0],  # original TF-IDF feature vector
            feature_names=feature_names
        )

        # Plot using SHAP tool
        shap_plot_path = os.path.join(OUTPUT_DIR, "shap_importance.png")

        shap.plots.bar(explanation, show=False)
        plt.tight_layout()
        plt.savefig(shap_plot_path)
        plt.close()

        print(f"📊 SHAP bar plot saved to {shap_plot_path}")





    return {"prediction": round(float(prediction), 2)}

# === Standalone Testing ===
if __name__ == "__main__":
    result = run_explanation("test_resume.pdf", "shap")
    print("\n✅ Final Result:", result)
