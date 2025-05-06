import os
import shap
import numpy as np
import pandas as pd
import spacy
from pdfminer.high_level import extract_text

OUTPUT_DIR = "outputs"

nlp = spacy.load("en_core_web_sm")

def run_explanation(filepath, tool, model, vectorizer):
    print("\n📄 Extracting text from PDF...")
    text = extract_text(filepath)
    text_cleaned = text.lower()
    features = vectorizer.transform([text_cleaned])

    prediction = model.predict(features)[0]
    print(f"\n🔎 Match Score Prediction: {prediction:.2f}%")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if tool == "shap":
        print("\n📌 Explaining with SHAP...")

        explainer = shap.Explainer(model)
        shap_values = explainer(features)

        feature_names = vectorizer.get_feature_names_out()

        explanation = shap.Explanation(
            values=shap_values.values[0],
            base_values=shap_values.base_values[0],
            data=features.toarray()[0],
            feature_names=feature_names
        )

        shap_plot_path = os.path.join(OUTPUT_DIR, "shap_importance.png")

        shap.plots.bar(explanation, show=False)

        print(f"📊 SHAP bar plot saved to {shap_plot_path}")

    return {"prediction": round(float(prediction), 2)}
