import os
import shap
import numpy as np
import pandas as pd
import spacy
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend suitable for servers
import matplotlib.pyplot as plt

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

        features_dense = features.toarray()

        # ✅ Explicitly use TreeExplainer for XGBoost
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(features_dense)

        feature_names = vectorizer.get_feature_names_out()

        print("Mean SHAP value (non-abs):", np.mean(shap_values))
        print("Max SHAP value:", np.max(shap_values))


        explanation = shap.Explanation(
            values=shap_values,
            base_values=explainer.expected_value,
            data=features_dense,
            feature_names=feature_names
        )

        shap_plot_path = os.path.join(OUTPUT_DIR, "shap_importance.png")
        shap.plots.bar(explanation, show=False)

        # Adjust figure to make room for long feature names
        fig = plt.gcf()
        fig.set_size_inches(10, 6)         # Wider figure
        plt.subplots_adjust(left=0.3)      # More space on the left for labels

        plt.savefig(shap_plot_path, bbox_inches="tight")  # ensures everything fits
        plt.clf()

        print(f"📊 SHAP global bar plot saved to {shap_plot_path}")
        # Extract top 5 impactful features
        top_indices = np.argsort(np.abs(shap_values[0]))[::-1][:5]
        top_features = [(feature_names[i], shap_values[0][i]) for i in top_indices]

        explanation_text = "Top SHAP features:\n" + "\n".join(
            [f"{name}: {value:.4f}" for name, value in top_features]
        )



    return {
        "prediction": round(float(prediction), 2),
        "explanation": explanation_text
    }
