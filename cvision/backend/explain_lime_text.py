# explain_lime_text.py
from lime.lime_text import LimeTextExplainer
from pdfminer.high_level import extract_text
import joblib
import os
import numpy as np  # ✅ Make sure to import numpy

def run_text_lime_with_xgb(filepath):
    print("\n📄 Extracting text...")
    text = extract_text(filepath)

    print("\n📦 Loading model and vectorizer...")
    model = joblib.load("xgb_text_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")

    def predict_fn(texts):
        X = vectorizer.transform(texts)
        preds = model.predict(X)  # preds is shape (n_samples,), values like 0-100
        preds = np.clip(preds, 0, 100)  # Safety: clip to 0-100
        preds = preds / 100  # Normalize to 0-1
        return np.vstack([1 - preds, preds]).T  # Shape (n_samples, 2)

    class_names = ["Not a Match", "Match"]
    explainer = LimeTextExplainer(class_names=class_names)

    exp = explainer.explain_instance(
        text_instance=text,
        classifier_fn=predict_fn,
        num_features=10
    )

    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "lime_text_explanation.html")
    exp.save_to_file(output_path)

    prediction_score = predict_fn([text])[0][1] * 100  # Get Match % (second column)

    return {
        "prediction": round(float(prediction_score), 2),
        "output_file": "lime_text_explanation.html"
    }
