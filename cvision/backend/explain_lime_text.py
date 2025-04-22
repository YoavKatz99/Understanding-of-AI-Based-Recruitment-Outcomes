# explain_lime_text.py
from lime.lime_text import LimeTextExplainer
from pdfminer.high_level import extract_text
import joblib
import os

def run_text_lime_with_xgb(filepath):
    print("\n📄 Extracting text...")
    text = extract_text(filepath)

    print("\n📦 Loading model and vectorizer...")
    model = joblib.load("xgb_text_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")

    def predict_proba(texts):
        X = vectorizer.transform(texts)
        return model.predict_proba(X)

    class_names = ["Not a Match", "Match"]
    explainer = LimeTextExplainer(class_names=class_names)

    exp = explainer.explain_instance(
        text_instance=text,
        classifier_fn=predict_proba,
        num_features=10
    )

    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "lime_text_explanation.html")
    exp.save_to_file(output_path)

    return {
        "prediction": round(float(predict_proba([text])[0][1]) * 100, 2),
        "output_file": "lime_text_explanation.html"
    }
