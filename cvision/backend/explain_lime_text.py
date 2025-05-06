from lime.lime_text import LimeTextExplainer
from pdfminer.high_level import extract_text
import os
import numpy as np

def run_text_lime_with_xgb(filepath, model, vectorizer):
    print("\n📄 Extracting text...")
    text = extract_text(filepath)

    def predict_fn(texts):
        X = vectorizer.transform(texts)
        preds = model.predict(X)
        preds = np.clip(preds, 0, 100)
        preds = preds / 100
        return np.vstack([1 - preds, preds]).T

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

    prediction_score = predict_fn([text])[0][1] * 100

    return {
        "prediction": round(float(prediction_score), 2),
        "output_file": "lime_text_explanation.html"
    }
