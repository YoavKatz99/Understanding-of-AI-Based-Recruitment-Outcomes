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

    html_content = exp.as_html()

    # ✅ Fix spacing by replacing white-space
    html_content = html_content.replace("white-space: pre-wrap;", "white-space: normal;")

    # ✅ Add iframe resizing support
    resize_script = """
    <script>
    window.onload = function () {
        setTimeout(function () {
        const height = document.body.scrollHeight;
        console.log("📏 iframe sending height:", height);
        window.parent.postMessage({ height: height }, "*");
        }, 500);
    };
    </script>
    <style>
    body { margin: 0; padding: 20px; overflow: hidden; }
    .lime_text_div { max-height: none !important; overflow: visible !important; }
    </style>
    """


    # ✅ Inject before </body>
    if "</body>" in html_content:
        html_content = html_content.replace("</body>", resize_script + "</body>")
    else:
        html_content += resize_script

    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "lime_text_explanation.html")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    prediction_score = predict_fn([text])[0][1] * 100

    return {
        "prediction": round(float(prediction_score), 2),
        "output_file": "lime_text_explanation.html"
    }
