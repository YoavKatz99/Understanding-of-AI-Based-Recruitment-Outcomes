from flask import Flask, request, jsonify
import joblib
import pandas as pd
import spacy
import os
from pdfminer.high_level import extract_text

app = Flask(__name__)

# Load trained model
model_path = "random_forest_model.pkl"
rf_model = joblib.load(model_path)

# Load Spacy model and define skills
nlp = spacy.load("en_core_web_sm")
stopwords = nlp.Defaults.stop_words
TECH_SKILLS = [
    'data science', 'computer vision', 'natural language processing', 'ai', 'ml',
    'machine learning', 'deep learning', 'logistic regression', 'classification',
    'scikit learn', 'numpy', 'pandas', 'matplotlib', 'seaborn', 'tensorflow',
    'keras', 'pytorch', 'cnn', 'rnn', 'nlp', 'opencv', 'django', 'mongodb', 'sql'
]

# Feature extraction
def process_text(text):
    doc = nlp(text.lower())
    return " ".join([token.lemma_ for token in doc if token.text not in stopwords])

def extract_features_from_text(text):
    text = process_text(text)
    features = {skill: 1 if skill in text else 0 for skill in TECH_SKILLS}
    return pd.DataFrame([features])

@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files["file"]

        # Save the file temporarily
        temp_path = "temp_resume.pdf"
        file.save(temp_path)

        # Extract text from PDF
        pdf_text = extract_text(temp_path)

        # Delete temp file
        os.remove(temp_path)

        # Predict
        features_df = extract_features_from_text(pdf_text)
        prediction = rf_model.predict(features_df)[0]

        return jsonify({"match_score": round(prediction, 2)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)