from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import numpy as np
import pandas as pd
import joblib
import spacy
from pdfminer.high_level import extract_text
import dice_ml
from dice_ml import Dice
from explain_resume import run_explanation
from explain_lime_text import run_text_lime_with_xgb


app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploaded"
OUTPUT_FOLDER = "outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# === Model and NLP Setup ===
model_path = "xgboost_model.pkl"
feature_names_path = "feature_names.txt"
model = joblib.load(model_path)
nlp = spacy.load("en_core_web_sm")

TECH_SKILLS = [
    'data_science', 'computer_vision', 'natural_language_processing', 'ai', 'ml',
    'machine_learning', 'deep_learning', 'logistic_regression', 'classification',
    'scikit_learn', 'numpy', 'pandas', 'matplotlib', 'seaborn', 'tensorflow',
    'keras', 'pytorch', 'cnn', 'rnn', 'nlp', 'opencv', 'django', 'mongodb', 'sql'
]

SKILL_DESCRIPTIONS = {
    'data_science': 'expertise in analyzing and interpreting complex data',
    'computer_vision': 'ability to process and analyze visual information using computers',
    'natural_language_processing': 'building systems that understand and process human language',
    'ai': 'building artificial intelligence systems',
    'ml': 'implementing machine learning algorithms',
    'machine_learning': 'creating systems that learn from data',
    'deep_learning': 'working with neural networks and deep learning architectures',
    'logistic_regression': 'experience with logistic regression models',
    'classification': 'classifying data into categories using algorithms',
    'scikit_learn': 'using the scikit-learn machine learning library',
    'numpy': 'working with numerical computations in Python',
    'pandas': 'data manipulation and analysis using pandas',
    'matplotlib': 'data visualization using matplotlib',
    'seaborn': 'statistical data visualization',
    'tensorflow': 'building and training ML models with TensorFlow',
    'keras': 'building neural networks with Keras',
    'pytorch': 'developing deep learning models with PyTorch',
    'cnn': 'implementing convolutional neural networks',
    'rnn': 'working with recurrent neural networks',
    'nlp': 'processing and analyzing natural language',
    'opencv': 'computer vision library for image processing',
    'django': 'web development with Django framework',
    'mongodb': 'working with MongoDB database',
    'sql': 'structured query language for database management'
}

feature_names = [line.strip() for line in open(feature_names_path)]

def process_text(text):
    doc = nlp(text.lower())
    return " ".join([token.lemma_ for token in doc])

def extract_features(text):
    text = process_text(text)
    return {skill: float(skill.replace("_", " ") in text) for skill in TECH_SKILLS}

# === DiCE Setup ===
X_train = np.ones((2, len(TECH_SKILLS)), dtype=np.float32)
X_train[1] = np.zeros(len(TECH_SKILLS), dtype=np.float32)
y_train = np.array([1, 0], dtype=np.int32)

df_dict = {skill: X_train[:, i] for i, skill in enumerate(TECH_SKILLS)}
df_dict["Match"] = y_train
sample_df = pd.DataFrame(df_dict).astype(float)

feature_ranges = {feat: [0.0, 1.0] for feat in TECH_SKILLS}

data_dice = dice_ml.Data(
    dataframe=sample_df,
    continuous_features=[],
    outcome_name="Match",
    outcome_type="classification",
    data_description={
        "feature_names": TECH_SKILLS,
        "feature_types": ["numerical"] * len(TECH_SKILLS),
        "feature_ranges": feature_ranges
    }
)

class ModelWrapper:
    def __init__(self, model, threshold=75):
        self.model = model
        self.threshold = threshold

    def predict(self, X):
        preds = self.model.predict(X.astype(float))
        return (preds >= self.threshold).astype(int)

    def predict_proba(self, X):
        probs = self.model.predict(X.astype(float))
        return np.stack([1 - probs / 100, probs / 100], axis=1)

model_dice = dice_ml.Model(model=ModelWrapper(model), backend="sklearn")
exp = Dice(data_dice, model_dice)

def generate_enhanced_explanations(query, cf_examples, prediction):
    current_skills = [skill for skill in TECH_SKILLS if query[skill].iloc[0] > 0.5]
    current_skills_formatted = [skill.replace("_", " ") for skill in current_skills]
    cf_df = cf_examples.cf_examples_list[0].final_cfs_df
    scenarios = []

    for idx, cf_row in cf_df.iterrows():
        cf_features = cf_row[TECH_SKILLS].values.reshape(1, -1)
        cf_score = float(model.predict(cf_features)[0])

        added_skills = []
        for skill in TECH_SKILLS:
            if float(cf_row[skill]) > 0.5 and float(query[skill].iloc[0]) < 0.5:
                added_skills.append(skill)

        if not added_skills:
            continue

        added_skills_info = []
        for skill in added_skills:
            added_skills_info.append({
                "name": skill.replace("_", " "),
                "description": SKILL_DESCRIPTIONS.get(skill, "")
            })

        score_improvement = cf_score - prediction

        scenarios.append({
            "added_skills": added_skills_info,
            "new_score": cf_score,
            "improvement": score_improvement
        })

    scenarios.sort(key=lambda x: x["improvement"], reverse=True)

    if scenarios:
        top = scenarios[0]
        summary = f"Your resume currently scores {prediction:.1f}%. "
        summary += f"Best improvement: add {', '.join([s['name'] for s in top['added_skills']])} to reach {top['new_score']:.1f}%"
        best_suggestion = f"Potential gain: +{top['improvement']:.1f}%"
    else:
        summary = f"Your resume currently scores {prediction:.1f}% match."
        best_suggestion = "No significant improvements found."

    return {
        "summary": summary,
        "best_suggestion": best_suggestion,
        "current_skills": current_skills_formatted,
        "scenarios": scenarios
    }

@app.route("/explain", methods=["POST"])
def explain():
    try:
        file = request.files['file']
        tool = request.form['tool']

        filename = file.filename or "temp_resume.pdf"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        result = run_explanation(filepath, tool)
        return jsonify(result)
    except Exception as e:
        print("💥 Error:", str(e))
        return jsonify({"error": str(e)}), 500

@app.route("/explain_carla", methods=["POST"])
def explain_dice():
    try:
        file = request.files['file']
        filename = file.filename or "temp_resume.pdf"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        text = extract_text(filepath)
        features_dict = extract_features(text)
        features_array = np.array([[features_dict.get(skill, 0.0) for skill in TECH_SKILLS]], dtype=np.float32)
        query = pd.DataFrame(features_array, columns=TECH_SKILLS).astype(float)

        prediction = float(model.predict(query)[0])
        cf_example = exp.generate_counterfactuals(query, total_CFs=5, desired_class=1)
        explanations = generate_enhanced_explanations(query, cf_example, prediction)

        output_filename = f"prediction_output_{filename.replace('.pdf', '')}.txt"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)

        with open(output_path, "w") as f:
            f.write(f"Predicted Match Percentage: {prediction:.1f}%\n\n")
            f.write("COUNTERFACTUAL EXPLANATIONS\n===========================\n\n")
            f.write(f"{explanations['summary']}\n\n")
            f.write(f"{explanations['best_suggestion']}\n\n")
            f.write("Your Current Skills:\n")
            if explanations['current_skills']:
                for skill in explanations['current_skills']:
                    f.write(f"- {skill}\n")
            else:
                f.write("- No recognized skills detected\n")

            f.write("\nImprovement Scenarios:\n")
            for i, scenario in enumerate(explanations['scenarios']):
                f.write(f"\nScenario {i+1}: Score improvement to {scenario['new_score']:.1f}%\n")
                f.write("Add these skills:\n")
                for skill in scenario['added_skills']:
                    f.write(f"- {skill['name']}: {skill['description']}\n")

        with open(output_path, "r") as f:
            output_text = f.read()

        return jsonify({
            "prediction": prediction,
            "output_file": output_filename,
            "output_text": output_text,
            "explanations": explanations,
            "counterfactuals": cf_example.to_json()
        })

    except Exception as e:
        print("💥 Error in DiCE explanation:", str(e))
        return jsonify({"error": str(e)}), 500

@app.route("/explain_lime_text", methods=["POST"])
def explain_lime_text():
    try:
        file = request.files['file']
        filename = file.filename or "temp_resume.pdf"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        result = run_text_lime_with_xgb(filepath)
        return jsonify(result)

    except Exception as e:
        print("💥 Error in LIME Text explanation:", str(e))
        return jsonify({"error": str(e)}), 500


@app.route("/outputs/<path:filename>")
def outputs(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

if __name__ == "__main__":
    print("🚦 Service is running on http://127.0.0.1:5000/")
    app.run(debug=True)
