# app.py
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import numpy as np
import pandas as pd
import joblib
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

# === Load Model and Vectorizer ===
model = joblib.load("xgb_text_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
feature_names = vectorizer.get_feature_names_out()

# === Setup DiCE with Genetic Algorithm ===
n_features = len(feature_names)

# Create dummy dataframe for DiCE setup
X_train_dummy = np.random.rand(100, n_features)
y_train_dummy = (np.random.rand(100) > 0.5).astype(int)

df_dummy = pd.DataFrame(X_train_dummy, columns=feature_names)
df_dummy["Match"] = y_train_dummy

data_dice = dice_ml.Data(
    dataframe=df_dummy,
    continuous_features=feature_names.tolist(),
    outcome_name="Match"
)

class ModelWrapper:
    def __init__(self, model, threshold=50):
        self.model = model
        self.threshold = threshold

    def predict(self, X):
        preds = self.model.predict(X)
        return (preds >= self.threshold).astype(int)

    def predict_proba(self, X):
        preds = self.model.predict(X)
        preds = np.clip(preds, 0, 100) / 100
        preds_class0 = 1 - preds
        preds_class1 = preds
        return np.stack([preds_class0, preds_class1], axis=1)


model_dice = dice_ml.Model(model=ModelWrapper(model, threshold=50), backend="sklearn")
exp = Dice(
    data_dice,
    model_dice,
    method="genetic"
)

# === Routes ===

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
        print("💥 Error in SHAP/LIME explanation:", str(e))
        return jsonify({"error": str(e)}), 500

@app.route("/explain_carla", methods=["POST"])
def explain_dice():
    try:
        file = request.files['file']
        filename = file.filename or "temp_resume.pdf"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # === Extract text
        text = extract_text(filepath)
        text_cleaned = text.lower()
        features = vectorizer.transform([text_cleaned])
        features_array = features.toarray()

        prediction = float(model.predict(features)[0])

        query_df = pd.DataFrame(features_array, columns=feature_names)

        # === Fix: create fresh ModelWrapper inside the function
        class ModelWrapper:
            def __init__(self, model, threshold=50):
                self.model = model
                self.threshold = threshold

            def predict(self, X):
                preds = self.model.predict(X)
                return (preds >= self.threshold).astype(int)

            def predict_proba(self, X):
                preds = self.model.predict(X)
                preds = np.clip(preds, 0, 100) / 100
                preds_class0 = 1 - preds
                preds_class1 = preds
                return np.stack([preds_class0, preds_class1], axis=1)

        model_dice = dice_ml.Model(model=ModelWrapper(model), backend="sklearn")

        # Setup DiCE (inside route)
        n_features = len(feature_names)
        X_train_dummy = np.random.rand(100, n_features)
        y_train_dummy = (np.random.rand(100) > 0.5).astype(int)

        df_dummy = pd.DataFrame(X_train_dummy, columns=feature_names)
        df_dummy["Match"] = y_train_dummy

        data_dice = dice_ml.Data(
            dataframe=df_dummy,
            continuous_features=feature_names.tolist(),
            outcome_name="Match"
        )

        exp = Dice(data_dice, model_dice, method="genetic")

        # === Pick important features
        nonzero_indices = np.where(features_array[0] > 0.01)[0]
        important_features = feature_names[nonzero_indices]
        important_features = important_features.tolist()[:30]

        # === Try generating counterfactuals safely
        try:
            cf_examples = exp.generate_counterfactuals(
                query_df,
                total_CFs=2,                  # Only try generating 2 counterfactuals
                desired_class="opposite",
                features_to_vary=important_features[:10],   # Use only top 10 important features
                verbose=False                 # Hide ugly progress bar
            )

        except Exception as dice_error:
            print("⚡ DiCE failed to generate counterfactuals:", str(dice_error))
            return jsonify({
                "prediction": round(prediction, 2),
                "scenarios": [],
                "error": "Could not generate counterfactuals. Try modifying the resume."
            })

        current_features = features_array[0]

        # === Parse counterfactuals
        scenarios = []
        for idx, cf in enumerate(cf_examples.cf_examples_list[0].final_cfs_df.iterrows()):
            cf_vector = cf[1].values
            added_words = []
            for i, (old_val, new_val) in enumerate(zip(current_features, cf_vector)):
                if old_val == 0 and new_val > 0.05:
                    word = feature_names[i]
                    added_words.append(word)

            if added_words:
                new_score = model.predict([cf_vector])[0]
                improvement = new_score - prediction
                scenarios.append({
                    "scenario": idx + 1,
                    "added_words": added_words,
                    "new_score": round(float(new_score), 2),
                    "improvement": round(float(improvement), 2)
                })

        # Sort scenarios
        scenarios = sorted(scenarios, key=lambda x: (len(x['added_words']), -x['improvement']))

        return jsonify({
            "prediction": round(prediction, 2),
            "scenarios": scenarios
        })

    except Exception as e:
        print("💥 Error in DiCE explanation route:", str(e))
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
