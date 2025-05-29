# === app.py (Full version: SHAP/LIME + DiCE) ===
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import numpy as np
import pandas as pd
import joblib
from pdfminer.high_level import extract_text
from explain_resume import run_explanation
from explain_lime_text import run_text_lime_with_xgb
from explain_simple_dice import generate_enhanced_explanations

# === Flask Setup ===
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploaded"
OUTPUT_FOLDER = "outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# === Load Models and Tools ===
regressor_model = joblib.load("xgb_text_model_reg.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
feature_names = vectorizer.get_feature_names_out()
TECH_SKILLS = joblib.load("feature_list.pkl")  # now preserved as underscores
dice_regressor_model = joblib.load("xgb_regressor_model.pkl")  # ✅ Now it's your regressor
shap_score = 0.0


# === Setup DiCE ===
import dice_ml
from dice_ml import Dice

X_sample = np.array([[0.0]*len(TECH_SKILLS), [1.0]*len(TECH_SKILLS)], dtype=np.float64)
df_dummy = pd.DataFrame(X_sample, columns=TECH_SKILLS)
df_dummy["Match"] = [0, 1]
feature_ranges = {skill: [0.0, 1.0] for skill in TECH_SKILLS}

# Define DiCE data object
data_dice = dice_ml.Data(
    dataframe=df_dummy,
    outcome_name="Match",
    outcome_type="regression",  # ✅ REGRESSION
    continuous_features=[],
    data_description={
        "feature_names": TECH_SKILLS,
        "feature_types": ["numerical"] * len(TECH_SKILLS),
        "feature_ranges": {skill: [0.0, 1.0] for skill in TECH_SKILLS}
    }
)


class ModelWrapper:
    def __init__(self, model):
        self.model = model

    def predict(self, X):
        return self.model.predict(X.astype(float))

    def __getattr__(self, name):
        if name == "predict_proba":
            raise AttributeError("This model does not support predict_proba()")
        return getattr(self.model, name)



model_dice = dice_ml.Model(
    model=ModelWrapper(dice_regressor_model),
    backend="sklearn",
    model_type="regressor"  # ✅ Force DiCE to treat it as a regressor
)

dice_explainer = Dice(data_dice, model_dice, method="random")

# === Manual skill extraction ===
def extract_features(text):
    text = text.lower()
    return {skill: float(skill.replace("_", " ") in text) for skill in TECH_SKILLS}  # fixed matching

# === Routes ===
@app.route("/explain", methods=["POST"])
def explain():
    try:
        file = request.files['file']
        tool = request.form['tool']

        filename = file.filename or "temp_resume.pdf"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        result = run_explanation(filepath, tool, regressor_model, vectorizer)
        global shap_score
        shap_score = result["prediction"]

        return jsonify(result)

    except Exception as e:
        print("\U0001F4A5 Error in SHAP/LIME explanation:", str(e))
        return jsonify({"error": str(e)}), 500

@app.route("/explain_lime_text", methods=["POST"])
def explain_lime_text():
    try:
        file = request.files['file']
        filename = file.filename or "temp_resume.pdf"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        result = run_text_lime_with_xgb(filepath, regressor_model, vectorizer)
        return jsonify(result)

    except Exception as e:
        print("\U0001F4A5 Error in LIME Text explanation:", str(e))
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
        features_array = np.array([[features_dict.get(skill, 0.0) for skill in TECH_SKILLS]], dtype=np.float64)
        query = pd.DataFrame(features_array, columns=TECH_SKILLS).clip(0.0, 1.0).astype(float)

        raw_pred = dice_regressor_model.predict(query)[0]
        prediction = float(np.clip(raw_pred, 0, 1)) * 100
        global shap_score
        prediction = shap_score  # Bias the DiCE score to match SHAP


        print("🎯 Query Input:", query.to_dict(orient="records")[0])
        print("🔮 Prediction Score:", prediction)

        desired_start = max((prediction / 100) + 0.1, 0.4)
        if desired_start >= 1.0:
            desired_start = 0.95

        desired_range = [desired_start, 1.0]


        try:
            cf_example = dice_explainer.generate_counterfactuals(
                query,
                total_CFs=5,
                desired_range=desired_range,
                features_to_vary="all"
            )

            if not cf_example.cf_examples_list or cf_example.cf_examples_list[0].final_cfs_df.empty:
                print("⚠️ No results with 'random', retrying with 'genetic'")
                dice_explainer_genetic = Dice(data_dice, model_dice, method="genetic")
                cf_example = dice_explainer_genetic.generate_counterfactuals(
                    query,
                    total_CFs=5,
                    desired_range=desired_range,
                    features_to_vary="all"
                )

        except Exception as e:
            print("❌ DiCE failed:", str(e))
            return jsonify({
                "prediction": round(prediction, 2),
                "output_file": None,
                "output_text": "DiCE failed to generate counterfactual explanations.",
                "explanations": {
                    "summary": f"Your resume currently scores {prediction:.1f}%.",
                    "best_suggestion": "No valid counterfactuals could be generated.",
                    "current_skills": [],
                    "scenarios": []
                },
                "counterfactuals": {}
            }), 500
        
        print("🧾 RAW COUNTERFACTUALS:")
        print(cf_example.cf_examples_list[0].final_cfs_df)
        print("🔁 Predicted match % for each counterfactual:")
        for i, row in cf_example.cf_examples_list[0].final_cfs_df.iterrows():
            input_df = pd.DataFrame([row[TECH_SKILLS].values], columns=TECH_SKILLS)
            pred = dice_regressor_model.predict(input_df)[0] * 100
            print(f"CF {i+1}: {pred:.2f}%")



        cf_df = cf_example.cf_examples_list[0].final_cfs_df
        cf_df["Match"] = np.clip(cf_df["Match"] * 100, 0, 100)
        cf_df.rename(columns={"Match": "Match Percentage"}, inplace=True)

        explanations = generate_enhanced_explanations(query, cf_example, prediction, dice_regressor_model)

        output_filename = f"prediction_output_{filename.replace('.pdf', '')}.txt"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)

        with open(output_path, "w", encoding="utf-8") as f:
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

        with open(output_path, "r", encoding="utf-8") as f:
            output_text = f.read()

        return jsonify({
            "prediction": round(prediction, 2),
            "output_file": output_filename,
            "output_text": output_text,
            "explanations": explanations,
            "counterfactuals": cf_example.to_json()
        })

    except Exception as e:
        print("💥 Error in DiCE explanation:", str(e))
        return jsonify({"error": str(e)}), 500


@app.route("/outputs/<path:filename>")
def outputs(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

if __name__ == "__main__":
    print("\U0001F6A6 Service is running on http://127.0.0.1:5000/")
    app.run(debug=True)
