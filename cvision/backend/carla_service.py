from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import matplotlib.pyplot as plt

# CARLA
from carla.models.catalog import MLModelCatalog
from carla.recourse_methods import GrowingSpheres
from carla.data.catalog import OnlineCatalog

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploaded"
OUTPUT_FOLDER = "outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/explain_carla', methods=['POST'])
def explain_carla():
    try:
        file = request.files['file']
        filename = file.filename or "temp_resume.pdf"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        print(f"📄 קובץ נשמר: {filepath}")

        # ניתוח תרחיש נגד-עובדתי עם CARLA
        print(f"🚀 התחלת ניתוח CARLA לדוגמה (נתוני adult)...")

        data = OnlineCatalog("adult")
        model = MLModelCatalog(data, model_type="ann")

        # שורת קלט לדוגמה - תשתנה אחר כך למידע מקובץ קורות חיים
        query = data.raw.iloc[[0]].copy()
        query["age"] = 30
        query["education"] = "Bachelors"

        gs = GrowingSpheres(model, data)
        cf = gs.get_counterfactuals(query)

        output_filename = f"carla_output_{filename.replace('.pdf', '')}.png"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)

        # שמירת התוצאה כתמונה
        cf.visualize(path=output_path)

        print(f"✅ תמונת תרחיש נשמרה: {output_path}")

        return jsonify({
            "prediction": 85.0,  # ציון פיקטיבי בינתיים
            "output_file": output_filename
        })

    except Exception as e:
        print("💥 שגיאה:", str(e))
        return jsonify({"error": str(e)}), 500

@app.route("/outputs/<path:filename>")
def outputs(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

if __name__ == "__main__":
    print("🚦 CARLA Service is starting on http://127.0.0.1:5001/")
    app.run(port=5001, debug=True)
