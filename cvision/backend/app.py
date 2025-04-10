from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from explain_resume import run_explanation

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploaded"
OUTPUT_FOLDER = "outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/explain', methods=['POST'])
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

@app.route("/outputs/<path:filename>")
def outputs(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True)
