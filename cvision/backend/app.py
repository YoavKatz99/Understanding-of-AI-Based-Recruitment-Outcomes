from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from explain_resume import run_explanation

app = Flask(__name__)
CORS(app)

@app.route('/explain', methods=['POST'])
def explain():
    try:
        file = request.files['file']
        tool = request.form['tool']

        # שמירה זמנית של הקובץ
        filepath = "temp_resume.pdf"
        file.save(filepath)

        result = run_explanation( tool,filepath)

        return jsonify(result)  # ← מחזיר את המילון כמו שהוא
    except Exception as e:
        print("💥 Error:", str(e))
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
