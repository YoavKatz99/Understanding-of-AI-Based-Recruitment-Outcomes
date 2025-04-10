from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os

# לצורך הדגמה בלבד (החליפי בקוד האמיתי של CARLA בהמשך)
import matplotlib.pyplot as plt

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploaded"
OUTPUT_FOLDER = "outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/explain_carla', methods=['POST'])
def explain_carla():
    try:
        # קבלת קובץ
        file = request.files['file']
        filename = file.filename or "temp_resume.pdf"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        print(f"📄 קובץ נשמר: {filepath}")

        # כאן את מריצה את CARLA בפועל (כרגע הדמיה עם גרף)
        print(f"🚀 מתחילה ניתוח CARLA לקובץ: {filename}")
        
        output_filename = f"carla_output_{filename.replace('.pdf', '')}.png"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)

        # יצירת תוצאה לדוגמה
        plt.figure()
        plt.text(0.5, 0.5, f"CARLA result for {filename}", fontsize=14, ha='center')
        plt.axis('off')
        plt.savefig(output_path)
        plt.close()

        print(f"✅ תמונה נשמרה ב: {output_path}")

        # החזרת תגובה
        return jsonify({
            "prediction": 85.0,
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
