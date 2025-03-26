
# app.py

from flask import Flask, request, jsonify
from config import users_collection, resumes_collection, explanations_collection
from bson import ObjectId
from services.cv_matcher import match_resume_to_job, match_resume_to_job_from_text
import datetime
import json
import os

app = Flask(__name__)

# מסייע להחזיר מזהים של מונגו בפורמט קריא
class JSONEncoderWithObjectId(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, ObjectId):
            return str(o)
        return super().default(o)

app.json_encoder = JSONEncoderWithObjectId

@app.route("/")
def home():
    return {"message": "Welcome to the CV Matching API 🎯"}

@app.route("/users", methods=["POST"])
def create_user():
    data = request.json
    if not all(k in data for k in ("email", "password_hash", "role")):
        return jsonify({"error": "Missing fields"}), 400

    user = {
        "email": data["email"],
        "password_hash": data["password_hash"],
        "role": data["role"],
        "created_at": datetime.datetime.utcnow()
    }
    result = users_collection.insert_one(user)
    return jsonify({"user_id": str(result.inserted_id)})

@app.route("/resumes", methods=["POST"])
def upload_resume():
    data = request.json
    if not all(k in data for k in ("user_id", "file_name", "raw_text", "job_title")):
        return jsonify({"error": "Missing fields"}), 400

    resume = {
        "user_id": ObjectId(data["user_id"]),
        "file_name": data["file_name"],
        "upload_date": datetime.datetime.utcnow(),
        "raw_text": data["raw_text"],
        "match_score": None,
        "job_title": data["job_title"],
        "analysis_completed": False
    }
    result = resumes_collection.insert_one(resume)
    return jsonify({"resume_id": str(result.inserted_id)})

@app.route("/analyze/<resume_id>", methods=["POST"])
@app.route("/analyze/<resume_id>", methods=["POST"])
def analyze_resume(resume_id):
    data = request.json
    resume = resumes_collection.find_one({"_id": ObjectId(resume_id)})

    if not resume:
        return jsonify({"error": "Resume not found"}), 404

    # משתמשים ישירות בטקסט במקום לקרוא קובץ PDF
    raw_text = resume["raw_text"]
    score, cleaned_text, tech_count = match_resume_to_job_from_text(raw_text, data["job_description_text"])

    resumes_collection.update_one(
        {"_id": ObjectId(resume_id)},
        {"$set": {
            "score": score,
            "clean_text": cleaned_text,
            "tech_count": tech_count
        }}
    )
    return jsonify({
        "score": score,
        "tech_count": tech_count
    })


    # שמירת תוכן קובץ CV זמנית
    file_path = f"temp_cv_{resume_id}.txt"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(resume["raw_text"])

    score, clean_text, tech_count = match_resume_to_job(file_path, data["job_description_text"])
    os.remove(file_path)

    resumes_collection.update_one(
        {"_id": ObjectId(resume_id)},
        {"$set": {
            "match_score": score,
            "analysis_completed": True
        }}
    )

    return jsonify({
        "resume_id": resume_id,
        "match_score": score,
        "tech_count": tech_count
    })

if __name__ == "__main__":
    app.run(debug=True)
