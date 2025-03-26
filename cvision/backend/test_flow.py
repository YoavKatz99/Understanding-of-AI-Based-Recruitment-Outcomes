
# test_flow.py

import requests

BASE_URL = "http://127.0.0.1:5000"

def safe_post(endpoint, json_data):
    try:
        res = requests.post(f"{BASE_URL}{endpoint}", json=json_data)
        print(f"📡 POST {endpoint} - Status: {res.status_code}")
        print("🧾 Response:", res.text)
        res.raise_for_status()
        return res.json()
    except requests.exceptions.RequestException as e:
        print("❌ Request failed:", e)
        return None
    except ValueError:
        print("❌ Failed to decode JSON response.")
        return None

# 1. יצירת משתמש חדש
user_data = {
    "email": "elin@example.com",
    "password_hash": "abc123",
    "role": "client"
}
user_response = safe_post("/users", user_data)
if not user_response:
    exit()

user_id = user_response.get("user_id")

# 2. הוספת קו"ח (טקסט מדומה)
resume_data = {
    "user_id": user_id,
    "file_name": "cv_elin.pdf",
    "raw_text": """
    Elin is a Machine Learning Engineer with experience in Python, TensorFlow,
    Keras, and AWS. She worked on computer vision projects using OpenCV and PyTorch.
    """,
    "job_title": "Machine Learning Engineer"
}
resume_response = safe_post("/resumes", resume_data)
if not resume_response:
    exit()

resume_id = resume_response.get("resume_id")

# 3. שליחת ניתוח לקו"ח
job_description = """
We are hiring a Machine Learning Engineer with skills in TensorFlow, PyTorch,
Computer Vision, AWS, and strong understanding of deep learning.
"""

analysis_response = safe_post(f"/analyze/{resume_id}", {"job_description_text": job_description})
if analysis_response:
    print("📊 Analysis result:", analysis_response)
