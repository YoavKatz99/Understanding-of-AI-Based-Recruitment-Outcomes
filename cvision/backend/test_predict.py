import requests

url = "http://127.0.0.1:5000/predict"
file_path = "test_resume.pdf"  # ודא שקובץ PDF זה קיים בתיקייה שלך

with open(file_path, "rb") as f:
    files = {"file": f}
    response = requests.post(url, files=files)

print("Status Code:", response.status_code)
print("Response:", response.json())
