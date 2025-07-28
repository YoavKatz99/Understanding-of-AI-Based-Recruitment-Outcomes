
import requests
import os

BASE_URL = "http://localhost:5000"
RESUME_FILE = "test_resume.pdf"  # Make sure this file exists in the current directory

def test_explain_shap():
    with open(RESUME_FILE, 'rb') as f:
        files = {'file': f}
        data = {'tool': 'shap'}
        r = requests.post(f"{BASE_URL}/explain", files=files, data=data)
        assert r.status_code == 200, "❌ Failed to get SHAP explanation"
        resp = r.json()
        assert "prediction" in resp, "❌ Missing prediction in SHAP response"
        assert "explanation" in resp, "❌ Missing explanation in SHAP response"
        print("✅ SHAP match score:", resp["prediction"])
        print("✅ SHAP explanation:", resp["explanation"])

def test_shap_plot_created():
    shap_plot_path = os.path.join("outputs", "shap_importance.png")
    assert os.path.exists(shap_plot_path), "❌ SHAP plot not found"
    assert os.path.getsize(shap_plot_path) > 0, "❌ SHAP plot is empty"
    print("✅ SHAP plot created successfully:", shap_plot_path)

def test_explain_lime_text():
    with open(RESUME_FILE, 'rb') as f:
        files = {'file': f}
        r = requests.post(f"{BASE_URL}/explain_lime_text", files=files)
        assert r.status_code == 200, "❌ Failed to get LIME explanation"
        resp = r.json()
        assert "explanation" in resp, "❌ Missing LIME explanation"
        print("✅ LIME explanation preview:", resp["explanation"][:150])

def test_explain_dice():
    with open(RESUME_FILE, 'rb') as f:
        files = {'file': f}
        r = requests.post(f"{BASE_URL}/explain_carla", files=files)
        assert r.status_code == 200, "❌ Failed to get DiCE explanation"
        resp = r.json()
        assert "prediction" in resp, "❌ Missing prediction in DiCE response"
        assert "counterfactuals" in resp, "❌ Missing counterfactuals"
        print("✅ DiCE prediction:", resp["prediction"])
        print("✅ DiCE explanation summary:", resp["explanations"]["summary"])

if __name__ == "__main__":
    test_explain_shap()
    test_shap_plot_created()
    test_explain_lime_text()
    test_explain_dice()
