
![תמונה1](https://github.com/user-attachments/assets/0d6429fd-8ffb-4985-997b-1c5ea58a9ebb)
# Understanding AI-Based Recruitment Outcomes

## 🔍 Overview

This project aims to bring transparency to AI-driven recruitment systems by using **Explainable AI (XAI)** techniques — specifically **SHAP**, **LIME**, and **DiCE** — to explain how resumes are evaluated. By analyzing resume content and comparing it to job descriptions, the system reveals the logic behind candidate scores and suggests actionable improvements.

---

## 🎯 Objectives

- ✅ Help candidates understand why their resume received a specific score.
- ✅ Visualize which skills and keywords influenced the outcome.
- ✅ Offer guidance for improving match scores via counterfactual suggestions.
- ✅ Increase fairness, trust, and satisfaction in automated hiring systems.

---

## 🧠 How It Works

- **SHAP** – Global feature importance: reveals which skills or keywords most influenced your match score.
- **LIME** – Local explanation: highlights the parts of your individual resume that mattered most for this score.
- **DiCE** – Counterfactuals: suggests what you can change (e.g., add skills) to improve your score.

---

## 🧪 User Guide

### Step 1: Select an Explanation Tool
Choose from:
- **SHAP** – Feature importance bar chart.
- **LIME** – Resume text with highlights.
- **DiCE** – Suggested changes to improve score.

### Step 2: Upload Resume
Click **“Choose File”** and upload your resume (PDF only).

### Step 3: Analyze
Click the **“Analyze”** button to run the model and generate an explanation.

### Step 4: Review Results
- **Match Score**: Percentage indicating how well your resume matches the job.
- **Explanation Output**: A bar chart, highlighted text, or skill suggestions, depending on the tool.
- **Interpretation Tips**: Guidance to help you act on the explanation.

---

## ⚙️ System Architecture

### 📁 Frontend (React)
- Allows users to upload resumes, select tools, and view explanations.

### 🧪 Backend (Flask)
- Handles resume parsing, model inference, and explanation generation.
- Uses **pdfminer.six** for text extraction and **TF-IDF + XGBoost** for scoring.

### 🔧 Models
- `xgb_text_model.pkl`: For SHAP and LIME (text-based XGBoost regression).
- `xgb_regressor_model.pkl`: For DiCE (binary skill-based XGBoost model).
- `tfidf_vectorizer.pkl`: TF-IDF transformer.
- `feature_list.pkl`: List of technical skills used by the DiCE model.

All models are saved using `joblib` and located in the `/backend` directory.

---

## 📂 Output Files

- `outputs/` folder (inside `/backend`) will contain:
  - SHAP: `shap_plot.png`
  - LIME: `lime_explanation.html`
  - DiCE: JSON with suggested skills or changes

---

## 🧪 Testing the App

### ✅ Prerequisites
- Python 3.8–3.10
- Node.js & npm for frontend
- Install backend dependencies:  
  ```bash
  pip install -r requirements.txt
  ```

### 🧪 Run Backend Tests

1. Start the Flask backend at `http://localhost:5000`
2. Place a test resume as `test_resume.pdf` in the root folder
3. Run tests:
   ```bash
   python test_app.py
   ```

### ✅ What is Tested
- **SHAP**: Checks prediction score + saved SHAP image
- **LIME**: Verifies valid HTML explanation is returned
- **DiCE**: Ensures counterfactual suggestions are received

---

## 🔁 Model Update Instructions

If retraining:
1. Replace `.pkl` model files in `/backend`.
2. Ensure:
   - **TF-IDF vocabulary remains unchanged**
   - **Skill feature list for DiCE matches the original `feature_list.pkl`**

---

## 🛠️ System Requirements

- **Python 3.8–3.10**
- **Flask**, **xgboost**, **scikit-learn**, **pdfminer.six**, **spacy**, **lime**, **dice-ml**
- **Node.js** for frontend
- Recommended: use virtual environments

---
