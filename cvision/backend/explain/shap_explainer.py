import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from backend.predict_api_rf import extract_features_from_text, process_text, TECH_SKILLS

# טען את המודל
rf_model = joblib.load("backend/random_forest_model.pkl")

# פונקציה שמייצרת הסבר SHAP עבור טקסט קורות חיים
def explain_with_shap(text, show_plot=True):
    features_df = extract_features_from_text(text)
    
    # הסבר באמצעות SHAP
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(features_df)

    # פלט טקסטואלי
    shap_dict = dict(zip(features_df.columns, shap_values[0]))
    sorted_shap = dict(sorted(shap_dict.items(), key=lambda item: abs(item[1]), reverse=True))

    # הדפסת השפעת התכונות
    print("Top Features (SHAP values):")
    for skill, value in list(sorted_shap.items())[:10]:
        print(f"{skill}: {value:.4f}")

    # גרף
    if show_plot:
        shap.initjs()
        shap.force_plot(explainer.expected_value, shap_values[0], features_df.iloc[0], matplotlib=True)
        plt.show()