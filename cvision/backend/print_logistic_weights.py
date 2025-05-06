# check_model_type.py
import joblib

model = joblib.load("xgb_regressor_model.pkl")  # or "xgb_regressor_model.pkl"
print("✅ Loaded model class:", type(model))
print("✅ Model objective:", getattr(model, 'objective', 'Not found'))

try:
    print("✅ Has predict_proba?:", hasattr(model, 'predict_proba'))
except Exception as e:
    print("⛔ Error checking predict_proba:", e)
