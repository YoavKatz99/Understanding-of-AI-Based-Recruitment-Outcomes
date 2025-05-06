# === dice_setup.py ===
import numpy as np
import pandas as pd
import joblib
import dice_ml
from dice_ml import Dice
from xgboost import XGBRegressor

# Define your manual skills (must match training)
TECH_SKILLS = [
    'python', 'machine learning', 'deep learning', 'data analysis', 'tensorflow',
    'pytorch', 'nlp', 'computer vision', 'classification', 'regression',
    'cloud computing', 'aws', 'gcp', 'azure', 'sql',
    'big data', 'data mining', 'analytics', 'predictive modeling',
    'neural network', 'reinforcement learning', 'keras', 'xgboost',
    'artificial intelligence', 'feature engineering', 'docker',
    'kubernetes', 'hadoop', 'spark', 'time series',
    'unsupervised learning', 'supervised learning', 'bert',
    'transformer', 'cnn', 'rnn', 'lstm', 'gan',
    'attention', 'optimization', 'loss function',
    'hyperparameter tuning', 'cross validation', 'grid search', 'bayesian optimization'
]

# Save TECH_SKILLS if needed elsewhere
joblib.dump(TECH_SKILLS, "feature_list.pkl")

# Create a minimal valid sample DataFrame with 0 and 1 examples
X_sample = np.zeros((2, len(TECH_SKILLS)))
X_sample[1, :] = 1.0
sample_df = pd.DataFrame(X_sample, columns=TECH_SKILLS)
sample_df["Match"] = [0, 1]

# Define feature ranges explicitly (0 to 1)
feature_ranges = {skill: [0.0, 1.0] for skill in TECH_SKILLS}

# Define DiCE Data
data_dice = dice_ml.Data(
    dataframe=sample_df,
    continuous_features=[],
    outcome_name="Match",
    outcome_type="regression",
    data_description={
        "feature_names": TECH_SKILLS,
        "feature_types": ["numerical"] * len(TECH_SKILLS),
        "feature_ranges": feature_ranges
    }
)

# Load trained model
model = joblib.load("xgb_simple_model.pkl")

# Wrap model
class ModelWrapper:
    def __init__(self, model):
        self.model = model

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        probs = self.model.predict(X)
        return np.stack([1 - probs / 100, probs / 100], axis=1)

# Define DiCE model and explainer
model_dice = dice_ml.Model(model=ModelWrapper(model), backend="sklearn")
exp = Dice(data_dice, model_dice, method="random")

# Export for app.py
joblib.dump(exp, "dice_explainer.pkl")
