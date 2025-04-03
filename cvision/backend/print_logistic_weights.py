import joblib
import pandas as pd

# טען את המודל
model = joblib.load("logistic_model.pkl")

# טען את שמות הפיצ'רים
with open("feature_names.txt") as f:
    feature_names = [line.strip() for line in f]

# קבל את המשקלים
coefficients = model.coef_[0]

# צור טבלה מסודרת
df_weights = pd.DataFrame({
    "Feature": feature_names,
    "Weight": coefficients
})

# מיין לפי התרומה הגבוהה ביותר
df_weights = df_weights.sort_values(by="Weight", ascending=False)

print("\n📊 Logistic Regression Feature Weights:\n")
print(df_weights.to_string(index=False))
