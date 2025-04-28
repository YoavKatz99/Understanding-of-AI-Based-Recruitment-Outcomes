import joblib

vectorizer = joblib.load("tfidf_vectorizer.pkl")
features = vectorizer.get_feature_names_out()

print(f"Total number of features: {len(features)}")
for i, word in enumerate(features):
    print(f"{i+1}. {word}")
