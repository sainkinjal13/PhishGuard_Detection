import joblib

features = joblib.load("models/url_features.pkl")

print("\nExpected features:")
for i, feature in enumerate(features, start=1):
    print(f"{i}. {feature}")

print("\nTotal expected features:", len(features))