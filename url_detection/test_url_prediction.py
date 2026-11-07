
import joblib
from pathlib import Path
from url_feature_extractor import extract_url_features

# Locate the main project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load URL phishing detection model
model = joblib.load(
    BASE_DIR / "models" / "phishguard_url_model.pkl"
)

url = input("Enter URL to check: ")

features = extract_url_features(url)

prediction = model.predict(features)[0]
probability = model.predict_proba(features)[0]

print("\n---------- PhishGuard Results ----------")

print("URL:", url)

if prediction == 0:
    print("Prediction: PHISHING")
else:
    print("Prediction: LEGITIMATE")

print("Phishing Probability:", round(probability[0] * 100, 2), "%")
print("Legitimate Probability:", round(probability[1] * 100, 2), "%")
