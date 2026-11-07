
import joblib
from pathlib import Path
from url_feature_extractor import extract_url_features

# Locate the main project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load anomaly detection model
model = joblib.load(
    BASE_DIR / "models" / "phishguard_anomaly_model.pkl"
)

url = input("Enter URL to check: ")

# Extract URL features
features = extract_url_features(url)

# Predict anomaly
result = model.predict(features)[0]

print("\n---------- PhishGuard Anomaly Results ----------")
print("URL:", url)

if result == -1:
    print("Anomaly Status: UNUSUAL URL PATTERN")
else:
    print("Anomaly Status: NORMAL URL PATTERN")
