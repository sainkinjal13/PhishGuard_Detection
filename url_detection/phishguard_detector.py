
import joblib
from pathlib import Path
from url_feature_extractor import extract_url_features

# Get the main project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load both trained models
url_model = joblib.load(
    BASE_DIR / "models" / "phishguard_url_model.pkl"
)

anomaly_model = joblib.load(
    BASE_DIR / "models" / "phishguard_anomaly_model.pkl"
)


def scan_url(url):

    features = extract_url_features(url)

    # Phishing prediction
    prediction = url_model.predict(features)[0]
    probabilities = url_model.predict_proba(features)[0]

    # Anomaly detection
    anomaly = anomaly_model.predict(features)[0]

    print("\n========== PHISHGUARD SCAN RESULTS ==========")
    print("URL:", url)

    if prediction == 0:
        print("Phishing Detection: PHISHING")
    else:
        print("Phishing Detection: LEGITIMATE")

    print("Phishing Probability:", round(probabilities[0] * 100, 2), "%")
    print("Legitimate Probability:", round(probabilities[1] * 100, 2), "%")

    if anomaly == -1:
        print("Anomaly Detection: UNUSUAL URL PATTERN")
    else:
        print("Anomaly Detection: NORMAL URL PATTERN")

    print("=============================================")


if __name__ == "__main__":
    url = input("Enter URL to scan: ")
    scan_url(url)
