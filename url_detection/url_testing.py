import joblib
from pathlib import Path
from url_feature_extractor import extract_url_features

# Get the main project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load the trained model
model = joblib.load(BASE_DIR / "models" / "phishguard_url_model.pkl")

print("\n========== PhishGuard URL Scanner ==========\n")

url = input("Enter the URL you want to check: ").strip()

if not url:
    print("Please enter a valid URL.")
else:
    try:
        # Extract URL features
        features = extract_url_features(url)

        # Make prediction
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]

        # Identify class probabilities
        classes = list(model.classes_)

        phishing_probability = probabilities[classes.index(0)]
        legitimate_probability = probabilities[classes.index(1)]

        print("\n--------------- Scan Results ---------------")
        print("URL:", url)

        print("\nExtracted Features:")
        print(features.T)

        print("\nPrediction:")
        if prediction == 0:
            print("Phishing URL")
        else:
            print("Legitimate URL")

        print("\nPrediction Probabilities:")
        print(f"Phishing: {phishing_probability * 100:.2f}%")
        print(f"Legitimate: {legitimate_probability * 100:.2f}%")

        print("--------------------------------------------")

    except Exception as e:
        print("Unable to scan this URL.")
        print("Error:", e)