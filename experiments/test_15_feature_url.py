
import pandas as pd
import joblib

# Load dataset and model
data = pd.read_csv("dataset/PhiUSIIL_Phishing_URL_Dataset.csv")

model = joblib.load("models/phishguard_webpage_model_15.pkl")
features = joblib.load("models/webpage_features_15.pkl")

print("\n========== PhishGuard 15-Feature Scanner ==========\n")

url = input("Enter URL: ").strip()

# Find URL column
url_column = "URL"

if url_column not in data.columns:
    print("URL column not found in dataset.")
else:
    # Search for exact URL
    matching_rows = data[
        data[url_column].astype(str).str.strip() == url
    ]

    if matching_rows.empty:
        print("\nURL not found in the dataset.")
        print("This URL cannot be tested using this method.")
    else:
        # Extract the 15 features
        input_features = matching_rows[features].iloc[[0]]

        # Predict
        prediction = model.predict(input_features)[0]
        probabilities = model.predict_proba(input_features)[0]

        classes = list(model.classes_)

        phishing_probability = probabilities[classes.index(0)]
        legitimate_probability = probabilities[classes.index(1)]

        print("\n--------------- Scan Results ---------------")
        print("URL:", url)

        print("\nPrediction:")

        if prediction == 0:
            print("Phishing URL")
        else:
            print("Legitimate URL")

        print("\nPrediction Probabilities:")
        print(f"Phishing: {phishing_probability * 100:.2f}%")
        print(f"Legitimate: {legitimate_probability * 100:.2f}%")

        print("--------------------------------------------")
