
import pandas as pd
import joblib

# Load dataset and model
data = pd.read_csv("dataset/PhiUSIIL_Phishing_URL_Dataset.csv")

model = joblib.load("models/phishguard_webpage_model_15.pkl")
features = joblib.load("models/webpage_features_15.pkl")

# Select 10 actual records
sample = data[features].head(10)
actual = data["label"].head(10)

# Predict
predictions = model.predict(sample)

# Display results
results = pd.DataFrame({
    "Actual Label": actual.values,
    "Predicted Label": predictions
})

results["Actual Result"] = results["Actual Label"].map({
    0: "Phishing",
    1: "Legitimate"
})

results["Predicted Result"] = results["Predicted Label"].map({
    0: "Phishing",
    1: "Legitimate"
})

print(results.to_string(index=False))

print("\nCorrect Predictions:",
      (actual.values == predictions).sum(), "/ 10")
