import joblib
import pandas as pd

# Load saved model
model = joblib.load("models/phishguard_url_model.pkl")
# Load selected features
selected_features = joblib.load("models/url_features.pkl")

print("Model loaded successfully!")
print("\nSelected features:")
print(selected_features)

print("\nNumber of features:", len(selected_features))

print("\nModel expects features:",
      model.n_features_in_)

print("\nModel feature names:")
print(model.feature_names_in_)