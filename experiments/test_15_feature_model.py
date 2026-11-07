
import joblib

model = joblib.load("models/phishguard_webpage_model_15.pkl")
features = joblib.load("models/webpage_features_15.pkl")

print("15-feature model loaded successfully!")
print("\nNumber of features:", len(features))
print("\nSelected features:")

for feature in features:
    print(feature)

print("\nModel expects:", model.n_features_in_, "features")
print("\nModel feature names:")
print(model.feature_names_in_)
