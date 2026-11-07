
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Load dataset and saved model
data = pd.read_csv("dataset/PhiUSIIL_Phishing_URL_Dataset.csv")

model = joblib.load("models/phishguard_webpage_model_15.pkl")
features = joblib.load("models/webpage_features_15.pkl")

# Prepare features and target
X = data[features]
y = data["label"]

# Recreate the original 80/20 split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Predict only on unseen test records
y_pred = model.predict(X_test)

print("\n========== 15-FEATURE MODEL EVALUATION ==========")

print("\nTraining records:", len(X_train))
print("Unseen test records:", len(X_test))

print("\nAccuracy:")
print(f"{accuracy_score(y_test, y_pred) * 100:.4f}%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Phishing", "Legitimate"]
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\n=================================================")
