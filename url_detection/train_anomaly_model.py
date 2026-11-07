
import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import IsolationForest

# Locate the main project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load dataset
data = pd.read_csv(
    BASE_DIR / "dataset" / "PhiUSIIL_Phishing_URL_Dataset.csv"
)

# Load existing URL model's feature order
features = joblib.load(
    BASE_DIR / "models" / "url_features.pkl"
)

# Keep only legitimate URLs (label = 1)
legitimate_data = data[data["label"] == 1]

# Select the same 10 features
X = legitimate_data[features]

# Train Isolation Forest
anomaly_model = IsolationForest(
    n_estimators=100,
    contamination=0.05,
    random_state=42
)

anomaly_model.fit(X)

# Save the model
joblib.dump(
    anomaly_model,
    BASE_DIR / "models" / "phishguard_anomaly_model.pkl"
)

print("Anomaly Detection Model trained successfully!")
print("Training samples:", len(X))
print("Features used:", len(features))
print("Model saved successfully!")
