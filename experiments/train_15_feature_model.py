
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# 1. Load dataset
data = pd.read_csv(
    "dataset/PhiUSIIL_Phishing_URL_Dataset.csv"
)

print("Dataset shape:", data.shape)

# 2. Select the original 15 features
selected_features = [
    'URLSimilarityIndex',
    'HasSocialNet',
    'HasCopyrightInfo',
    'HasDescription',
    'IsHTTPS',
    'DomainTitleMatchScore',
    'HasSubmitButton',
    'IsResponsive',
    'URLTitleMatchScore',
    'SpacialCharRatioInURL',
    'HasHiddenFields',
    'HasFavicon',
    'URLCharProb',
    'CharContinuationRate',
    'HasTitle'
]

# 3. Prepare X and y
X = data[selected_features]
y = data['label']

print("\nNumber of features:", X.shape[1])
print("Feature shape:", X.shape)
print("Target shape:", y.shape)

# Check missing values
print("\nMissing values:")
print(X.isnull().sum().sum())

# 4. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining shape:", X_train.shape)
print("Testing shape:", X_test.shape)

# 5. Train Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

# 6. Evaluate model
y_pred = rf_model.predict(X_test)

print("\n========== MODEL RESULTS ==========")

print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 7. Save model separately
joblib.dump(
    rf_model,
    "models/phishguard_webpage_model_15.pkl"
)

joblib.dump(
    selected_features,
    "models/webpage_features_15.pkl"
)

print("\n15-feature model saved successfully!")
