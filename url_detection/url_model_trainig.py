
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Locate main project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load dataset
data = pd.read_csv(
    BASE_DIR / "dataset" / "PhiUSIIL_Phishing_URL_Dataset.csv"
)

print(data)

# Basic dataset analysis
print("\nDataset Information:")
data.info()

print("\nDataset Description:")
print(data.describe())

print("\nMissing Values:")
print(data.isnull().sum())

print("\nNumber of duplicate rows:", data.duplicated().sum())


# Select numerical columns
numerical_data = data.select_dtypes(
    include=['int64', 'float64']
)

# Calculate correlation
correlation = numerical_data.corr()

# Plot correlation heatmap
plt.figure(figsize=(20, 16))

sns.heatmap(
    correlation,
    annot=False,
    cmap='coolwarm',
    linewidths=0.3,
    center=0
)

plt.title(
    "Correlation Heatmap - PhiUSIIL Phishing URL Dataset",
    fontsize=16
)

plt.tight_layout()
plt.show()


# Calculate correlation with target label
target_corr = (
    numerical_data.corr()['label']
    .drop('label')
)

# Sort by absolute correlation
top_features = target_corr.abs().sort_values(
    ascending=False
).head(20)

print("\nTop 20 Features:")
print(top_features)


# Select 15 features
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

X = data[selected_features]
y = data['label']

# Selected features correlation
corr_matrix = data[selected_features].corr()

plt.figure(figsize=(14, 10))

sns.heatmap(
    corr_matrix,
    annot=True,
    cmap='coolwarm',
    fmt='.2f',
    center=0
)

plt.title("Selected Features Correlation")
plt.tight_layout()
plt.show()

print("\nNumber of selected features:", X.shape[1])
print("Feature dataset shape:", X.shape)
print("Target dataset shape:", y.shape)


# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nX_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)


# Create Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

# Train model
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)


# Evaluate model
print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)


# Display confusion matrix
plt.figure(figsize=(8, 6))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    linewidths=0.5,
    xticklabels=['Phishing (0)', 'Legitimate (1)'],
    yticklabels=['Phishing (0)', 'Legitimate (1)']
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix - PhishGuard")

plt.tight_layout()
plt.show()


# Feature importance
importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': rf_model.feature_importances_
}).sort_values(
    by='Importance',
    ascending=False
)

print("\nFeature Importance:")
print(importance)


# Plot feature importance
plt.figure(figsize=(10, 7))

sns.barplot(
    data=importance,
    x='Importance',
    y='Feature'
)

plt.title("Random Forest Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Features")

plt.tight_layout()
plt.show()


# Save model and features in models folder
joblib.dump(
    rf_model,
    BASE_DIR / "models" / "phishguard_url_model.pkl"
)

print("\nModel saved successfully!")

joblib.dump(
    selected_features,
    BASE_DIR / "models" / "url_features.pkl"
)

print("Selected features saved successfully!")
