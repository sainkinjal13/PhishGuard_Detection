
import pandas as pd
import joblib

from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# 1. Load dataset
data = pd.read_csv("dataset/PhiUSIIL_Phishing_URL_Dataset.csv")

# 2. Load selected features
features = joblib.load("models/webpage_features_15.pkl")

X = data[features]
y = data["label"]

# 3. Use Domain as the grouping column
groups = data["Domain"]

# 4. Split by domain
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_index, test_index = next(
    splitter.split(X, y, groups=groups)
)

X_train = X.iloc[train_index]
X_test = X.iloc[test_index]

y_train = y.iloc[train_index]
y_test = y.iloc[test_index]

# 5. Train a fresh Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# 6. Predict unseen domains
y_pred = model.predict(X_test)

# 7. Evaluate
print("\n========== DOMAIN-BASED EVALUATION ==========")

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

print("\nTraining domains:", groups.iloc[train_index].nunique())
print("Testing domains:", groups.iloc[test_index].nunique())

print(
    "\nOverlapping domains:",
    len(
        set(groups.iloc[train_index])
        & set(groups.iloc[test_index])
    )
)

print("\nAccuracy:")
print(f"{accuracy_score(y_test, y_pred) * 100:.4f}%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=["Phishing", "Legitimate"],
    zero_division=0
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=[0, 1]))

print("\n=============================================")
