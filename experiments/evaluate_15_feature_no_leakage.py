
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# 1. Load dataset
data = pd.read_csv("dataset/PhiUSIIL_Phishing_URL_Dataset.csv")

# 2. Separate features and target
X = data.select_dtypes(include=["int64", "float64"]).drop(
    columns=["label"]
)
y = data["label"]
groups = data["Domain"]

# 3. Split by domain BEFORE feature selection
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

# 4. Calculate correlations using training data only
train_corr = X_train.copy()
train_corr["label"] = y_train.values

target_corr = (
    train_corr.corr(numeric_only=True)["label"]
    .drop("label")
)

# 5. Select top 15 features
selected_features = (
    target_corr.abs()
    .sort_values(ascending=False)
    .head(15)
    .index.tolist()
)

print("\nSelected 15 features:")
for feature in selected_features:
    print(feature)

# 6. Prepare selected training and testing data
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

# 7. Train Random Forest
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_selected, y_train)

# 8. Evaluate
y_pred = model.predict(X_test_selected)

print("\n========== NO-LEAKAGE EVALUATION ==========")

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

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

print("\n===========================================")
