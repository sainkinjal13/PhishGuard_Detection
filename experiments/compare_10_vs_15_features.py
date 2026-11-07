
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

# 2. Define 10-feature and 15-feature sets

features_10 = [
    'IsHTTPS',
    'NoOfOtherSpecialCharsInURL',
    'DegitRatioInURL',
    'URLLength',
    'LetterRatioInURL',
    'NoOfDegitsInURL',
    'SpacialCharRatioInURL',
    'NoOfSubDomain',
    'NoOfLettersInURL',
    'DomainLength'
]

features_15 = [
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

y = data["label"]
groups = data["Domain"]

# 3. Create one identical domain-based split
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_index, test_index = next(
    splitter.split(data, y, groups=groups)
)

y_train = y.iloc[train_index]
y_test = y.iloc[test_index]

# 4. Evaluate both models
experiments = {
    "10-Feature URL Model": features_10,
    "15-Feature Webpage Model": features_15
}

results = []

for name, features in experiments.items():

    X_train = data.iloc[train_index][features]
    X_test = data.iloc[test_index][features]

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    )

    accuracy = accuracy_score(y_test, predictions)

    print("\n==========", name, "==========")
    print(f"Accuracy: {accuracy * 100:.4f}%")
    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Phishing", "Legitimate"],
        zero_division=0
    ))

    print("Confusion Matrix:")
    print(cm)

    results.append({
        "Model": name,
        "Features": len(features),
        "Accuracy (%)": round(accuracy * 100, 4),
        "Phishing Missed": cm[0, 1],
        "False Positives": cm[1, 0]
    })

# 5. Final comparison
print("\n========== FINAL MODEL COMPARISON ==========")

comparison = pd.DataFrame(results)
print(comparison.to_string(index=False))

print("\n============================================")
