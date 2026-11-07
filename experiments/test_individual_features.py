
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# 1. Load dataset
data = pd.read_csv("dataset/PhiUSIIL_Phishing_URL_Dataset.csv")

# 2. Define features
individual_features = [
    "URLSimilarityIndex",
    "IsHTTPS",
    "HasTitle"
]

all_features = [
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

# 3. Create the same domain-based split
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

# 4. Test individual features
results = []

for feature in individual_features + ["ALL_15_FEATURES"]:

    if feature == "ALL_15_FEATURES":
        selected = all_features
    else:
        selected = [feature]

    X_train = data.iloc[train_index][selected]
    X_test = data.iloc[test_index][selected]

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions, labels=[0, 1])

    results.append({
        "Feature": feature,
        "Accuracy (%)": round(accuracy * 100, 4),
        "Phishing Missed": cm[0, 1],
        "False Positives": cm[1, 0]
    })

# 5. Display comparison
results_df = pd.DataFrame(results)

print("\n========== INDIVIDUAL FEATURE COMPARISON ==========")
print(results_df.to_string(index=False))
print("\n===================================================")
