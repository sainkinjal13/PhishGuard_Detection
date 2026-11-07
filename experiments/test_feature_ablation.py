
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# 1. Load dataset
data = pd.read_csv("dataset/PhiUSIIL_Phishing_URL_Dataset.csv")

# 2. Define feature groups
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

without_similarity = [
    f for f in all_features
    if f != 'URLSimilarityIndex'
]

without_all_similarity = [
    f for f in all_features
    if f not in [
        'URLSimilarityIndex',
        'DomainTitleMatchScore',
        'URLTitleMatchScore'
    ]
]

# 3. Domain-based split
X = data[all_features]
y = data["label"]
groups = data["Domain"]

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_index, test_index = next(
    splitter.split(X, y, groups=groups)
)

y_train = y.iloc[train_index]
y_test = y.iloc[test_index]

# 4. Define experiments
experiments = {
    "All 15 Features": all_features,
    "Without URLSimilarityIndex": without_similarity,
    "Without 3 Similarity Features": without_all_similarity
}

results = []

# 5. Train and evaluate each model
for name, features in experiments.items():

    X_train = data.iloc[train_index][features]
    X_test = data.iloc[test_index][features]

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    )

    results.append({
        "Experiment": name,
        "Features": len(features),
        "Accuracy (%)": round(
            accuracy_score(y_test, predictions) * 100, 4
        ),
        "Phishing Missed": cm[0, 1],
        "False Positives": cm[1, 0]
    })

# 6. Display comparison
results_df = pd.DataFrame(results)

print("\n========== FEATURE ABLATION RESULTS ==========")
print(results_df.to_string(index=False))
print("\n==============================================")
