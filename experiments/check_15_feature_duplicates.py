
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# 1. Load dataset
data = pd.read_csv("dataset/PhiUSIIL_Phishing_URL_Dataset.csv")

# 2. Select the same 15 features
features = [
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

X = data[features]
y = data["label"]
groups = data["Domain"]

# 3. Recreate the same domain-based split
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_index, test_index = next(
    splitter.split(X, y, groups=groups)
)

X_train = X.iloc[train_index].copy()
X_test = X.iloc[test_index].copy()

y_train = y.iloc[train_index]
y_test = y.iloc[test_index]

# 4. Attach labels
train_data = X_train.copy()
train_data["train_label"] = y_train.values

test_data = X_test.copy()
test_data["actual_label"] = y_test.values

# 5. Find unique training feature combinations
train_patterns = (
    train_data
    .groupby(features, dropna=False)
    .agg(
        train_label=("train_label", "first"),
        label_count=("train_label", "nunique"),
        train_records=("train_label", "size")
    )
    .reset_index()
)

# 6. Compare test records with training combinations
comparison = test_data.merge(
    train_patterns,
    on=features,
    how="left",
    indicator=True
)

matched = comparison[comparison["_merge"] == "both"]

# 7. Count results
same_label = matched[
    (matched["label_count"] == 1) &
    (matched["actual_label"] == matched["train_label"])
]

conflicting_label = matched[
    (matched["label_count"] == 1) &
    (matched["actual_label"] != matched["train_label"])
]

ambiguous_label = matched[matched["label_count"] > 1]

print("\n========== DUPLICATE FEATURE ANALYSIS ==========")

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

print("\nUnique training feature combinations:",
      len(train_patterns))

print("Test records matching a training combination:",
      len(matched))

print("Test records with matching labels:",
      len(same_label))

print("Test records with conflicting labels:",
      len(conflicting_label))

print("Test records matching ambiguous training labels:",
      len(ambiguous_label))

print("\n================================================")
