
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load dataset
data = pd.read_csv("dataset/PhiUSIIL_Phishing_URL_Dataset.csv")

# 2. Select the 15 features
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

# 3. Calculate feature correlation with target
correlation = data[features + ["label"]].corr(numeric_only=True)

target_corr = (
    correlation["label"]
    .drop("label")
    .sort_values(key=abs, ascending=False)
)

print("\n========== FEATURE CORRELATION WITH LABEL ==========")
print(target_corr)

# 4. Display correlation heatmap
plt.figure(figsize=(14, 10))

sns.heatmap(
    correlation,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)

plt.title("15-Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

# 5. Check highly correlated feature pairs
print("\n========== HIGHLY CORRELATED FEATURE PAIRS ==========")

feature_corr = data[features].corr(numeric_only=True)

for i in range(len(features)):
    for j in range(i + 1, len(features)):
        value = feature_corr.iloc[i, j]

        if abs(value) >= 0.90:
            print(
                features[i],
                "<-->",
                features[j],
                ":",
                round(value, 4)
            )

# 6. Check unique values
print("\n========== UNIQUE VALUE COUNTS ==========")

for feature in features:
    print(feature, ":", data[feature].nunique())

print("\n==============================================")
