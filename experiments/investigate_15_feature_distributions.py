
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load dataset
data = pd.read_csv("dataset/PhiUSIIL_Phishing_URL_Dataset.csv")

# 2. Select 15 features
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

# 3. Compare average feature values by class
print("\n========== FEATURE DISTRIBUTION ==========")

summary = data.groupby("label")[features].agg(
    ["mean", "median", "std"]
)

print("\nLabel 0 = Phishing")
print("Label 1 = Legitimate")

print(summary.T)

# 4. Compare binary feature distributions
binary_features = [
    feature for feature in features
    if data[feature].nunique() == 2
]

print("\n========== BINARY FEATURE DISTRIBUTIONS ==========")

for feature in binary_features:
    print(f"\n{feature}")

    distribution = pd.crosstab(
        data[feature],
        data["label"],
        normalize="columns"
    ) * 100

    print(distribution.round(2))

# 5. Plot distributions of important numerical features
numerical_features = [
    'URLSimilarityIndex',
    'DomainTitleMatchScore',
    'URLTitleMatchScore',
    'URLCharProb',
    'CharContinuationRate',
    'SpacialCharRatioInURL'
]

for feature in numerical_features:
    plt.figure(figsize=(8, 5))

    sns.boxplot(
        data=data,
        x="label",
        y=feature,
        hue="label",
        legend=False
    )

    plt.title(f"{feature} Distribution by Class")
    plt.xlabel("Label (0 = Phishing, 1 = Legitimate)")
    plt.ylabel(feature)
    plt.tight_layout()
    plt.show()

print("\nDistribution analysis completed!")
