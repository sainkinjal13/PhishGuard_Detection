

import pandas as pd
import re
from urllib.parse import urlparse

# Load the dataset
data = pd.read_csv(
    "dataset/PhiUSIIL_Phishing_URL_Dataset.csv"
)

# Display dataset information
print("Dataset loaded successfully!")

expected = (data['NoOfObfuscatedChar'] > 0).astype(int)

matches = (data['HasObfuscation'] == expected).sum()
mismatches = (data['HasObfuscation'] != expected).sum()

print("Matching rows:", matches)
print("Mismatching rows:", mismatches)

print(data[[
    'HasObfuscation',
    'NoOfObfuscatedChar',
    'ObfuscationRatio',
    'NoOfEqualsInURL'
]].head(10))

print("\nUnique values in missing features:")

for col in [
    'HasObfuscation',
    'NoOfObfuscatedChar',
    'ObfuscationRatio',
    'NoOfEqualsInURL'
]:
    print(f"\n{col}:")
    print(data[col].value_counts().head(10))

print("\nDataset Shape:")
print(data.shape)

print("\nAvailable Features:")
for column in data.columns:
    print(column)


# URL-based features

url_features = [
    'URLLength',
    'IsDomainIP',
    'IsHTTPS',
    'NoOfSubDomain',
    'HasObfuscation',
    'NoOfObfuscatedChar',
    'ObfuscationRatio',
    'NoOfLettersInURL',
    'LetterRatioInURL',
    'NoOfDegitsInURL',
    'DegitRatioInURL',
    'NoOfEqualsInURL',
    'NoOfQMarkInURL',
    'NoOfAmpersandInURL',
    'NoOfOtherSpecialCharsInURL',
    'SpacialCharRatioInURL',
    'TLDLength',
    'DomainLength'
]

print("\nSelected URL Features:")
for feature in url_features:
    print(feature)

print("\nTotal selected features:", len(url_features))


# Create URL-only dataset

url_data = data[url_features + ['label']].copy()

# Display dataset information
print("\nURL Dataset Shape:")
print(url_data.shape)

print("\nFirst 5 Records:")
print(url_data.head())

print("\nMissing Values:")
print(url_data.isnull().sum().sum())

print("\nDuplicate Records:")
print(url_data.duplicated().sum())


# Check duplicate feature combinations
feature_duplicates = url_data.duplicated(
    subset=url_features
).sum()

print("Duplicate feature combinations:",
      feature_duplicates)

# Check whether duplicate features have conflicting labels
label_counts = (
    url_data.groupby(url_features)["label"]
    .nunique()
)

conflicting_features = label_counts[
    label_counts > 1
]

print("Feature combinations with conflicting labels:",
      len(conflicting_features))

# Find feature combinations with conflicting labels
conflicting_records = url_data[
    url_data.duplicated(subset=url_features, keep=False)
]

conflicting_records = conflicting_records[
    conflicting_records.groupby(url_features)['label']
    .transform('nunique') > 1
]

print("Conflicting Records:")
print(conflicting_records.head(20))

print("\nLabel Distribution:")
print(conflicting_records['label'].value_counts())

# Count records belonging to conflicting feature combinations
conflicting_groups = (
    url_data.groupby(url_features)['label']
    .nunique()
)

conflicting_groups = conflicting_groups[
    conflicting_groups > 1
]

conflicting_records = url_data.merge(
    conflicting_groups.reset_index()[url_features],
    on=url_features,
    how='inner'
)

print("Conflicting feature combinations:", len(conflicting_groups))
print("Total records in conflicting combinations:", len(conflicting_records))

print("\nRecords per label:")
print(conflicting_records['label'].value_counts())

# Count feature combinations and their labels
combination_analysis = (
    url_data.groupby(url_features)['label']
    .value_counts()
    .unstack(fill_value=0)
)

combination_analysis['Total'] = (
    combination_analysis.sum(axis=1)
)

# Show the 10 most frequent combinations
print(
    combination_analysis
    .sort_values('Total', ascending=False)
    .head(10)
)

print("\nTop 10 Feature Combinations:")

top_10 = combination_analysis.sort_values(
    'Total', ascending=False
).head(10)

print(top_10[[0, 1, 'Total']].to_string())

print("\nOverall Label Distribution:")
print(url_data['label'].value_counts())

print("\nLabel Percentages:")
print(
    url_data['label']
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# Separate features and target
X = url_data[url_features]
y = url_data['label']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Train Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)

rf_model.fit(X_train, y_train)

# Predictions
y_pred = rf_model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

from sklearn.model_selection import GroupShuffleSplit

# Keep domains aligned with the URL feature rows
groups = data.loc[url_data.index, 'Domain']

# Split by domain
gss = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    gss.split(X, y, groups=groups)
)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

# Train a fresh model
group_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)

group_model.fit(X_train, y_train)

# Evaluate
y_pred = group_model.predict(X_test)

print("\nDOMAIN-BASED EVALUATION")
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nCommon Domains:",
      len(set(groups.iloc[train_idx]) &
          set(groups.iloc[test_idx])))

import pandas as pd

# Feature importance
importance = pd.DataFrame({
    'Feature': url_features,
    'Importance': group_model.feature_importances_
})

importance = importance.sort_values(
    by='Importance',
    ascending=False
)

print("\nURL Feature Importance:")
print(importance.to_string(index=False))

import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score
)

# Separate a validation set from training domains
groups_train = groups.iloc[train_idx]

gss_val = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=21
)

fit_idx, val_idx = next(
    gss_val.split(X_train, y_train, groups_train)
)

X_fit = X_train.iloc[fit_idx]
X_val = X_train.iloc[val_idx]

y_fit = y_train.iloc[fit_idx]
y_val = y_train.iloc[val_idx]

# Train a model using only the fitting portion
validation_model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)

validation_model.fit(X_fit, y_fit)

# Get phishing probabilities (class 0)
phishing_index = list(validation_model.classes_).index(0)

phishing_probabilities = validation_model.predict_proba(
    X_val
)[:, phishing_index]

# Test different thresholds
for threshold in [0.5, 0.4, 0.3, 0.2]:

    predictions = np.where(
        phishing_probabilities >= threshold,
        0,
        1
    )

    cm = confusion_matrix(
        y_val,
        predictions,
        labels=[0, 1]
    )

    print("\nThreshold:", threshold)
    print("Confusion Matrix:")
    print(cm)

    print(
        "Phishing Precision:",
        round(precision_score(
            y_val, predictions,
            pos_label=0, zero_division=0
        ), 4)
    )

    print(
        "Phishing Recall:",
        round(recall_score(
            y_val, predictions,
            pos_label=0, zero_division=0
        ), 4)
    )
    
    from sklearn.ensemble import ExtraTreesClassifier

extra_model = ExtraTreesClassifier(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)

extra_model.fit(X_fit, y_fit)

# Phishing probabilities
phishing_index = list(extra_model.classes_).index(0)

extra_probabilities = extra_model.predict_proba(
    X_val
)[:, phishing_index]

# Test thresholds
for threshold in [0.5, 0.4, 0.3, 0.2]:

    predictions = np.where(
        extra_probabilities >= threshold,
        0,
        1
    )

    cm = confusion_matrix(
        y_val,
        predictions,
        labels=[0, 1]
    )

    print("\nThreshold:", threshold)
    print("Confusion Matrix:")
    print(cm)

    print("Phishing Recall:", round(
        recall_score(
            y_val,
            predictions,
            pos_label=0
        ), 4
    ))

    print("Phishing Precision:", round(
        precision_score(
            y_val,
            predictions,
            pos_label=0
        ), 4
    ))
    
    # Select top 10 features
top_features = importance.head(10)['Feature'].tolist()

print("Selected Features:")
print(top_features)

# Prepare reduced dataset
X_reduced = url_data[top_features]
y_reduced = url_data['label']

# Domain-based split
groups_reduced = data.loc[url_data.index, 'Domain']

train_idx, test_idx = next(
    gss.split(X_reduced, y_reduced, groups=groups_reduced)
)

X_train_reduced = X_reduced.iloc[train_idx]
X_test_reduced = X_reduced.iloc[test_idx]

y_train_reduced = y_reduced.iloc[train_idx]
y_test_reduced = y_reduced.iloc[test_idx]

# Train model
reduced_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)

reduced_model.fit(X_train_reduced, y_train_reduced)

# Evaluate
predictions = reduced_model.predict(X_test_reduced)

print("\nTop 10 Feature Model Accuracy:")
print(accuracy_score(y_test_reduced, predictions))

print("\nClassification Report:")
print(classification_report(y_test_reduced, predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test_reduced, predictions))


import matplotlib.pyplot as plt
import seaborn as sns

# Select URL features and target
corr_data = url_data[url_features + ['label']]

# Calculate correlation
correlation_matrix = corr_data.corr(method='pearson')

# Plot heatmap
plt.figure(figsize=(18, 14))

sns.heatmap(
    correlation_matrix,
    annot=True,
    cmap='coolwarm',
    fmt='.2f',
    center=0,
    linewidths=0.5
)

plt.title('PhishGuard - URL Feature Correlation Heatmap',
          fontsize=16)

plt.xticks(rotation=75)
plt.yticks(rotation=0)
plt.tight_layout()

plt.show()

import pandas as pd

importance = pd.DataFrame({
    'Feature': url_features,
    'Importance': group_model.feature_importances_
})

importance = importance.sort_values(
    by='Importance',
    ascending=False
)

print(importance.to_string(index=False))

top_features = importance.head(10)['Feature'].tolist()

print("Selected 10 Features:")
print(top_features)

X_top10 = url_data[top_features]
y = url_data['label']

groups = data.loc[url_data.index, 'Domain']

train_idx, test_idx = next(
    gss.split(X_top10, y, groups=groups)
)

X_train = X_top10.iloc[train_idx]
X_test = X_top10.iloc[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]

top10_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)

top10_model.fit(X_train, y_train)

y_pred = top10_model.predict(X_test)

# Confusion Matrix - 10 Feature Model

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1]
)

plt.figure(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Phishing', 'Legitimate'],
    yticklabels=['Phishing', 'Legitimate']
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("PhishGuard - URL Detection Confusion Matrix")

plt.tight_layout()
plt.show()

print("\n10 Feature Model Accuracy:")
print(accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

import joblib

joblib.dump(top10_model, "models/phishguard_url_model.pkl")

joblib.dump(top_features, "models/url_features.pkl")

print("PhishGuard URL model saved successfully!")

def extract_url_features(url):
    
    if not re.match(r"^https?://", url, re.IGNORECASE):
        url = "https://" + url

    parsed_url = urlparse(url)

    domain = parsed_url.netloc.lower().split("@")[-1].split(":")[0]
    path = parsed_url.path
    query = parsed_url.query

    full_url = url
    clean_url = re.sub(
        r"^https?://", "", full_url, flags=re.IGNORECASE
    )

    clean_url = re.sub(
        r"^www\.", "", clean_url, flags=re.IGNORECASE
    )

    clean_url = clean_url.rstrip("/")

    features = {}

    features['URLLength'] = len(url) - 1
    print("Original URL:", url)
    print("Clean URL:", clean_url)
    print("Original length:", len(url))
    print("Clean length:", len(clean_url))
    print("URLLength feature:", features['URLLength'])

    features['IsDomainIP'] = int(
        bool(re.fullmatch(r"\d{1,3}(?:\.\d{1,3}){3}", domain))
    )

    features['IsHTTPS'] = int(parsed_url.scheme.lower() == "https")

    features['NoOfSubDomain'] = max(
        len(domain.split(".")) - 2, 0
    )
    
    features['NoOfLettersInURL'] = sum(
    char.isalpha() for char in clean_url
    ) - 1 if clean_url.lower().startswith("www.") else sum(
    char.isalpha() for char in clean_url
    )
    print("Clean URL:", clean_url)
    print("Letters counted:", sum(char.isalpha() for char in clean_url))
    print("URL without www:", clean_url[4:] if clean_url.lower().startswith("www.") else clean_url)

    features['NoOfDegitsInURL'] = sum(
        char.isdigit() for char in full_url
    )

    features['DomainLength'] = len(domain)

    features['TLDLength'] = (
        len(domain.split(".")[-1]) if "." in domain else 0
    )

    features['NoOfQMarkInURL'] = full_url.count("?")

    features['NoOfAmpersandInURL'] = full_url.count("&")

    features['DegitRatioInURL'] = (
        features['NoOfDegitsInURL'] / len(full_url)
    )

    features['LetterRatioInURL'] = (
        features['NoOfLettersInURL'] / features['URLLength']
    )
    features['NoOfOtherSpecialCharsInURL'] = sum(
    not char.isalnum() for char in clean_url
    )
    
    features['SpacialCharRatioInURL'] = (
        features['NoOfOtherSpecialCharsInURL'] / features['URLLength']
    )
    expected_features = joblib.load("models/url_features.pkl")

    result = pd.DataFrame([features])

    result = result[expected_features]

    return result

# Test URL
test_url = "https://www.aap.org"

clean_test_url = re.sub(
    r"^https?://", "", test_url, flags=re.IGNORECASE
)

clean_test_url = clean_test_url.rstrip("/")

print("Clean test URL:", clean_test_url)
print("Letter count:", sum(c.isalpha() for c in clean_test_url))
print("Letter count without www:", sum(
    c.isalpha() for c in clean_test_url[4:]
))


test_url = "https://www.example.com"

# Extract features
result = extract_url_features(test_url)

# Display results
print("Extracted features:")
print(result)

print("\nFeature count:", len(result.columns))

print("Extracted features:")
print(result)

print("\nFeature count:", len(result.columns))

print("\nExtracted feature names:")
print(result.columns.tolist())

print("\nExpected model features:")
print(joblib.load("models/url_features.pkl"))

print(result.columns.tolist())

print("\nExtracted feature values:")
print(result.T)

# Load trained model
model = joblib.load("models/phishguard_url_model.pkl")

# Predict using extracted features
prediction = model.predict(result)
probabilities = model.predict_proba(result)

print("\nPhishGuard Prediction:")

if prediction[0] == 0:
    print("Result: Potential Phishing URL")
else:
    print("Result: Potentially Legitimate URL")

print("Probabilities:", probabilities)

# Sample URL for testing
sample = data[data['URL'] == "https://www.example.com"]

if not sample.empty:
    print("\nOriginal dataset feature values:")
    print(sample[url_features].iloc[0].to_string())
else:
    print("URL not found in dataset.")
    
# Select an actual URL from the dataset
sample_url = data['URL'].iloc[100]

print("\nSelected URL:")
print(sample_url)

# Original feature values
original_values = data.loc[
    data['URL'] == sample_url, url_features
].iloc[0]

# Extract features from the same URL
extracted_values = extract_url_features(sample_url)

# Compare values
comparison = pd.DataFrame({
    'Original Dataset': original_values,
    'Extracted Values': extracted_values.iloc[0]
})

print("\nFeature Comparison:")
print(comparison)

print("\nURL character analysis:")

print("URL:", sample_url)
print("Length:", len(sample_url))

for char in sorted(set(sample_url)):
    print(
        repr(char),
        "Count:",
        sample_url.count(char),
        "Alphabet:",
        char.isalpha(),
        "Digit:",
        char.isdigit(),
        "Alphanumeric:",
        char.isalnum()
    )
    
print("Dataset URL:", repr(sample_url))
print("Dataset URL length:", len(sample_url))

print("\nDataset feature values:")
print(data.loc[data["URL"] == sample_url, url_features].iloc[0])

expected_features = joblib.load("models/url_features.pkl")

original_values = data.loc[
    data["URL"] == sample_url, expected_features
].iloc[0]

extracted_values = extract_url_features(sample_url).iloc[0]

comparison = pd.DataFrame({
    "Original Dataset": original_values,
    "Extracted Values": extracted_values
})

comparison["Difference"] = (
    comparison["Original Dataset"] - comparison["Extracted Values"]
).abs()

print(comparison)

from collections import Counter

url = sample_url

print("URL:", url)
print("Length:", len(url))

print("\nCharacter groups:")

print("Letters:", sum(c.isalpha() for c in url))
print("Digits:", sum(c.isdigit() for c in url))
print("Alphanumeric:", sum(c.isalnum() for c in url))
print("Non-alphanumeric:", sum(not c.isalnum() for c in url))

print("\nIndividual character counts:")
print(Counter(url))

print("\nURL without protocol:", url.replace("https://", ""))
print("Length without protocol:", len(url.replace("https://", "")))

print("\nURL without final slash:", url.rstrip("/"))
print("Length without final slash:", len(url.rstrip("/")))

for i in range(10):
    url = data["URL"].iloc[i]

    original = data.loc[
        data["URL"] == url, 
        ["URLLength", "NoOfLettersInURL",
         "NoOfOtherSpecialCharsInURL"]
    ].iloc[0]

    print("\nURL:", url)
    print("Actual length:", len(url))
    print("Dataset values:")
    print(original.to_dict())
    
# Prepare URL for character-based feature calculations




for i in range(10):
    url = data["URL"].iloc[i]

    original = data.loc[
        data["URL"] == url,
        ["URLLength", "NoOfLettersInURL",
         "NoOfOtherSpecialCharsInURL"]
    ].iloc[0]

    clean_url = re.sub(
        r"^https?://", "", url, flags=re.IGNORECASE
    )
    clean_url = re.sub(
        r"^www\.", "", clean_url, flags=re.IGNORECASE
    )
    clean_url = clean_url.rstrip("/")

    extracted = {
        "URLLength": len(url) - 1,
        "NoOfLettersInURL": sum(c.isalpha() for c in clean_url),
        "NoOfOtherSpecialCharsInURL": sum(
            not c.isalnum() for c in clean_url
        )
    }

    print("\nURL:", url)
    print("Dataset:", original.to_dict())
    print("Extracted:", extracted)
    
test_url = "https://auth-document-shared.datingg-voice.workers.dev/"

print(extract_url_features(test_url).T)

test_urls = [
    "https://www.aap.org",
    "https://aap.org"
]

for test_url in test_urls:
    print("\nTesting:", test_url)
    print(extract_url_features(test_url))
    
aap_rows = data[
    data['URL'].str.contains('aap.org', case=False, na=False)
]

print(
    aap_rows[
        ['URL', 'URLLength', 'NoOfLettersInURL',
         'NoOfOtherSpecialCharsInURL']
    ].to_string(index=False)
)

test_url = "https://www.aap.org"

print("URL:", test_url)
print("Length:", len(test_url))

for i, char in enumerate(test_url):
    print(i, repr(char), char.isalpha())
    
print("Letters in original URL:",
      sum(char.isalpha() for char in test_url))

print("Letters after removing protocol:",
      sum(char.isalpha() for char in
          re.sub(r"^https?://", "", test_url)))

print("Letters after removing www:",
      sum(char.isalpha() for char in
          re.sub(r"^www\.", "",
                 re.sub(r"^https?://", "", test_url))))

test_urls = [
    "https://www.aap.org",
    "https://www.recaap.org"
]

for test_url in test_urls:
    clean_url = re.sub(r"^https?://", "", test_url)
    clean_url = re.sub(r"^www\.", "", clean_url)

    print("\nURL:", test_url)
    print("Clean URL:", clean_url)
    print("Letters:", sum(char.isalpha() for char in clean_url))
    
print("\nExamples of URLs with obfuscation:")

print(
    data.loc[
        data['HasObfuscation'] == 1,
        ['URL', 'HasObfuscation', 'NoOfObfuscatedChar',
         'ObfuscationRatio', 'NoOfEqualsInURL']
    ].head(10).to_string(index=False)
)

import re

sample = data[data['HasObfuscation'] == 1].head(10)

for _, row in sample.iterrows():
    url = row['URL']

    encoded_parts = re.findall(r'%[0-9a-fA-F]{2}', url)

    print("\nURL:", url)
    print("Dataset URLLength:", row['URLLength'])
    print("Encoded sequences:", len(encoded_parts))
    print("Encoded characters (3 per sequence):", len(encoded_parts) * 3)
    print("Dataset NoOfObfuscatedChar:", row['NoOfObfuscatedChar'])
    print("Dataset ObfuscationRatio:", row['ObfuscationRatio'])
    
row = data[data['URL'].str.contains(
    'digital3riiversaccess', na=False
)].iloc[0]

print(row[['URL', 'URLLength', 'HasObfuscation',
           'NoOfObfuscatedChar', 'ObfuscationRatio']].to_string())

sample = data[data['HasObfuscation'] == 1].head(10)

for _, row in sample.iterrows():
    print("\nURL:", row['URL'])
    print("HasObfuscation:", row['HasObfuscation'])
    print("NoOfObfuscatedChar:", row['NoOfObfuscatedChar'])
    print("ObfuscationRatio:", row['ObfuscationRatio'])
    
import re

obfuscated = data[data['HasObfuscation'] == 1].copy()

obfuscated['EncodedCount'] = obfuscated['URL'].apply(
    lambda url: len(re.findall(r'%[0-9a-fA-F]{2}', str(url)))
)

obfuscated['CalculatedCount'] = obfuscated['EncodedCount'] * 3

mismatches = obfuscated[
    obfuscated['NoOfObfuscatedChar'] != obfuscated['CalculatedCount']
]

print("\nFirst 15 mismatched URLs:")

print(
    mismatches[
        ['URL', 'EncodedCount', 'CalculatedCount',
         'NoOfObfuscatedChar']
    ].head(15).to_string(index=False)
)

url = mismatches.iloc[0]['URL']

print("URL:", url)
print("\nEncoded sequences:")

print(re.findall(r'%[0-9a-fA-F]{2}', url))

print("\nDataset count:",
      mismatches.iloc[0]['NoOfObfuscatedChar'])

url = mismatches.iloc[1]['URL']

print("URL:", url)

print("\nEncoded sequences:")
print(re.findall(r'%[0-9a-fA-F]{2}', url))

print("\nNumber of encoded sequences:",
      len(re.findall(r'%[0-9a-fA-F]{2}', url)))

print("\nDataset count:",
      mismatches.iloc[1]['NoOfObfuscatedChar'])

import re
from urllib.parse import unquote

url = mismatches.iloc[1]['URL']

encoded = re.findall(r'%[0-9a-fA-F]{2}', url)

print("Encoded sequences and decoded characters:")

for item in encoded:
    print(item, "->", unquote(item))

print("\nDataset NoOfObfuscatedChar:",
      mismatches.iloc[1]['NoOfObfuscatedChar'])

url = mismatches.iloc[1]['URL']

print("URL:", url)
print("\nPercent sign positions:")

for i, char in enumerate(url):
    if char == '%':
        print(i, url[i:i+3])

print("\nTotal URL length:", len(url))
print("Dataset count:", mismatches.iloc[1]['NoOfObfuscatedChar'])

from urllib.parse import urlparse
import re

url = mismatches.iloc[1]['URL']
parsed = urlparse(url)

print("Domain:", parsed.netloc)
print("Path:", parsed.path)
print("Query:", parsed.query)

for name, part in [
    ("Domain", parsed.netloc),
    ("Path", parsed.path),
    ("Query", parsed.query)
]:
    count = len(re.findall(r'%[0-9a-fA-F]{2}', part))
    print(name, "encoded sequences:", count)