
import pandas as pd
from pathlib import Path
from url_feature_extractor import extract_url_features

# Locate the main project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load dataset
data = pd.read_csv(
    BASE_DIR / "dataset" / "PhiUSIIL_Phishing_URL_Dataset.csv"
)

features = [
    "IsHTTPS",
    "NoOfOtherSpecialCharsInURL",
    "DegitRatioInURL",
    "URLLength",
    "LetterRatioInURL",
    "NoOfDegitsInURL",
    "SpacialCharRatioInURL",
    "NoOfSubDomain",
    "NoOfLettersInURL",
    "DomainLength"
]

samples = data.sample(100, random_state=42)

total_matches = 0
total_values = 0

for _, row in samples.iterrows():

    extracted = extract_url_features(row["URL"])

    for feature in features:

        dataset_value = round(float(row[feature]), 3)
        extracted_value = round(float(extracted.iloc[0][feature]), 3)

        total_values += 1

        if dataset_value == extracted_value:
            total_matches += 1

print("Total feature values:", total_values)
print("Matching values:", total_matches)
print("Mismatching values:", total_values - total_matches)

print(
    "Feature matching percentage:",
    round((total_matches / total_values) * 100, 2),
    "%"
)

print("\nMISMATCHES BY FEATURE")

mismatch_counts = {feature: 0 for feature in features}

for _, row in samples.iterrows():

    extracted = extract_url_features(row["URL"])

    for feature in features:

        dataset_value = round(float(row[feature]), 3)
        extracted_value = round(float(extracted.iloc[0][feature]), 3)

        if dataset_value != extracted_value:
            mismatch_counts[feature] += 1

for feature, count in mismatch_counts.items():
    print(feature, ":", count)

print("\nFIRST 5 URL MISMATCH EXAMPLES")

count = 0

for _, row in samples.iterrows():

    extracted = extract_url_features(row["URL"])

    differences = []

    for feature in features:

        original = round(float(row[feature]), 3)
        predicted = round(float(extracted.iloc[0][feature]), 3)

        if original != predicted:
            differences.append(
                f"{feature}: Dataset={original}, Extractor={predicted}"
            )

    if differences:
        print("\nURL:", row["URL"])

        for difference in differences:
            print(difference)

        count += 1

    if count == 5:
        break
