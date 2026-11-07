import pandas as pd

data = pd.read_csv("dataset/PhiUSIIL_Phishing_URL_Dataset.csv")

# Select a simple URL from the dataset
sample = data[
    data["URL"].str.startswith("https://", na=False)
].iloc[0]

print("Selected URL:")
print(sample["URL"])

print("\nOriginal dataset features:")

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

print(sample[features])