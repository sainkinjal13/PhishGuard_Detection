import re
import pandas as pd
from urllib.parse import urlparse

def tokenize_url(url):
    
    parsed_url = urlparse(url)

    domain = parsed_url.netloc.lower()
    path = parsed_url.path.lower()
    query = parsed_url.query.lower()

    # Combine URL components
    text = domain + " " + path + " " + query

    # Split URL into tokens
    tokens = re.split(r'[^a-zA-Z0-9]+', text)

    # Remove empty tokens
    tokens = [token for token in tokens if token]

    return tokens



def extract_nlp_features(url):
    
    tokens = tokenize_url(url)

    features = {}

    # Token-based features
    features['NoOfTokens'] = len(tokens)

    features['AverageTokenLength'] = (
        sum(len(token) for token in tokens) / len(tokens)
        if tokens else 0
    )

    features['LongestTokenLength'] = (
        max(len(token) for token in tokens)
        if tokens else 0
    )

    features['ShortestTokenLength'] = (
        min(len(token) for token in tokens)
        if tokens else 0
    )

    features['NoOfNumericTokens'] = sum(
        token.isdigit() for token in tokens
    )

    features['NoOfAlphabeticTokens'] = sum(
        token.isalpha() for token in tokens
    )

    features['NoOfMixedTokens'] = sum(
        any(c.isalpha() for c in token) and
        any(c.isdigit() for c in token)
        for token in tokens
    )

    return pd.DataFrame([features])

test_url = "https://auth-document-shared.datingg-voice.workers.dev/"

print("Tokens:", tokenize_url(test_url))

features = extract_nlp_features(test_url)

print("\nExtracted NLP Features:")
print(features.T)

# Load dataset
data = pd.read_csv("dataset/PhiUSIIL_Phishing_URL_Dataset.csv")

# Extract NLP features for every URL
nlp_data = data['URL'].apply(
    lambda url: extract_nlp_features(str(url))
)

# Combine all extracted features
nlp_data = pd.concat(
    nlp_data.tolist(),
    ignore_index=True
)

# Add target label
nlp_data['label'] = data['label'].values

print("\nNLP Dataset Shape:")
print(nlp_data.shape)

print("\nFirst 5 Records:")
print(nlp_data.head())

print("\nMissing Values:")
print(nlp_data.isnull().sum())