
import pandas as pd
import re
import joblib
from pathlib import Path
from urllib.parse import urlparse

# Locate the main project directory
BASE_DIR = Path(__file__).resolve().parent.parent


def extract_url_features(url):

    if not re.match(r"^https?://", url, re.IGNORECASE):
        url = "https://" + url

    parsed_url = urlparse(url)

    domain = parsed_url.netloc.lower().split("@")[-1].split(":")[0]

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

    features['IsDomainIP'] = int(
        bool(re.fullmatch(r"\d{1,3}(?:\.\d{1,3}){3}", domain))
    )

    features['IsHTTPS'] = int(parsed_url.scheme.lower() == "https")

    features['NoOfSubDomain'] = max(
        len(domain.split(".")) - 2, 0
    )

    features['NoOfLettersInURL'] = max(
        sum(char.isalpha() for char in clean_url) - 1, 0
    )

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

    expected_features = joblib.load(
        BASE_DIR / "models" / "url_features.pkl"
    )

    result = pd.DataFrame([features])

    result = result[expected_features]

    return result
