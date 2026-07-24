import json
import joblib
import numpy as np
import pandas as pd

# Feature extraction from raw URL
# Since our model was trained on dataset features,
# we map URL characteristics to the same feature names

import re
import urllib.parse
from urllib.parse import urlparse


def extract_features_from_url(url: str) -> dict:
    """
    Extracts the same 15 features from a raw URL string
    that our dataset already had pre-extracted.

    This is what makes real-time prediction possible —
    user types a URL, we compute features on the fly.
    """
    # Clean input
    if not url.startswith(('http://', 'https://')):
        url = 'http://' + url

    parsed = urlparse(url)
    hostname = parsed.hostname or ''

    # 1. length_url
    length_url = len(url)

    # 2. length_hostname
    length_hostname = len(hostname)

    # 3. ip — does hostname look like an IP?
    ip_pattern = re.compile(r'(\d{1,3}\.){3}\d{1,3}')
    ip = 1 if ip_pattern.match(hostname) else 0

    # 4. nb_dots
    nb_dots = url.count('.')

    # 5. nb_hyphens
    nb_hyphens = url.count('-')

    # 6. nb_at
    nb_at = 1 if '@' in url else 0

    # 7. nb_slash
    nb_slash = url.replace('https://', '').replace('http://', '').count('/')

    # 8. nb_percent
    nb_percent = url.count('%')

    # 9. nb_subdomains
    parts = hostname.split('.')
    nb_subdomains = max(0, len(parts) - 2)

    # 10. https_token
    https_token = 1 if url.startswith('https') else 0

    # 11. phish_hints — suspicious keywords
    keywords = ['login', 'verify', 'secure', 'account',
                'update', 'banking', 'confirm', 'password',
                'signin', 'support', 'service', 'paypal',
                'ebay', 'amazon', 'microsoft', 'apple']
    phish_hints = sum(1 for k in keywords if k in url.lower())

    # 12. domain_age — unknown for real-time, use neutral value
    # (would need WHOIS lookup for real value)
    domain_age = 1000

    # 13. page_rank — unknown for real-time, use neutral value
    # (would need Google API for real value)
    page_rank = 2
    

    # 14. nb_redirection
    nb_redirection = url.count('//')  - 1  # subtract protocol //

    # 15. prefix_suffix
    prefix_suffix = 1 if '-' in hostname else 0

    return {
        'length_url':       length_url,
        'length_hostname':  length_hostname,
        'ip':               ip,
        'nb_dots':          nb_dots,
        'nb_hyphens':       nb_hyphens,
        'nb_at':            nb_at,
        'nb_slash':         nb_slash,
        'nb_percent':       nb_percent,
        'nb_subdomains':    nb_subdomains,
        'https_token':      https_token,
        'phish_hints':      phish_hints,
        'domain_age':       domain_age,
        'page_rank':        page_rank,
        'nb_redirection':   nb_redirection,
        'prefix_suffix':    prefix_suffix
    }


def get_risk_level(probability: float) -> tuple:
    """
    Converts raw probability into risk label + color.

    Returns: (risk_label, color_hex)
    """
    if probability < 0.30:
        return "LOW RISK",    "#22C55E"   # green
    elif probability < 0.60:
        return "MEDIUM RISK", "#F59E0B"   # orange
    elif probability < 0.80:
        return "HIGH RISK",   "#EF4444"   # red
    else:
        return "CRITICAL",    "#7F1D1D"   # dark red


def get_warnings(features: dict) -> list:
    """
    Returns human-readable warnings based on extracted features.
    Shown in the Streamlit app to explain the prediction.
    """
    warnings = []

    if features['ip']:
        warnings.append("Uses IP address instead of domain name")
    if features['nb_at']:
        warnings.append("Contains @ symbol - browser ignores everything before it")
    if features['nb_dots'] > 4:
        warnings.append(f"Too many dots ({features['nb_dots']}) - subdomain stacking")
    if features['length_url'] > 75:
        warnings.append(f"Very long URL ({features['length_url']} chars)")
    if features['phish_hints'] > 0:
        warnings.append(f"Contains {features['phish_hints']} suspicious keyword(s)")
    if features['prefix_suffix']:
        warnings.append("Hyphen found in domain - common phishing pattern")
    if not features['https_token']:
        warnings.append("No HTTPS - connection is not encrypted")
    if features['nb_subdomains'] > 2:
        warnings.append(f"Many subdomains ({features['nb_subdomains']})")
    if features['nb_redirection'] > 1:
        warnings.append("Multiple redirections detected in URL")

    if not warnings:
        warnings.append("No suspicious patterns detected in URL structure")

    return warnings


def predict(url: str) -> dict:
    """
    Master prediction function.
    Loads saved model, extracts features, returns full result.
    """
    # Load saved model files
    model         = joblib.load("models/best_model.pkl")
    scaler        = joblib.load("models/scaler.pkl")
    feature_names = json.load(open("models/feature_names.json"))

    # Extract features from URL
    features = extract_features_from_url(url)

    # Build feature vector in correct order
    vector = pd.DataFrame([features])[feature_names]

    # Scale
    vector_scaled = scaler.transform(vector)

    # Predict
    label       = model.predict(vector_scaled)[0]
    probability = model.predict_proba(vector_scaled)[0][1]  # phishing probability

    risk_label, risk_color = get_risk_level(probability)
    warnings               = get_warnings(features)

    return {
        'url':        url,
        'prediction': 'Phishing' if label == 1 else 'Legitimate',
        'label':      int(label),
        'confidence': round(probability * 100, 2),
        'risk_level': risk_label,
        'risk_color': risk_color,
        'features':   features,
        'warnings':   warnings
    }


if __name__ == "__main__":
    # Quick test
    test_urls = [
        "https://www.google.com",
        "http://paypal-secure-login.com/verify@fake.com",
        "http://192.168.1.1/login.php",
    ]

    for url in test_urls:
        result = predict(url)
        print(f"\nURL: {url}")
        print(f"  Verdict:    {result['prediction']}")
        print(f"  Confidence: {result['confidence']}% phishing probability")
        print(f"  Risk:       {result['risk_level']}")
        print(f"  Warnings:   {result['warnings']}")
        
        