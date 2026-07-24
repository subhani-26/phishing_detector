import pandas as pd

# These are the features we'll use from the dataset
# We pick these because they cover URL structure, domain info, and content
SELECTED_FEATURES = [
    'length_url',
    'length_hostname',
    'ip',
    'nb_dots',
    'nb_hyphens',
    'nb_at',
    'nb_slash',
    'nb_percent',
    'nb_subdomains',
    'https_token',
    'phish_hints',
    'domain_age',
    'page_rank',
    'nb_redirection',
    'prefix_suffix'
]

def load_and_select_features(filepath: str) -> pd.DataFrame:
    """Load dataset and keep only selected features + label."""
    df = pd.read_csv(filepath)
    
    # Keep selected features + target column
    cols_to_keep = SELECTED_FEATURES + ['status']
    df = df[cols_to_keep]
    
    # Convert label: legitimate → 0, phishing → 1
    df['label'] = df['status'].map({'legitimate': 0, 'phishing': 1})
    df = df.drop(columns=['status'])
    
    print(f"Selected features: {len(SELECTED_FEATURES)}")
    print(f"Dataset shape: {df.shape}")
    print(f"\nFirst 3 rows:")
    print(df.head(3))
    
    return df

def explore_features(df: pd.DataFrame):
    """
    Shows basic statistics about each feature.
    Helps us understand the data before training.
    """
    print("\n" + "="*50)
    print("FEATURE STATISTICS BY CLASS")
    print("="*50)
    
    # Separate phishing and legitimate
    phishing   = df[df['label'] == 1]
    legitimate = df[df['label'] == 0]
    
    features = [col for col in df.columns if col != 'label']
    
    print(f"\n{'Feature':<20} {'Legit Mean':>12} {'Phish Mean':>12} {'Difference':>12}")
    print("-" * 58)
    
    for feature in features:
        legit_mean = legitimate[feature].mean()
        phish_mean = phishing[feature].mean()
        diff       = phish_mean - legit_mean
        marker = " <<" if abs(diff) > 0.5 else ""
        print(f"{feature:<20} {legit_mean:>12.2f} {phish_mean:>12.2f} {diff:>+12.2f}{marker}")


if __name__ == "__main__":
    df = load_and_select_features("data/dataset_phishing.csv")
    explore_features(df)


    