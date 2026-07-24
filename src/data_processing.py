import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def check_data_quality(df: pd.DataFrame):
    """
    Step 1: Always inspect your data before touching it.
    Check for missing values and duplicates.
    """
    print("="*50)
    print("DATA QUALITY CHECK")
    print("="*50)

    # Missing values per column
    missing = df.isnull().sum()
    print(f"\nMissing values: {missing.sum()}")

    # Duplicate rows
    duplicates = df.duplicated().sum()
    print(f"Duplicate rows: {duplicates}")

    # Class distribution
    print(f"\nClass distribution:")
    print(f"  Legitimate (0): {(df['label']==0).sum()}")
    print(f"  Phishing   (1): {(df['label']==1).sum()}")

    print(f"\nDataset shape: {df.shape}")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Step 2: Fix any data quality issues found above.
    """
    original_size = len(df)

    # Remove duplicates
    df = df.drop_duplicates()

    # Fill any missing values with column median
    df = df.fillna(df.median(numeric_only=True))

    print(f"\nRows removed (duplicates): {original_size - len(df)}")
    print(f"Clean dataset shape: {df.shape}")

    return df


def split_and_scale(df: pd.DataFrame):
    """
    Step 3: Separate features from label,
    split into train/test, and scale.
    """
    # Separate features (X) and label (y)
    X = df.drop(columns=['label'])
    y = df['label']

    print(f"\nFeature matrix X: {X.shape}")
    print(f"Label vector   y: {y.shape}")

    # Split: 80% train, 20% test
    # stratify=y ensures both splits have equal phishing/legit ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"Test set:  {X_test.shape[0]} samples")

    # Scale features
    # StandardScaler makes mean=0 and std=1 for each feature
    # Important for Logistic Regression and SVM
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    print("\nScaling complete.")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, list(X.columns)


if __name__ == "__main__":
    # Import from feature_extraction
    import sys
    sys.path.insert(0, '.')
    from src.feature_extraction import load_and_select_features

    # Run the full pipeline
    df = load_and_select_features("data/dataset_phishing.csv")

    check_data_quality(df)
    df = clean_data(df)
    X_train, X_test, y_train, y_test, scaler, feature_names = split_and_scale(df)

    print(f"\nFinal X_train shape: {X_train.shape}")
    print(f"Final X_test shape:  {X_test.shape}")