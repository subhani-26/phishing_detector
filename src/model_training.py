import os
import sys
import json
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, confusion_matrix
)

sys.path.insert(0, '.')
from src.feature_extraction import load_and_select_features
from src.data_processing import check_data_quality, clean_data, split_and_scale


def get_models() -> dict:
    """
    Define all 4 models we want to compare.
    Each has sensible default hyperparameters.
    """
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=100,   # 100 decision trees vote together
            max_depth=10,       # prevents overfitting
            random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8,        # shallow tree = less overfitting
            random_state=42
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000,      # enough iterations to converge
            random_state=42
        ),
        "Support Vector Machine": CalibratedClassifierCV(
            SVC(kernel='rbf', random_state=42),
            ensemble=False
        )
    }


def train_and_evaluate(X_train, X_test, y_train, y_test) -> tuple:
    """
    Train all 4 models and evaluate each one.
    Returns trained models and their results.
    """
    models  = get_models()
    results = {}
    trained = {}

    print("\n" + "="*55)
    print("MODEL TRAINING & EVALUATION")
    print("="*55)

    for name, model in models.items():
        print(f"\nTraining: {name}...")

        # Train
        model.fit(X_train, y_train)

        # Predict on test set
        y_pred = model.predict(X_test)

        # Calculate metrics
        acc  = accuracy_score(y_test, y_pred)  * 100
        prec = precision_score(y_test, y_pred) * 100
        rec  = recall_score(y_test, y_pred)    * 100
        f1   = f1_score(y_test, y_pred)        * 100
        cm   = confusion_matrix(y_test, y_pred)

        results[name] = {
            'accuracy':         round(acc,  2),
            'precision':        round(prec, 2),
            'recall':           round(rec,  2),
            'f1_score':         round(f1,   2),
            'confusion_matrix': cm.tolist()
        }
        trained[name] = model

        # Print results
        print(f"  Accuracy:  {acc:.2f}%")
        print(f"  Precision: {prec:.2f}%")
        print(f"  Recall:    {rec:.2f}%")
        print(f"  F1-Score:  {f1:.2f}%")

    # Find best model by F1-score
    best_name = max(results, key=lambda x: results[x]['f1_score'])

    print(f"\n{'='*55}")
    print(f"BEST MODEL: {best_name}")
    print(f"F1-Score:   {results[best_name]['f1_score']}%")
    print(f"{'='*55}")

    return trained, results, best_name


def save_best_model(model, scaler, feature_names, model_name):
    """
    Save the best model + scaler to disk.
    These files are loaded later by the Streamlit app.
    """
    os.makedirs("models", exist_ok=True)

    joblib.dump(model,  "models/best_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")

    with open("models/feature_names.json", "w") as f:
        json.dump(feature_names, f)

    with open("models/model_info.json", "w") as f:
        json.dump({'model_name': model_name}, f)

    print(f"\nModel saved to: models/best_model.pkl")
    print(f"Scaler saved to: models/scaler.pkl")


if __name__ == "__main__":
    # Full pipeline
    df = load_and_select_features("data/dataset_phishing.csv")
    clean_data_df = clean_data(df)
    X_train, X_test, y_train, y_test, scaler, feature_names = split_and_scale(clean_data_df)

    trained_models, results, best_name = train_and_evaluate(
        X_train, X_test, y_train, y_test
    )

    save_best_model(
        trained_models[best_name],
        scaler,
        feature_names,
        best_name
    )

    
    # Save results for Streamlit dashboard
    import json
    with open("models/evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nEvaluation results saved.")
    