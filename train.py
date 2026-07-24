# train.py
# Run this file to retrain the model from scratch
# Usage: python train.py

import sys
sys.path.insert(0, '.')

from src.feature_extraction  import load_and_select_features
from src.data_processing     import check_data_quality, clean_data, split_and_scale
from src.model_training      import train_and_evaluate, save_best_model
from src.visualizations      import (plot_model_comparison, plot_confusion_matrix,
                                      plot_feature_importance, plot_class_distribution)
import json

def main():
    print("=" * 55)
    print("  PHISHGUARD - FULL TRAINING PIPELINE")
    print("=" * 55)

    # Step 1: Load data
    print("\n[STEP 1] Loading dataset...")
    df = load_and_select_features("data/dataset_phishing.csv")

    # Step 2: Check quality
    print("\n[STEP 2] Checking data quality...")
    check_data_quality(df)

    # Step 3: Clean
    print("\n[STEP 3] Cleaning data...")
    df = clean_data(df)

    # Step 4: Split and scale
    print("\n[STEP 4] Splitting and scaling...")
    X_train, X_test, y_train, y_test, scaler, feature_names = split_and_scale(df)

    # Step 5: Train all models
    print("\n[STEP 5] Training models...")
    trained_models, results, best_name = train_and_evaluate(
        X_train, X_test, y_train, y_test
    )

    # Step 6: Save best model
    print("\n[STEP 6] Saving best model...")
    save_best_model(
        trained_models[best_name],
        scaler,
        feature_names,
        best_name
    )

    # Save evaluation results
    with open("models/evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Step 7: Generate visualizations
    print("\n[STEP 7] Generating visualizations...")
    plot_model_comparison(results)
    plot_class_distribution(y_test)

    for name, metrics in results.items():
        plot_confusion_matrix(metrics['confusion_matrix'], name)

    best_model = trained_models[best_name]
    plot_feature_importance(best_model, feature_names)

    # Final summary
    print("\n" + "=" * 55)
    print("TRAINING COMPLETE - SUMMARY")
    print("=" * 55)
    print(f"\n{'Model':<30} {'Accuracy':>10} {'F1':>10}")
    print("-" * 52)
    for name, m in results.items():
        marker = " <-- BEST" if name == best_name else ""
        print(f"{name:<30} {m['accuracy']:>9.2f}%  {m['f1_score']:>8.2f}%{marker}")

    print("\n Next steps:")
    print("  Run app  : streamlit run app.py")
    print("=" * 55)


if __name__ == "__main__":
    main()