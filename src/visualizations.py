import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Make sure output folder exists
os.makedirs("visualizations", exist_ok=True)


def plot_model_comparison(results: dict):
    """
    Bar chart comparing all 4 models across 4 metrics.
    """
    model_names = list(results.keys())
    metrics     = ['accuracy', 'precision', 'recall', 'f1_score']
    labels      = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    colors      = ['#3B82F6', '#EF4444', '#22C55E', '#F59E0B']

    x      = np.arange(len(model_names))
    width  = 0.18
    offset = np.linspace(-1.5, 1.5, 4) * width

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor('#F8FAFC')
    ax.set_facecolor('#F8FAFC')

    for i, (metric, label, color) in enumerate(zip(metrics, labels, colors)):
        values = [results[m][metric] for m in model_names]
        bars   = ax.bar(x + offset[i], values, width,
                        label=label, color=color, alpha=0.85,
                        edgecolor='white', linewidth=1.2)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.4,
                    f'{val:.1f}%',
                    ha='center', va='bottom',
                    fontsize=8, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=10)
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_title('Model Comparison — Phishing Detection',
                 fontsize=14, fontweight='bold')
    ax.set_ylim(0, 110)
    ax.legend(fontsize=10)
    ax.yaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig('visualizations/model_comparison.png', dpi=150)
    plt.close()
    print("Saved: model_comparison.png")


def plot_confusion_matrix(cm_data: list, model_name: str):
    """
    Heatmap of confusion matrix for one model.
    """
    cm     = np.array(cm_data)
    labels = ['Legitimate', 'Phishing']

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor('#F8FAFC')

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels,
                linewidths=2, linecolor='white',
                cbar=False, ax=ax,
                annot_kws={'size': 14, 'weight': 'bold'})

    ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
    ax.set_ylabel('Actual',    fontsize=12, fontweight='bold')
    ax.set_title(f'Confusion Matrix\n{model_name}',
                 fontsize=13, fontweight='bold')

    safe_name = model_name.replace(' ', '_').lower()
    plt.tight_layout()
    plt.savefig(f'visualizations/cm_{safe_name}.png', dpi=150)
    plt.close()
    print(f"Saved: cm_{safe_name}.png")


def plot_feature_importance(model, feature_names: list):
    """
    Horizontal bar chart of Random Forest feature importances.
    Shows which features matter most for detection.
    """
    if not hasattr(model, 'feature_importances_'):
        print("Model does not support feature importances.")
        return

    importances = model.feature_importances_
    indices     = np.argsort(importances)  # sort ascending for horizontal bar

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor('#F8FAFC')
    ax.set_facecolor('#F8FAFC')

    colors = ['#3B82F6' if importances[i] > np.mean(importances)
              else '#94A3B8' for i in indices]

    ax.barh(range(len(indices)),
            importances[indices],
            color=colors, edgecolor='white')
    

    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices], fontsize=10)
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_title('Feature Importance — Random Forest',
                 fontsize=14, fontweight='bold')
    ax.xaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig('visualizations/feature_importance.png', dpi=150)
    plt.close()
    print("Saved: feature_importance.png")


def plot_class_distribution(y):
    """
    Pie chart showing phishing vs legitimate split.
    """
    counts = [int((y == 0).sum()), int((y == 1).sum())]
    labels = ['Legitimate', 'Phishing']
    colors = ['#22C55E', '#EF4444']

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor('#F8FAFC')

    wedges, texts, autotexts = ax.pie(
        counts, labels=labels, colors=colors,
        autopct='%1.1f%%', startangle=90,
        explode=(0.05, 0.05), shadow=True,
        textprops={'fontsize': 12}
    )
    for at in autotexts:
        at.set_fontweight('bold')

    ax.set_title('Dataset Class Distribution',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('visualizations/class_distribution.png', dpi=150)
    plt.close()
    print("Saved: class_distribution.png")


if __name__ == "__main__":
    import joblib
    import sys
    sys.path.insert(0, '.')
    from src.feature_extraction import load_and_select_features
    from src.data_processing import clean_data, split_and_scale

    # Load results
    with open("models/evaluation_results.json") as f:
        results = json.load(f)

    # Load model and data
    model         = joblib.load("models/best_model.pkl")
    feature_names = json.load(open("models/feature_names.json"))

    df       = load_and_select_features("data/dataset_phishing.csv")
    df       = clean_data(df)
    X_train, X_test, y_train, y_test, scaler, _ = split_and_scale(df)

    print("\nGenerating visualizations...")

    # 1. Model comparison
    plot_model_comparison(results)

    # 2. Confusion matrix for each model
    for name, metrics in results.items():
        plot_confusion_matrix(metrics['confusion_matrix'], name)

    # 3. Feature importance
    plot_feature_importance(model, feature_names)

    # 4. Class distribution
    plot_class_distribution(y_test)

    print("\nAll charts saved to visualizations/")
