"""
Model Evaluation and Visualization Module.
Calculates Accuracy, Precision, Recall, AUC, and generates comparative plots
replicating Table 6, Table 7, Table 8, Figure 10, 11, 12, and 13 from Akinjole et al. (2024).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, roc_auc_score,
    roc_curve, confusion_matrix, classification_report
)
from typing import Dict, Any, Tuple
from pathlib import Path


def evaluate_models(
    models: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """
    Evaluates all trained models on test data.
    Separates results into Base Models (Table 6) and Ensemble Models (Table 7).
    """
    base_names = ["Random Forest", "Decision Tree", "SVM", "XGBoost", "ADABoost", "MLP"]
    ensemble_names = ["Voting A", "Voting B", "Stacking A", "Stacking B"]
    
    base_results = []
    ensemble_results = []
    detailed_eval = {}

    for name, model in models.items():
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            df_vals = model.decision_function(X_test)
            y_proba = (df_vals - df_vals.min()) / (df_vals.max() - df_vals.min() + 1e-8)
        else:
            y_proba = y_pred.astype(float)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_proba)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        cm = confusion_matrix(y_test, y_pred)

        detailed_eval[name] = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "auc": auc,
            "fpr": fpr,
            "tpr": tpr,
            "confusion_matrix": cm,
            "y_pred": y_pred,
            "y_proba": y_proba
        }

        row = {
            "Model": name,
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "AUC": round(auc, 4)
        }

        if name in base_names:
            base_results.append(row)
        elif name in ensemble_names:
            ensemble_results.append(row)

    df_base = pd.DataFrame(base_results)
    df_ensemble = pd.DataFrame(ensemble_results)
    return df_base, df_ensemble, detailed_eval


def plot_roc_curves(
    detailed_eval: Dict[str, Dict[str, Any]],
    model_names: list,
    title: str = "Receiver Operating Characteristic (ROC) Curve",
    save_path: Path = None
) -> plt.Figure:
    """
    Plots ROC curves for specified models (matching Figures 10 and 12 in the paper).
    """
    fig, ax = plt.subplots(figsize=(9, 7))
    palette = sns.color_palette("tab10", len(model_names))

    for i, name in enumerate(model_names):
        if name in detailed_eval:
            fpr = detailed_eval[name]["fpr"]
            tpr = detailed_eval[name]["tpr"]
            auc_val = detailed_eval[name]["auc"]
            ax.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.2f})", color=palette[i], lw=2)

    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label="Random Guess (AUC = 0.50)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300)
    return fig


def plot_metrics_comparison(
    df_metrics: pd.DataFrame,
    title: str = "Model Performance Comparison",
    save_path: Path = None
) -> plt.Figure:
    """
    Plots multi-metric grouped bar chart matching Figure 11.
    """
    df_melt = pd.melt(
        df_metrics,
        id_vars=["Model"],
        value_vars=["Accuracy", "Precision", "Recall", "AUC"],
        var_name="Metric",
        value_name="Score"
    )

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.barplot(data=df_melt, x="Model", y="Score", hue="Metric", palette="muted", ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.xticks(rotation=20, ha="right", fontsize=11)
    ax.legend(loc="lower right")
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300)
    return fig


def plot_auc_comparison(
    df_all: pd.DataFrame,
    title: str = "Model Performance Comparison (AUC)",
    save_path: Path = None
) -> plt.Figure:
    """
    Plots AUC bar chart across all models matching Figure 13.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(df_all["Model"], df_all["AUC"], color="#3c8d9e", edgecolor="black", width=0.6)
    
    # Highlight highest bar (Stacking A)
    best_idx = df_all["AUC"].idxmax()
    bars[best_idx].set_color("#1f77b4")
    bars[best_idx].set_edgecolor("black")
    bars[best_idx].set_linewidth(1.5)

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.015,
            f"{height * 100:.1f}%",
            ha="center", va="bottom", fontsize=9, fontweight="bold"
        )

    ax.set_ylim(0, 1.15)
    ax.set_ylabel("AUC Score", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.xticks(rotation=30, ha="right", fontsize=10)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300)
    return fig
