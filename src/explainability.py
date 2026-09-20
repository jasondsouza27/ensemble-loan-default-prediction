"""
Explainability Module (SHAP - SHapley Additive exPlanations).
Implements model interpretability for credit default prediction
matching Section 4.2, Figure 9, and Table A3 of Akinjole et al. (2024).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from typing import Dict, Any, List, Optional
from pathlib import Path


def get_tree_explainer(model, X_sample: pd.DataFrame):
    """
    Initializes a SHAP TreeExplainer for tree-based models (XGBoost / Random Forest).
    """
    try:
        explainer = shap.TreeExplainer(model)
        return explainer
    except Exception:
        # Fallback to KernelExplainer on a small background
        background = shap.kmeans(X_sample, 10)
        explainer = shap.KernelExplainer(model.predict_proba, background)
        return explainer


def compute_shap_values(explainer, X: pd.DataFrame):
    """
    Computes SHAP values for the given dataset.
    """
    shap_values = explainer(X)
    return shap_values


def plot_global_feature_importance(
    shap_values,
    feature_names: List[str],
    max_display: int = 25,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """
    Generates global feature importance plot matching Figure 9 in the paper.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Calculate mean absolute SHAP values
    if hasattr(shap_values, "values"):
        vals = np.abs(shap_values.values)
        if len(vals.shape) == 3: # multi-class or binary two outputs
            vals = vals[:, :, 1]
    else:
        vals = np.abs(shap_values)
        if isinstance(vals, list):
            vals = vals[1]
            
    mean_shap = np.mean(vals, axis=0)
    df_imp = pd.DataFrame({
        "Feature": feature_names,
        "Importance": mean_shap
    }).sort_values("Importance", ascending=True).tail(max_display)
    
    ax.barh(df_imp["Feature"], df_imp["Importance"], color="#2c3e50", height=0.7)
    ax.set_xlabel("Mean |SHAP value| (Impact on Default Prediction)", fontsize=11)
    ax.set_title(f"Top {max_display} Feature Importances (SHAP)", fontsize=13, fontweight="bold")
    ax.grid(axis="x", linestyle=":", alpha=0.6)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300)
    return fig


def explain_single_loan(
    explainer,
    single_applicant_df: pd.DataFrame,
    feature_names: List[str]
) -> pd.DataFrame:
    """
    Computes the SHAP breakdown for a single applicant to display in the UI / Table A3.
    """
    shap_exp = explainer(single_applicant_df)
    
    if hasattr(shap_exp, "values"):
        vals = shap_exp.values[0]
        if len(vals.shape) == 2:
            vals = vals[:, 1]
    else:
        vals = shap_exp[0]

    df_contrib = pd.DataFrame({
        "Feature": feature_names,
        "Applicant_Value": single_applicant_df.iloc[0].values,
        "SHAP_Contribution": vals
    }).sort_values(by="SHAP_Contribution", key=abs, ascending=False)
    
    # Interpret direction:
    # Positive SHAP increases default risk, Negative decreases default risk (or vice versa depending on target encoding)
    df_contrib["Impact"] = np.where(
        df_contrib["SHAP_Contribution"] > 0,
        "Increases Default Risk (+)",
        "Decreases Default Risk (-)"
    )
    
    return df_contrib
