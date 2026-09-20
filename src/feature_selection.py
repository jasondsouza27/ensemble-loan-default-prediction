"""
Feature Selection Module for Loan Default Risk Prediction.
Implements Recursive Feature Elimination with Cross-Validation (RFECV)
using XGBoost optimized on recall, per Section 3.5 and Figure 8 of Akinjole et al. (2024).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFECV, RFE
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from typing import Tuple, List, Optional
from pathlib import Path


def run_rfecv_selection(
    X: pd.DataFrame,
    y: pd.Series,
    min_features_to_select: int = 20,
    target_features: int = 48,
    cv_folds: int = 3,
    step: int = 1,
    random_state: int = 42
) -> Tuple[List[str], pd.DataFrame, Optional[plt.Figure]]:
    """
    Executes RFECV using XGBoost with recall scoring to select the optimal features.
    Matches Section 3.5 and Figure 8.
    """
    print(f"\n[Feature Selection] Running RFECV with XGBoost (Recall scorer, step={step})...")
    
    estimator = XGBClassifier(
        n_estimators=50,
        max_depth=5,
        learning_rate=0.1,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1
    )
    
    n_initial_features = X.shape[1]
    
    # If initial features <= target_features, keep all or rank via RFE
    if n_initial_features <= target_features:
        print(f"[Feature Selection] Initial features ({n_initial_features}) <= target ({target_features}). Using all features.")
        estimator.fit(X, y)
        importances = pd.DataFrame({
            "Feature": X.columns,
            "Importance": estimator.feature_importances_
        }).sort_values("Importance", ascending=False)
        return list(X.columns), importances, None
        
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    rfecv = RFECV(
        estimator=estimator,
        step=step,
        cv=cv,
        scoring="recall",
        min_features_to_select=min_features_to_select,
        n_jobs=-1
    )
    
    rfecv.fit(X, y)
    
    selected_features = list(X.columns[rfecv.support_])
    print(f"[Feature Selection] Optimal number of features selected: {rfecv.n_features_}")
    
    # Compute feature importances of selected features
    estimator.fit(X[selected_features], y)
    importances = pd.DataFrame({
        "Feature": selected_features,
        "Importance": estimator.feature_importances_
    }).sort_values("Importance", ascending=False)
    
    # Plotting RFECV curve matching Figure 8
    fig, ax = plt.subplots(figsize=(10, 5))
    n_scores = len(rfecv.cv_results_['mean_test_score'])
    x_range = range(min_features_to_select, min_features_to_select + n_scores * step, step)
    ax.plot(x_range, rfecv.cv_results_['mean_test_score'], 'b-o', markersize=3, label="Mean CV Score (Recall)")
    ax.axvline(x=rfecv.n_features_, color='red', linestyle='--', label=f"Optimal Features = {rfecv.n_features_}")
    ax.set_xlabel("Number of Features Selected")
    ax.set_ylabel("Cross-Validation Score (Recall)")
    ax.set_title("Number of Features vs. Cross-Validation Score (RFECV)")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    
    return selected_features, importances, fig
